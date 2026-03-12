import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles

from dashscope_stt import DashScopeSTT
from dashscope_tts import DashScopeTTS
from dashscope_prompts import DASHSCOPE_TTS_SYSTEM_PROMPT
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)

load_dotenv()

STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


system_prompt = f"""
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

{DASHSCOPE_TTS_SYSTEM_PROMPT}
"""

agent = create_agent(
    model="openai:qwen-turbo",
    tools=[add_to_order, confirm_order],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)

    Uses DashScope SenseVoice for speech-to-text.
    Supports Chinese, English, Japanese, Cantonese, and Korean.
    """
    stt = DashScopeSTT(sample_rate=16000, language="auto")

    async def send_audio():
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()

    send_task = asyncio.create_task(send_audio())

    try:
        async for event in stt.receive_events():
            yield event
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Agent Responses)

    Uses DashScope Qwen LLM to process user input and generate responses.
    """
    thread_id = str(uuid4())

    async for event in event_stream:
        yield event

        if event.type == "stt_output":
            try:
                inputs = {"messages": [HumanMessage(content=event.transcript)]}
                stream = agent.astream(inputs, {"configurable": {"thread_id": thread_id}})

                async for message in stream:
                    if isinstance(message, dict):
                        text = message.get("content") or message.get("text") or ""

                        if not text:
                            msgs = message.get("messages", [])
                            if not msgs:
                                msgs = message.get("model", {}).get("messages", [])
                            if msgs:
                                last_msg = msgs[-1] if isinstance(msgs, list) else msgs
                                if hasattr(last_msg, "content"):
                                    text = last_msg.content
                                elif isinstance(last_msg, dict):
                                    text = last_msg.get("content") or last_msg.get("text") or ""
                                else:
                                    text = str(last_msg)

                        if text:
                            yield AgentChunkEvent.create(text)

                        tool_calls = message.get("tool_calls") or []
                        for tool_call in tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )
                        continue

                    if isinstance(message, AIMessage):
                        content = message.content
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict):
                                    text_parts.append(item.get("text", ""))
                                elif hasattr(item, "get"):
                                    text_parts.append(str(item.get("text", "")))
                                else:
                                    text_parts.append(str(item))
                            full_text = "".join(text_parts)
                        else:
                            full_text = str(content) if content else ""

                        if full_text:
                            yield AgentChunkEvent.create(full_text)

                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                yield ToolCallEvent.create(
                                    id=tool_call.get("id", str(uuid4())),
                                    name=tool_call.get("name", "unknown"),
                                    args=tool_call.get("args", {}),
                                )

                    elif isinstance(message, ToolMessage):
                        yield ToolResultEvent.create(
                            tool_call_id=getattr(message, "tool_call_id", ""),
                            name=getattr(message, "name", "unknown"),
                            result=str(message.content) if message.content else "",
                        )

                yield AgentEndEvent.create()

            except Exception as e:
                print(f"[ERROR] Agent stream: {e}")
                yield AgentEndEvent.create()


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Audio)

    Uses DashScope CosyVoice for text-to-speech.
    """
    tts = DashScopeTTS(voice="longhua", sample_rate=24000)
    buffer: list[str] = []
    
    async for event in event_stream:
        yield event
        
        if event.type == "agent_chunk":
            buffer.append(event.text)
            
        elif event.type == "agent_end":
            text_to_speak = "".join(buffer)
            buffer = []
            
            if text_to_speak.strip():
                await tts.send_text(text_to_speak)
                async for tts_event in tts.receive_events():
                    yield tts_event
    
    await tts.close()


pipeline = (
    RunnableGenerator(_stt_stream)
    | RunnableGenerator(_agent_stream)
    | RunnableGenerator(_tts_stream)
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        try:
            while True:
                data = await websocket.receive_bytes()
                yield data
        except WebSocketDisconnect:
            pass

    try:
        output_stream = pipeline.atransform(websocket_audio_stream())
        async for event in output_stream:
            await websocket.send_json(event_to_dict(event))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error in websocket handler: {e}")
        raise


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
