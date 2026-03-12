"""
DashScope Real-Time Streaming STT (Speech-to-Text)

使用阿里云 DashScope 的 SenseVoice 语音识别服务进行流式语音转文字。
支持 WebSocket 流式接口，适用于实时对话场景。

Input: PCM 16-bit audio buffer (bytes)
Output: STT events (stt_chunk for partials, stt_output for final transcripts)

API 文档: https://help.aliyun.com/zh/dashscope/developer-reference/api-sensevoice
"""

import asyncio
import json
import os
import uuid
from typing import AsyncIterator, Optional

import websockets

from events import STTChunkEvent, STTEvent, STTOutputEvent


class DashScopeSTT:
    """
    DashScope SenseVoice 语音识别客户端

    支持多种语言的实时语音识别，包括中文、英文、日语、粤语等。
    使用 WebSocket 流式接口进行实时识别。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        format: str = "pcm",
        language: str = "auto",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required")

        self.sample_rate = sample_rate
        self.format = format
        self.language = language
        self._ws = None
        self._connection_signal = asyncio.Event()
        self._ready_signal = asyncio.Event()
        self._close_signal = asyncio.Event()
        self._context_id = None

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """接收语音识别事件"""
        import contextlib

        while not self._close_signal.is_set():
            _, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._close_signal.wait()),
                    asyncio.create_task(self._connection_signal.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            with contextlib.suppress(asyncio.CancelledError):
                for task in pending:
                    task.cancel()

            if self._close_signal.is_set():
                break

            if self._ws and self._ws.close_code is None:
                self._connection_signal.clear()
                try:
                    async for raw_message in self._ws:
                        try:
                            message = json.loads(raw_message)
                            header = message.get("header", {})
                            payload = message.get("payload", {})
                            event_type = header.get("event", "")

                            if event_type == "task-started":
                                self._ready_signal.set()

                            elif event_type == "result-generated":
                                output = payload.get("output", {})
                                sentence = output.get("sentence", {})
                                text = sentence.get("text", "")
                                sentence_end = sentence.get("sentence_end", False)

                                if text:
                                    if sentence_end:
                                        yield STTOutputEvent.create(text)
                                    else:
                                        yield STTChunkEvent.create(text)

                            elif event_type == "task-finished":
                                break

                        except json.JSONDecodeError:
                            continue

                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    print(f"DashScope STT error: {e}")

    async def send_audio(self, audio_chunk: bytes) -> None:
        """发送音频数据 - 使用二进制帧"""
        ws = await self._ensure_connection()
        await asyncio.wait_for(self._ready_signal.wait(), timeout=5.0)
        await ws.send(audio_chunk)

    async def close(self) -> None:
        """关闭连接"""
        if self._ws and self._ws.close_code is None:
            await self._ws.close()
        self._ws = None
        self._close_signal.set()

    async def _ensure_connection(self):
        """确保 WebSocket 连接已建立"""
        if self._close_signal.is_set():
            raise RuntimeError(
                "DashScopeSTT tried establishing a connection after it was closed"
            )
        if self._ws and self._ws.close_code is None:
            return self._ws

        url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/paraformer-realtime-v2"

        self._context_id = str(uuid.uuid4())
        parameters = {
            "format": self.format,
            "sample_rate": self.sample_rate,
            "language": self.language,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-DataInspection": "{\"input\":\"\"}",
        }

        init_message = {
            "header": {
                "action": "run-task",
                "task": "asr",
                "streaming": "duplex",
                "task_id": self._context_id,
            },
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": "paraformer-realtime-v2",
                "parameters": parameters,
                "input": "",
            }
        }

        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            subprotocols=["dashscope-protocol"],
        )

        await self._ws.send(json.dumps(init_message))

        self._connection_signal.set()
        return self._ws
