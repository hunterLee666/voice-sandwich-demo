"""
DashScope Text-to-Speech Streaming (CosyVoice)

使用阿里云 DashScope 的 CosyVoice 语音合成服务进行流式文本转语音。
支持中文、英文等多种语言，提供自然流畅的中文语音。

Input: Text strings
Output: TTS events (tts_chunk for audio chunks)

API 文档: https://help.aliyun.com/zh/dashscope/developer-reference/cosyvoice-api-details
"""

import json
import os
import uuid
from typing import AsyncIterator, Literal, Optional

import websockets

from events import TTSChunkEvent


class DashScopeTTS:
    """DashScope CosyVoice 语音合成客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "longhua",
        format: Literal["pcm", "mp3", "wav"] = "pcm",
        sample_rate: int = 24000,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required")

        self.voice = voice
        self.format = format
        self.sample_rate = sample_rate
        self._task_id = str(uuid.uuid4())
        self._text: Optional[str] = None

    async def send_text(self, text: Optional[str]) -> None:
        """发送文本进行语音合成"""
        if text is None or not text.strip():
            return
        self._text = text

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        """接收语音合成事件"""
        if not self._text:
            return

        text = self._text
        url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            ws = await websockets.connect(url, additional_headers=headers)

            await ws.send(json.dumps({
                "header": {
                    "action": "run-task",
                    "task_id": self._task_id,
                    "streaming": "duplex"
                },
                "payload": {
                    "task_group": "audio",
                    "task": "tts",
                    "function": "SpeechSynthesizer",
                    "model": "cosyvoice-v1",
                    "parameters": {
                        "text_type": "PlainText",
                        "voice": self.voice,
                        "format": self.format,
                        "sample_rate": self.sample_rate,
                        "rate": 1.0,
                    },
                    "input": {}
                }
            }))
            
            async for raw_message in ws:
                if isinstance(raw_message, bytes):
                    audio_chunk = bytes(raw_message)
                    if audio_chunk:
                        yield TTSChunkEvent.create(audio_chunk)
                    continue

                try:
                    message = json.loads(raw_message)
                    header = message.get("header", {})
                    event = header.get("event")

                    if event == "task-started":
                        await ws.send(json.dumps({
                            "header": {
                                "action": "continue-task",
                                "task_id": self._task_id,
                                "streaming": "duplex"
                            },
                            "payload": {"input": {"text": text}}
                        }))
                        await ws.send(json.dumps({
                            "header": {
                                "action": "finish-task",
                                "task_id": self._task_id,
                                "streaming": "duplex"
                            },
                            "payload": {"input": {}}
                        }))

                    elif event in ("task-finished", "task-failed"):
                        break

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            print(f"DashScope TTS Error: {e}")
        finally:
            self._text = None

    async def close(self) -> None:
        """关闭连接"""
        self._text = None
