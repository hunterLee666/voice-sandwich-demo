import WebSocket from "ws";
import { writableIterator } from "../utils";
import type { VoiceAgentEvent } from "../types";

interface DashScopeTTSOptions {
  apiKey?: string;
  voice?: string; // longhua, shujing, longcheng, longhua_customer_service
  format?: "pcm" | "mp3" | "wav";
  sampleRate?: number;
}

/**
 * DashScope CosyVoice 语音合成客户端
 *
 * 支持多种中文声音和情感风格，适合中文对话场景。
 * 使用 WebSocket 流式接口进行实时语音合成。
 */
export class DashScopeTTS {
  apiKey: string;
  voice: string;
  format: string;
  sampleRate: number;

  protected _bufferIterator = writableIterator<VoiceAgentEvent.TTSChunk>();
  protected _connectionPromise: Promise<WebSocket> | null = null;

  protected get _connection(): Promise<WebSocket> {
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      const url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/cosyvoice-v1";

      const ws = new WebSocket(url, {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "X-DashScope-DataInspection": "{\"input\":\"\"}",
        },
        protocol: "dashscope-protocol",
      });

      ws.on("open", () => {
        resolve(ws);
      });

      ws.on("message", (data: WebSocket.RawData) => {
        try {
          const message = JSON.parse(data.toString());
          const header = message.header || {};
          const payload = message.payload || {};
          const name = header.name || "";

          if (name === "GeneratedAudio") {
            const audioBase64 = payload.audio || "";
            if (audioBase64) {
              this._bufferIterator.push({
                type: "tts_chunk",
                audio: audioBase64,
                ts: Date.now(),
              });
            }
          } else if (name === "TaskCompleted") {
            // 合成完成
          } else if (name === "Error") {
            const errorMessage = header.message || "Unknown error";
            console.error(`DashScope TTS error: ${errorMessage}`);
          }
        } catch (error) {
          console.error("DashScope TTS JSON decode error:", error);
        }
      });

      ws.on("error", (error) => {
        this._bufferIterator.cancel();
        reject(error);
      });

      ws.on("close", () => {
        this._connectionPromise = null;
      });
    });

    return this._connectionPromise;
  }

  constructor(options: DashScopeTTSOptions = {}) {
    this.apiKey = options.apiKey || process.env.DASHSCOPE_API_KEY || "";
    this.voice = options.voice || "longhua";
    this.format = options.format || "pcm";
    this.sampleRate = options.sampleRate || 24000;

    if (!this.apiKey) {
      throw new Error("DashScope API key is required");
    }
  }

  async sendText(text: string): Promise<void> {
    if (!text || !text.trim()) {
      return;
    }

    const conn = await this._connection;
    if (conn.readyState === WebSocket.OPEN) {
      const message = {
        header: {
          action: "run",
          task: "tts",
          streaming: "duplex",
        },
        payload: {
          task_group: "audio",
          task: "tts",
          function: "speech_synthesis",
          model: "cosyvoice-v1",
          parameters: {
            voice: this.voice,
            format: this.format,
            sample_rate: this.sampleRate,
          },
          input: {
            text: text,
          },
        },
      };
      conn.send(JSON.stringify(message));
    }
  }

  async *receiveEvents(): AsyncGenerator<VoiceAgentEvent.TTSChunk> {
    yield* this._bufferIterator;
  }

  async close(): Promise<void> {
    if (this._connectionPromise) {
      const ws = await this._connectionPromise;
      ws.close();
    }
  }
}
