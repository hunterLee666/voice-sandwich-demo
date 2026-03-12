import WebSocket from "ws";
import { writableIterator } from "../utils";
import type { VoiceAgentEvent } from "../types";

interface DashScopeSTTOptions {
  apiKey?: string;
  sampleRate?: number;
  format?: string;
  language?: string; // auto, zh, en, ja, yue, ko
}

/**
 * DashScope SenseVoice 语音识别客户端
 *
 * 支持多种语言的实时语音识别，包括中文、英文、日语、粤语等。
 * 使用 WebSocket 流式接口进行实时识别。
 */
export class DashScopeSTT {
  apiKey: string;
  sampleRate: number;
  format: string;
  language: string;

  protected _bufferIterator = writableIterator<VoiceAgentEvent.STTEvent>();
  protected _connectionPromise: Promise<WebSocket> | null = null;

  protected get _connection(): Promise<WebSocket> {
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      const url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/sensevoice-v1";

      const ws = new WebSocket(url, {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "X-DashScope-DataInspection": "{\"input\":\"\"}",
        },
        protocol: "dashscope-protocol",
      });

      ws.on("open", () => {
        // 发送初始化消息
        const initMessage = {
          header: {
            action: "run",
            task: "asr",
            streaming: "duplex",
          },
          payload: {
            task_group: "audio",
            task: "asr",
            function: "recognition",
            model: "paraformer-v2",
            parameters: {
              format: this.format,
              sample_rate: this.sampleRate,
              language: this.language,
            },
          },
        };
        ws.send(JSON.stringify(initMessage));
        resolve(ws);
      });

      ws.on("message", (data: WebSocket.RawData) => {
        try {
          const message = JSON.parse(data.toString());
          const header = message.header || {};
          const payload = message.payload || {};
          const name = header.name || "";

          if (name === "SentenceBegin") {
            const result = payload.result || {};
            const text = result.text || "";
            if (text) {
              this._bufferIterator.push({
                type: "stt_chunk",
                transcript: text,
                ts: Date.now(),
              });
            }
          } else if (name === "TranscriptionResultChanged") {
            const result = payload.result || {};
            const text = result.text || "";
            if (text) {
              this._bufferIterator.push({
                type: "stt_chunk",
                transcript: text,
                ts: Date.now(),
              });
            }
          } else if (name === "SentenceEnd") {
            const result = payload.result || {};
            const text = result.text || "";
            if (text) {
              this._bufferIterator.push({
                type: "stt_output",
                transcript: text,
                ts: Date.now(),
              });
            }
          } else if (name === "Error") {
            const errorMessage = header.message || "Unknown error";
            console.error(`DashScope STT error: ${errorMessage}`);
          }
        } catch (error) {
          console.error("DashScope STT JSON decode error:", error);
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

  constructor(options: DashScopeSTTOptions = {}) {
    this.apiKey = options.apiKey || process.env.DASHSCOPE_API_KEY || "";
    this.sampleRate = options.sampleRate || 16000;
    this.format = options.format || "pcm";
    this.language = options.language || "auto";

    if (!this.apiKey) {
      throw new Error("DashScope API key is required");
    }
  }

  async sendAudio(buffer: Uint8Array): Promise<void> {
    const conn = await this._connection;
    // DashScope 接收 base64 编码的音频数据
    const audioBase64 = Buffer.from(buffer).toString("base64");
    const message = {
      payload: audioBase64,
    };
    conn.send(JSON.stringify(message));
  }

  async *receiveEvents(): AsyncGenerator<VoiceAgentEvent.STTEvent> {
    yield* this._bufferIterator;
  }

  async close(): Promise<void> {
    if (this._connectionPromise) {
      const ws = await this._connectionPromise;
      ws.close();
    }
  }
}
