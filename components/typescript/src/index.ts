import "dotenv/config";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { createAgent, AIMessage, ToolMessage } from "langchain";
import path from "node:path";
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { cors } from "hono/cors";
import { createNodeWebSocket } from "@hono/node-ws";
import type { WSContext } from "hono/ws";
import type WebSocket from "ws";
import { iife, writableIterator } from "./utils";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { v4 as uuidv4 } from "uuid";
import { DASHSCOPE_TTS_SYSTEM_PROMPT, DashScopeTTS } from "./dashscope";
import { DashScopeSTT } from "./dashscope/stt";
import type { VoiceAgentEvent } from "./types";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const STATIC_DIR = path.join(__dirname, "../../web/dist");
const PORT = parseInt(process.env.PORT ?? "8000");

if (!existsSync(STATIC_DIR)) {
  console.error(
    `Web build not found at ${STATIC_DIR}.\n` +
      "Run 'make build-web' or 'make dev-ts' from the project root."
  );
  process.exit(1);
}

const app = new Hono();
const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });

app.use("/*", cors());

const addToOrder = tool(
  async ({ item, quantity }) => {
    return `Added ${quantity} x ${item} to the order.`;
  },
  {
    name: "add_to_order",
    description: "Add an item to the customer's sandwich order.",
    schema: z.object({
      item: z.string(),
      quantity: z.number(),
    }),
  }
);

const confirmOrder = tool(
  async ({ orderSummary }) => {
    return `Order confirmed: ${orderSummary}. Sending to kitchen.`;
  },
  {
    name: "confirm_order",
    description: "Confirm the final order with the customer.",
    schema: z.object({
      orderSummary: z.string().describe("Summary of the order"),
    }),
  }
);

const systemPrompt = `
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

${DASHSCOPE_TTS_SYSTEM_PROMPT}
`;

// 使用 DashScope Qwen 模型
// DashScope 提供 OpenAI 兼容接口
// 模型选择：qwen-plus, qwen-max, qwen-turbo 等
const agent = createAgent({
  model: "openai:qwen-turbo",
  tools: [addToOrder, confirmOrder],
  checkpointer: new MemorySaver(),
  systemPrompt: systemPrompt,
});

/**
 * Transform stream: Audio (Uint8Array) → Voice Events (VoiceAgentEvent)
 *
 * 使用 DashScope SenseVoice 进行语音识别
 * 支持中文、英文、日语、粤语、韩语
 */
async function* sttStream(
  audioStream: AsyncIterable<Uint8Array>
): AsyncGenerator<VoiceAgentEvent> {
  // 使用 DashScope SenseVoice 进行语音识别
  // language="auto" 自动检测语言
  const stt = new DashScopeSTT({ sampleRate: 16000, language: "auto" });
  const passthrough = writableIterator<VoiceAgentEvent>();

  const producer = iife(async () => {
    try {
      for await (const audioChunk of audioStream) {
        await stt.sendAudio(audioChunk);
      }
    } finally {
      await stt.close();
    }
  });

  const consumer = iife(async () => {
    for await (const event of stt.receiveEvents()) {
      passthrough.push(event);
    }
  });

  try {
    yield* passthrough;
  } finally {
    await Promise.all([producer, consumer]);
  }
}

/**
 * Transform stream: Voice Events → Voice Events (with Agent Responses)
 *
 * 使用 DashScope Qwen 大模型处理用户输入并生成回复
 */
async function* agentStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  const threadId = uuidv4();

  for await (const event of eventStream) {
    yield event;
    if (event.type === "stt_output") {
      const stream = await agent.stream(
        { messages: [new HumanMessage(event.transcript)] },
        {
          configurable: { thread_id: threadId },
          streamMode: "messages",
        }
      );

      for await (const [message] of stream) {
        if (AIMessage.isInstance(message) && message.tool_calls) {
          yield { type: "agent_chunk", text: message.text, ts: Date.now() };
          for (const toolCall of message.tool_calls) {
            yield {
              type: "tool_call",
              id: toolCall.id ?? uuidv4(),
              name: toolCall.name,
              args: toolCall.args,
              ts: Date.now(),
            };
          }
        }
        if (ToolMessage.isInstance(message)) {
          yield {
            type: "tool_result",
            toolCallId: message.tool_call_id ?? "",
            name: message.name ?? "unknown",
            result:
              typeof message.content === "string"
                ? message.content
                : JSON.stringify(message.content),
            ts: Date.now(),
          };
        }
      }

      yield { type: "agent_end", ts: Date.now() };
    }
  }
}

/**
 * Transform stream: Voice Events → Voice Events (with Audio)
 *
 * 使用 DashScope CosyVoice 进行语音合成
 * 支持中文语音，音色可选：longhua（男声）、shujing（女声）等
 */
async function* ttsStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  // 使用 DashScope CosyVoice
  // voice 可选: longhua, shujing, longcheng, longhua_customer_service
  const tts = new DashScopeTTS({
    voice: "longhua",
    sampleRate: 24000,
  });
  const passthrough = writableIterator<VoiceAgentEvent>();

  const producer = iife(async () => {
    try {
      let buffer: string[] = [];
      for await (const event of eventStream) {
        passthrough.push(event);
        if (event.type === "agent_chunk") {
          buffer.push(event.text);
        }
        if (event.type === "agent_end") {
          await tts.sendText(buffer.join(""));
          buffer = [];
        }
      }
    } finally {
      await tts.close();
    }
  });

  const consumer = iife(async () => {
    for await (const event of tts.receiveEvents()) {
      passthrough.push(event);
    }
  });

  try {
    yield* passthrough;
  } finally {
    await Promise.all([producer, consumer]);
  }
}

app.get("/*", serveStatic({ root: STATIC_DIR }));

app.get(
  "/ws",
  upgradeWebSocket(async () => {
    let currentSocket: WSContext<WebSocket> | undefined;

    const inputStream = writableIterator<Uint8Array>();

    const transcriptEventStream = sttStream(inputStream);
    const agentEventStream = agentStream(transcriptEventStream);
    const outputEventStream = ttsStream(agentEventStream);

    const flushPromise = iife(async () => {
      for await (const event of outputEventStream) {
        currentSocket?.send(JSON.stringify(event));
      }
    });

    return {
      onOpen(_, ws) {
        currentSocket = ws;
      },
      onMessage(event) {
        const data = event.data;
        if (Buffer.isBuffer(data)) {
          inputStream.push(new Uint8Array(data));
        } else if (data instanceof ArrayBuffer) {
          inputStream.push(new Uint8Array(data));
        }
      },
      async onClose() {
        inputStream.cancel();
        await flushPromise;
      },
    };
  })
);

const server = serve({
  fetch: app.fetch,
  port: PORT,
});

injectWebSocket(server);

console.log(`Server is running on port ${PORT}`);
