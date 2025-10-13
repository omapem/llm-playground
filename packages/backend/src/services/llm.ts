import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import type { ChatRequest, StreamChunk } from '../types/index.js';

export class LLMService {
  private openai: OpenAI | null = null;
  private anthropic: Anthropic | null = null;

  constructor() {
    if (process.env.OPENAI_API_KEY) {
      this.openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
      });
    }

    if (process.env.ANTHROPIC_API_KEY) {
      this.anthropic = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY,
      });
    }
  }

  async *streamChat(request: ChatRequest): AsyncGenerator<StreamChunk> {
    const { model, messages, parameters = {} } = request;

    if (model.startsWith('gpt-')) {
      yield* this.streamOpenAI(model, messages, parameters);
    } else if (model.startsWith('claude-')) {
      yield* this.streamAnthropic(model, messages, parameters);
    } else {
      throw new Error(`Unsupported model: ${model}`);
    }
  }

  private async *streamOpenAI(
    model: string,
    messages: ChatRequest['messages'],
    parameters: ChatRequest['parameters']
  ): AsyncGenerator<StreamChunk> {
    if (!this.openai) {
      throw new Error('OpenAI API key not configured');
    }

    const stream = await this.openai.chat.completions.create({
      model,
      messages,
      temperature: parameters?.temperature ?? 0.7,
      max_tokens: parameters?.maxTokens ?? 2000,
      top_p: parameters?.topP ?? 1,
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      const done = chunk.choices[0]?.finish_reason === 'stop';

      yield {
        id: chunk.id,
        content,
        role: 'assistant',
        model,
        done,
      };
    }
  }

  private async *streamAnthropic(
    model: string,
    messages: ChatRequest['messages'],
    parameters: ChatRequest['parameters']
  ): AsyncGenerator<StreamChunk> {
    if (!this.anthropic) {
      throw new Error('Anthropic API key not configured');
    }

    const stream = await this.anthropic.messages.create({
      model,
      messages: messages.filter(m => m.role !== 'system') as Array<{
        role: 'user' | 'assistant';
        content: string;
      }>,
      max_tokens: parameters?.maxTokens ?? 2000,
      temperature: parameters?.temperature ?? 0.7,
      top_p: parameters?.topP ?? 1,
      stream: true,
    });

    for await (const event of stream) {
      if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
        yield {
          id: crypto.randomUUID(),
          content: event.delta.text,
          role: 'assistant',
          model,
          done: false,
        };
      } else if (event.type === 'message_stop') {
        yield {
          id: crypto.randomUUID(),
          content: '',
          role: 'assistant',
          model,
          done: true,
        };
      }
    }
  }
}
