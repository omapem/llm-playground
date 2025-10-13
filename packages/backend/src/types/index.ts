import { z } from 'zod';

export const ChatRequestSchema = z.object({
  messages: z.array(
    z.object({
      role: z.enum(['user', 'assistant', 'system']),
      content: z.string(),
    })
  ),
  model: z.string(),
  parameters: z
    .object({
      temperature: z.number().min(0).max(2).optional(),
      maxTokens: z.number().min(1).max(4096).optional(),
      topP: z.number().min(0).max(1).optional(),
    })
    .optional(),
});

export type ChatRequest = z.infer<typeof ChatRequestSchema>;

export interface StreamChunk {
  id: string;
  content: string;
  role: 'assistant';
  model: string;
  done: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: 'openai' | 'anthropic';
  maxTokens: number;
}
