import type { Message, LLMParameters } from '@/types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

export interface ChatRequestPayload {
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
  }>;
  model: string;
  parameters?: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
  };
}

export interface StreamChunk {
  id: string;
  content: string;
  role: 'assistant';
  model: string;
  done: boolean;
  meta?: {
    inputTokens: number;
    outputTokens: number;
    cost: number;
  };
}

export class APIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: unknown
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Stream chat completion from the API using Server-Sent Events
   */
  async *streamChat(
    messages: Message[],
    model: string,
    parameters: LLMParameters
  ): AsyncGenerator<StreamChunk, void, unknown> {
    const payload: ChatRequestPayload = {
      messages: messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      })),
      model,
      parameters: {
        temperature: parameters.temperature,
        maxTokens: parameters.maxTokens,
        topP: parameters.topP,
      },
    };

    const response = await fetch(`${this.baseURL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(errorData.error || 'Failed to stream chat', response.status, errorData);
    }

    if (!response.body) {
      throw new APIError('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6); // Remove 'data: ' prefix

            if (data.trim()) {
              try {
                const chunk: StreamChunk = JSON.parse(data);
                yield chunk;
              } catch (error) {
                console.error('Failed to parse SSE data:', data, error);
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Get available models from the API
   */
  async getModels(): Promise<Array<{ id: string; name: string; provider: string }>> {
    const response = await fetch(`${this.baseURL}/api/models`);
    if (!response.ok) {
      throw new APIError('Failed to fetch models', response.status);
    }
    return response.json();
  }

  /**
   * Get all conversations
   */
  async getConversations(): Promise<
    Array<{
      id: string;
      title: string;
      createdAt: string;
      updatedAt: string;
      messageCount?: number;
    }>
  > {
    const response = await fetch(`${this.baseURL}/api/conversations`);
    if (!response.ok) {
      throw new APIError('Failed to fetch conversations', response.status);
    }
    return response.json();
  }

  /**
   * Create a new conversation
   */
  async createConversation(
    title?: string
  ): Promise<{ id: string; title: string; createdAt: string; updatedAt: string }> {
    const response = await fetch(`${this.baseURL}/api/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (!response.ok) {
      throw new APIError('Failed to create conversation', response.status);
    }
    return response.json();
  }

  /**
   * Get messages for a conversation
   */
  async getMessages(conversationId: string): Promise<Array<Message>> {
    const response = await fetch(`${this.baseURL}/api/conversations/${conversationId}/messages`);
    if (!response.ok) {
      throw new APIError('Failed to fetch messages', response.status);
    }
    return response.json();
  }

  /**
   * Add a message to a conversation
   */
  async addMessage(
    conversationId: string,
    message: { role: string; content: string; model?: string; parameters?: any; tokens?: number }
  ): Promise<Message> {
    const response = await fetch(`${this.baseURL}/api/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(message),
    });
    if (!response.ok) {
      throw new APIError('Failed to add message', response.status);
    }
    return response.json();
  }

  /**
   * Patch/update a message (e.g., final streamed content, tokens, cost)
   */
  async updateMessage(
    conversationId: string,
    messageId: string,
    data: Partial<{ content: string; model: string; tokens: number; cost: number; parameters: any }>
  ): Promise<Message> {
    const response = await fetch(
      `${this.baseURL}/api/conversations/${conversationId}/messages/${messageId}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }
    );
    if (!response.ok) {
      throw new APIError('Failed to update message', response.status);
    }
    return response.json();
  }
}

// Export a singleton instance
export const apiClient = new APIClient();
