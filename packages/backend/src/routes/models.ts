import { FastifyPluginAsync } from 'fastify';
import type { ModelInfo } from '../types/index.js';

const modelsRoutes: FastifyPluginAsync = async (server) => {
  server.get('/models', async () => {
    const models: ModelInfo[] = [
      {
        id: 'gpt-4',
        name: 'GPT-4',
        provider: 'openai',
        maxTokens: 8192,
      },
      {
        id: 'gpt-3.5-turbo',
        name: 'GPT-3.5 Turbo',
        provider: 'openai',
        maxTokens: 4096,
      },
      {
        id: 'claude-3-opus-20240229',
        name: 'Claude 3 Opus',
        provider: 'anthropic',
        maxTokens: 4096,
      },
      {
        id: 'claude-3-sonnet-20240229',
        name: 'Claude 3 Sonnet',
        provider: 'anthropic',
        maxTokens: 4096,
      },
    ];

    // Filter based on available API keys
    const availableModels = models.filter((model) => {
      if (model.provider === 'openai') return !!process.env.OPENAI_API_KEY;
      if (model.provider === 'anthropic') return !!process.env.ANTHROPIC_API_KEY;
      return false;
    });

    return { models: availableModels };
  });
};

export default modelsRoutes;
