import { FastifyPluginAsync } from 'fastify';
import { LLMService } from '../services/llm.js';
import { ChatRequestSchema } from '../types/index.js';

const chatRoutes: FastifyPluginAsync = async (server) => {
  const llmService = new LLMService();

  server.post('/chat', async (request, reply) => {
    try {
      const body = ChatRequestSchema.parse(request.body);

      // Set up SSE headers
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });

      // Stream the response
      for await (const chunk of llmService.streamChat(body)) {
        reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }

      reply.raw.end();
    } catch (error) {
      server.log.error(error);
      return reply.status(400).send({
        error: error instanceof Error ? error.message : 'Invalid request',
      });
    }
  });
};

export default chatRoutes;
