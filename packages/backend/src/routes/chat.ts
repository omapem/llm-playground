import { FastifyPluginAsync } from 'fastify';
import { LLMService } from '../services/llm.js';
import { ChatRequestSchema } from '../types/index.js';

const chatRoutes: FastifyPluginAsync = async (server) => {
  const llmService = new LLMService();

  server.post('/chat', async (request, reply) => {
    try {
      const body = ChatRequestSchema.parse(request.body);

      // Set up SSE headers with CORS
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Access-Control-Allow-Origin': process.env.NODE_ENV === 'production' ? 'false' : '*',
        'Access-Control-Allow-Credentials': 'true',
      });

      try {
        // Stream the response
        for await (const chunk of llmService.streamChat(body)) {
          reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
        }
        reply.raw.end();
      } catch (streamError) {
        // If streaming fails after headers are sent, just end the stream
        server.log.error(streamError);
        reply.raw.write(`data: ${JSON.stringify({
          error: streamError instanceof Error ? streamError.message : 'Streaming failed',
          done: true
        })}\n\n`);
        reply.raw.end();
      }
    } catch (error) {
      server.log.error(error);
      // Only send HTTP error if headers haven't been sent yet
      if (!reply.sent) {
        return reply.status(400).send({
          error: error instanceof Error ? error.message : 'Invalid request',
        });
      }
    }
  });
};

export default chatRoutes;
