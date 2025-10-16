import { FastifyPluginAsync } from 'fastify';
import { LLMService } from '../services/llm.js';
import { ChatRequestSchema } from '../types/index.js';
import { estimateMessageTokens, estimateTokens } from '../services/tokenizer.js';
import { estimateCost } from '../services/pricing.js';

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
        const inputTokens = estimateMessageTokens(body.messages, body.model);
        let outputContent = '';
        let streamedTokens = 0;
        for await (const chunk of llmService.streamChat(body)) {
          if (chunk.content) {
            streamedTokens += estimateTokens(chunk.content, body.model);
            outputContent += chunk.content;
          }
          reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
        }
        const cost = estimateCost(body.model, inputTokens, streamedTokens);
        const finalChunk = {
          id: 'final',
          content: '',
          role: 'assistant' as const,
          model: body.model,
          done: true,
          meta: { inputTokens, outputTokens: streamedTokens, cost },
        };
        if (process.env.DEBUG_TOKENS) {
          server.log.info({ finalMeta: finalChunk.meta }, 'Emitting final token meta');
        }
        reply.raw.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
        reply.raw.end();
      } catch (streamError) {
        // If streaming fails after headers are sent, emit an SSE error event then end
        server.log.error(streamError);
        const errMsg = streamError instanceof Error ? streamError.message : 'Streaming failed';
        const model = (body as any)?.model ?? 'unknown';
        const errorChunk = {
          id: 'error',
          content: `[Error] ${errMsg}`,
          role: 'assistant' as const,
          model,
          done: true,
        };
        reply.raw.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
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
