import Fastify from 'fastify';
import cors from '@fastify/cors';
import env from '@fastify/env';
import chatRoutes from './routes/chat.js';
import modelsRoutes from './routes/models.js';

const envSchema = {
  type: 'object',
  required: ['PORT'],
  properties: {
    PORT: {
      type: 'number',
      default: 3000,
    },
    OPENAI_API_KEY: {
      type: 'string',
    },
    ANTHROPIC_API_KEY: {
      type: 'string',
    },
    NODE_ENV: {
      type: 'string',
      default: 'development',
    },
  },
};

async function buildServer() {
  const server = Fastify({
    logger: {
      level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
    },
  });

  // Register plugins
  await server.register(env, {
    schema: envSchema,
    dotenv: true,
  });

  await server.register(cors, {
    origin: process.env.NODE_ENV === 'production' ? false : true,
    credentials: true,
  });

  // Register routes
  await server.register(chatRoutes, { prefix: '/api' });
  await server.register(modelsRoutes, { prefix: '/api' });

  // Health check
  server.get('/health', async () => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  return server;
}

async function start() {
  try {
    const server = await buildServer();
    const port = server.config.PORT as number;

    await server.listen({ port, host: '0.0.0.0' });
    console.log(`🚀 Server listening on http://localhost:${port}`);
  } catch (err) {
    console.error('Error starting server:', err);
    process.exit(1);
  }
}

start();
