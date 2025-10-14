import { FastifyInstance } from 'fastify';
import prisma from '../services/db';

export default async function conversationRoutes(app: FastifyInstance) {
  // Get all conversations (using stored totalCost and counts)
  app.get('/conversations', async (_request, reply) => {
    const conversations = await prisma.conversation.findMany({
      orderBy: { updatedAt: 'desc' },
      include: { _count: { select: { messages: true } } },
    });
    type Conv = (typeof conversations)[number];
    reply.send(
      conversations.map((c: Conv) => ({
        id: c.id,
        title: c.title,
        createdAt: c.createdAt,
        updatedAt: c.updatedAt,
        userId: c.userId,
        totalCost: c.totalCost ?? 0,
        messageCount: c._count.messages,
      }))
    );
  });

  // Create a conversation
  app.post('/conversations', async (request, reply) => {
    const { title } = request.body as { title?: string };
    const conversation = await prisma.conversation.create({
      data: { title, totalCost: 0 },
    });
    reply.send(conversation);
  });

  // Get messages for a conversation
  app.get('/conversations/:id/messages', async (request, reply) => {
    const { id } = request.params as { id: string };
    const messages = await prisma.message.findMany({
      where: { conversationId: id },
      orderBy: { createdAt: 'asc' },
    });
    reply.send(messages);
  });

  // Add a message to a conversation
  app.post('/conversations/:id/messages', async (request, reply) => {
    const { id } = request.params as { id: string };
    const { role, content } = request.body as { role: string; content: string };
    const message = await prisma.message.create({
      data: { conversationId: id, role, content },
    });
    await prisma.conversation.update({
      where: { id },
      data: { updatedAt: new Date() },
    });
    reply.send(message);
  });

  // Update (patch) a message (e.g., after streaming completes or progressive persist)
  app.patch('/conversations/:conversationId/messages/:messageId', async (request, reply) => {
    const { conversationId, messageId } = request.params as {
      conversationId: string;
      messageId: string;
    };
    const { content, model, tokens, cost, parameters } = request.body as {
      content?: string;
      model?: string;
      tokens?: number;
      cost?: number;
      parameters?: unknown;
    };

    // Fetch previous cost if we might update cost to compute delta
    let previousCost: number | null = null;
    if (cost !== undefined) {
      const prior = await prisma.message.findUnique({
        where: { id: messageId },
        select: { cost: true },
      });
      previousCost = prior?.cost ?? null;
    }

    const updated = await prisma.message.update({
      where: { id: messageId },
      data: {
        ...(content !== undefined ? { content } : {}),
        ...(model !== undefined ? { model } : {}),
        ...(tokens !== undefined ? { tokens } : {}),
        ...(cost !== undefined ? { cost } : {}),
        ...(parameters !== undefined ? { parameters } : {}),
      },
    });

    if (cost !== undefined) {
      const delta = (cost ?? 0) - (previousCost ?? 0);
      if (delta !== 0) {
        await prisma.conversation.update({
          where: { id: conversationId },
          data: {
            updatedAt: new Date(),
            totalCost: { increment: delta },
          },
        });
      } else {
        await prisma.conversation.update({
          where: { id: conversationId },
          data: { updatedAt: new Date() },
        });
      }
    } else {
      await prisma.conversation.update({
        where: { id: conversationId },
        data: { updatedAt: new Date() },
      });
    }
    reply.send(updated);
  });
}
