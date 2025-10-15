import { useState } from 'react';
import { useChatStore } from '@/store/chatStore';
import { apiClient } from '@/services/api';
import type { Message } from '@/types';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { Send } from 'lucide-react';

export default function MessageInput() {
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const {
    addMessage,
    appendToMessage,
    updateMessageMeta,
    isStreaming,
    setIsStreaming,
    currentConversation,
    createConversation,
    parameters,
    selectedModel,
    setInProgressAssistantId,
  } = useChatStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    // Auto-create conversation if none exists
    let conversation = currentConversation;
    if (!conversation) {
      try {
        conversation = await createConversation('New Conversation');
      } catch {
        setError('Failed to create conversation');
        return;
      }
    }

    setError(null);
    const userMessage: Message = {
      id: crypto.randomUUID(),
      conversationId: conversation.id,
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    };

    await addMessage(userMessage);
    setInput('');
    setIsStreaming(true);

    // Create assistant message placeholder
    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      conversationId: conversation.id,
      role: 'assistant',
      content: '',
      model: selectedModel,
      parameters: parameters as unknown as Record<string, unknown>,
      timestamp: Date.now(),
    };
    const persistedAssistant = await addMessage(assistantMessage);
    const assistantMessageId = persistedAssistant.id;
    // Track in-progress assistant id in store for robust targeting
    setInProgressAssistantId(assistantMessageId);

    try {
      // Stream the response. Use latest messages from store, excluding the empty assistant placeholder
      const latestMessages = useChatStore
        .getState()
        .messages.filter(
          (m) =>
            m.conversationId === conversation.id && (m.role !== 'assistant' || m.content.length > 0)
        );
      const conversationMessages = latestMessages;

      let lastPersistTime = Date.now();
      let lastPersistLength = 0;
      const PERSIST_INTERVAL_MS = 1200; // time-based cap
      const PERSIST_MIN_DELTA = 120; // only persist if at least this many new chars since last persist
      for await (const chunk of apiClient.streamChat(
        conversationMessages,
        selectedModel,
        parameters
      )) {
        if (chunk.content) {
          const targetId = useChatStore.getState().inProgressAssistantId || assistantMessageId;
          appendToMessage(targetId, chunk.content);
        }

        // Progressive persistence (throttled)
        const now = Date.now();
        const latestAssistant = useChatStore
          .getState()
          .messages.find((m) => m.id === assistantMessageId);
        if (latestAssistant) {
          const len = latestAssistant.content.length;
          if (
            len - lastPersistLength >= PERSIST_MIN_DELTA ||
            now - lastPersistTime > PERSIST_INTERVAL_MS
          ) {
            lastPersistTime = now;
            lastPersistLength = len;
            apiClient
              .updateMessage(conversation.id, assistantMessageId, {
                content: latestAssistant.content,
              })
              .catch(() => {
                /* ignore transient errors */
              });
          }
        }

        if (chunk.meta) {
          console.log('[stream] final meta received', chunk.meta);
          // Update UI meta immediately so badge reflects accurate values
          const targetId = useChatStore.getState().inProgressAssistantId || assistantMessageId;
          updateMessageMeta(targetId, {
            inputTokens: chunk.meta.inputTokens,
            outputTokens: chunk.meta.outputTokens,
            tokens: chunk.meta.outputTokens, // alias
            cost: chunk.meta.cost,
          });

          // Persist final token/cost metadata to backend right away to update conversation totalCost
          try {
            const targetId2 = useChatStore.getState().inProgressAssistantId || assistantMessageId;
            await apiClient.updateMessage(conversation.id, targetId2, {
              model: selectedModel,
              tokens: chunk.meta.outputTokens,
              cost: chunk.meta.cost,
              parameters: parameters as unknown as Record<string, unknown>,
            });
            // Optionally refresh conversations to reflect updated totalCost quickly
            apiClient
              .getConversations()
              .then((raw) => {
                const mapped = raw.map(
                  (c: {
                    id: string;
                    title?: string;
                    createdAt: string;
                    updatedAt: string;
                    messageCount?: number;
                    totalCost?: number;
                  }) => ({
                    id: c.id,
                    title: c.title || 'Untitled',
                    createdAt: new Date(c.createdAt).getTime(),
                    updatedAt: new Date(c.updatedAt).getTime(),
                    messageCount: c.messageCount ?? 0,
                    totalCost: c.totalCost ?? 0,
                  })
                );
                useChatStore.setState({ conversations: mapped });
              })
              .catch(() => {});
          } catch (e) {
            // Non-fatal; we'll attempt another persist in finally
            console.warn('Failed to persist final meta immediately; will retry in finally.', e);
          }

          // Break after processing meta final chunk
          break;
        }

        // Ignore early done flag (e.g., provider finish) until meta arrives
        if (chunk.done && !chunk.meta) {
          // continue waiting for meta
          continue;
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get response';
      setError(errorMessage);
      console.error('Stream chat error:', err);
    } finally {
      setIsStreaming(false);
      setInProgressAssistantId(null);
      // Persist final assistant message content to backend
      try {
        const finalId = useChatStore.getState().inProgressAssistantId || assistantMessageId;
        const latestAssistant = useChatStore.getState().messages.find((m) => m.id === finalId);
        if (latestAssistant) {
          await apiClient.updateMessage(conversation.id, finalId, {
            content: latestAssistant.content,
            model: latestAssistant.model || selectedModel,
            tokens: latestAssistant.outputTokens || latestAssistant.tokens,
            cost: latestAssistant.cost,
            parameters: parameters as unknown as Record<string, unknown>,
          });
          // Refresh conversations to show updated totalCost if cost applied
          apiClient
            .getConversations()
            .then((raw) => {
              const mapped = raw.map(
                (c: {
                  id: string;
                  title?: string;
                  createdAt: string;
                  updatedAt: string;
                  messageCount?: number;
                  totalCost?: number;
                }) => ({
                  id: c.id,
                  title: c.title || 'Untitled',
                  createdAt: new Date(c.createdAt).getTime(),
                  updatedAt: new Date(c.updatedAt).getTime(),
                  messageCount: c.messageCount ?? 0,
                  totalCost: c.totalCost ?? 0,
                })
              );
              // Update store manually
              useChatStore.setState({ conversations: mapped });
            })
            .catch(() => {});
        }
      } catch (persistErr) {
        console.error('Failed to persist assistant message final content:', persistErr);
      }
    }
  };

  return (
    <div className="space-y-2">
      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-600">{error}</div>}
      <form onSubmit={handleSubmit} className="flex items-end gap-3">
        <div className="flex-1 min-w-0">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
            className="w-full resize-none min-h-[56px] max-h-40"
            rows={2}
            disabled={isStreaming}
          />
        </div>
        <div className="flex-shrink-0 self-center -mb-1">
          <Button
            type="submit"
            disabled={!input.trim() || isStreaming}
            variant="default"
            size="icon"
            title={
              !input.trim() ? 'Type a message to send' : isStreaming ? 'Sending...' : 'Send message'
            }
            aria-label={
              !input.trim() ? 'Type a message to send' : isStreaming ? 'Sending' : 'Send message'
            }
            className="h-10 w-10"
          >
            {isStreaming ? <span aria-hidden>⏳</span> : <Send className="h-4 w-4" aria-hidden />}
          </Button>
        </div>
      </form>
    </div>
  );
}
