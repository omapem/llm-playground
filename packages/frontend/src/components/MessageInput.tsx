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
    isStreaming,
    setIsStreaming,
    currentConversation,
    createConversation,
    messages,
    parameters,
    selectedModel,
  } = useChatStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    // Auto-create conversation if none exists
    let conversation = currentConversation;
    if (!conversation) {
      createConversation();
      // Get the newly created conversation from store
      conversation = useChatStore.getState().currentConversation;
      if (!conversation) {
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

    addMessage(userMessage);
    setInput('');
    setIsStreaming(true);

    // Create assistant message placeholder
    const assistantMessageId = crypto.randomUUID();
    const assistantMessage: Message = {
      id: assistantMessageId,
      conversationId: conversation.id,
      role: 'assistant',
      content: '',
      model: selectedModel,
      timestamp: Date.now(),
    };

    addMessage(assistantMessage);

    try {
      // Stream the response
      const conversationMessages = [...messages, userMessage];

      for await (const chunk of apiClient.streamChat(
        conversationMessages,
        selectedModel,
        parameters
      )) {
        if (chunk.content) {
          appendToMessage(assistantMessageId, chunk.content);
        }

        if (chunk.done) {
          break;
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get response';
      setError(errorMessage);
      console.error('Stream chat error:', err);
    } finally {
      setIsStreaming(false);
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
