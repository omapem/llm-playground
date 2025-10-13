import { useState } from 'react';
import { useChatStore } from '@/store/chatStore';

export default function MessageInput() {
  const [input, setInput] = useState('');
  const { addMessage, isStreaming, currentConversation } = useChatStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming || !currentConversation) return;

    const userMessage = {
      id: crypto.randomUUID(),
      conversationId: currentConversation.id,
      role: 'user' as const,
      content: input.trim(),
      timestamp: Date.now(),
    };

    addMessage(userMessage);
    setInput('');

    // API call will be implemented in Phase 2
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
          }
        }}
        placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
        className="flex-1 resize-none rounded-lg border border-gray-300 px-4 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
        rows={3}
        disabled={isStreaming}
      />
      <button
        type="submit"
        disabled={!input.trim() || isStreaming}
        className="rounded-lg bg-primary-600 px-6 py-2 text-white transition-colors hover:bg-primary-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isStreaming ? 'Sending...' : 'Send'}
      </button>
    </form>
  );
}
