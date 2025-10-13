import { useChatStore } from '@/store/chatStore';
import clsx from 'clsx';

export default function ConversationList() {
  const { conversations, currentConversation, setCurrentConversation, createConversation } =
    useChatStore();

  return (
    <div className="flex h-full flex-col">
      <div className="p-4">
        <button
          onClick={createConversation}
          className="w-full rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-700"
        >
          + New Conversation
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 ? (
          <div className="p-4 text-center text-sm text-gray-500">
            No conversations yet. Create one to get started!
          </div>
        ) : (
          <div className="space-y-1 px-2">
            {conversations.map((conversation) => (
              <button
                key={conversation.id}
                onClick={() => setCurrentConversation(conversation)}
                className={clsx(
                  'w-full rounded-lg px-3 py-2 text-left text-sm transition-colors',
                  currentConversation?.id === conversation.id
                    ? 'bg-primary-100 text-primary-900'
                    : 'text-gray-700 hover:bg-gray-100'
                )}
              >
                <div className="truncate font-medium">{conversation.title}</div>
                <div className="mt-1 text-xs opacity-60">
                  {conversation.messageCount} messages •{' '}
                  {new Date(conversation.updatedAt).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
