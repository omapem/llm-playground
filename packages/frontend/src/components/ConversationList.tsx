import { useChatStore } from '@/store/chatStore';
import clsx from 'clsx';
import { Button } from './ui/button';
import { Card } from './ui/card';

export default function ConversationList() {
  const { conversations, currentConversation, setCurrentConversation, createConversation } =
    useChatStore();

  return (
    <div className="flex flex-col min-h-0 flex-1">
      <div className="p-4">
        <Button className="w-full" onClick={createConversation}>
          + New Conversation
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {conversations.length === 0 ? (
          <Card className="p-4 text-center text-sm text-gray-500">
            No conversations yet. Create one to get started!
          </Card>
        ) : (
          <div className="space-y-2">
            {conversations.map((conversation) => {
              const isSelected = currentConversation?.id === conversation.id;
              return (
                <Button
                  key={conversation.id}
                  size="sm"
                  variant={isSelected ? 'default' : 'ghost'}
                  className={clsx(
                    'w-full justify-start gap-3 text-sm rounded-md text-left conv-list-padding',
                    isSelected ? 'font-semibold' : 'text-gray-700'
                  )}
                  onClick={() => setCurrentConversation(conversation.id)}
                >
                  <div className="flex-1 text-left">
                    <div className="truncate">{conversation.title}</div>
                    <div className="mt-0.5 text-xs opacity-60">
                      {conversation.messageCount} messages ·{' '}
                      {new Date(conversation.updatedAt).toLocaleDateString()}
                    </div>
                  </div>
                </Button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
