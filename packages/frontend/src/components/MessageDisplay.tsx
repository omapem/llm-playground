import { Message } from '@/types';
import clsx from 'clsx';

interface MessageDisplayProps {
  messages: Message[];
}

export default function MessageDisplay({ messages }: MessageDisplayProps) {
  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-gray-700">Welcome to LLM Playground</h2>
          <p className="mt-2 text-gray-500">Start a conversation by typing a message below</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-4">
      {messages.map((message) => (
        <div
          key={message.id}
          className={clsx(
            'flex',
            message.role === 'user' ? 'justify-end' : 'justify-start'
          )}
        >
          <div
            className={clsx(
              'max-w-3xl rounded-lg px-4 py-3',
              message.role === 'user'
                ? 'bg-primary-600 text-white'
                : 'bg-white border border-gray-200 text-gray-900'
            )}
          >
            <div className="mb-1 flex items-center gap-2">
              <span className="text-xs font-medium opacity-75">
                {message.role === 'user' ? 'You' : message.model || 'Assistant'}
              </span>
              <span className="text-xs opacity-50">
                {new Date(message.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="whitespace-pre-wrap">{message.content}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
