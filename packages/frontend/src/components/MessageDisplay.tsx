import { Message } from '@/types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import clsx from 'clsx';
import { WavyBackground } from './ui/shadcn-io/wavy-background';
import { AuroraBackground } from './ui/shadcn-io/aurora-background';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { useChatStore } from '@/store/chatStore';

interface MessageDisplayProps {
  messages: Message[];
}

export default function MessageDisplay({ messages }: MessageDisplayProps) {
  const { isStreaming, inProgressAssistantId } = useChatStore();
  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <WavyBackground
          backgroundFill="white"
          colors={['#38bdf8', '#818cf8', '#c084fc', '#e879f9']}
          waveWidth={50}
          blur={10}
          speed="fast"
          waveOpacity={0.5}
          containerClassName="h-full w-full"
          className="flex items-center justify-center"
        >
          <Card className="max-w-xl text-center">
            <CardHeader>
              <CardTitle>Welcome to LLM Playground</CardTitle>
              <CardDescription>Start a conversation by typing a message below</CardDescription>
            </CardHeader>
          </Card>
        </WavyBackground>
      </div>
    );
  }

  return (
    <AuroraBackground>
      <div className="w-full max-w-3xl h-full flex flex-col space-y-4 p-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx('flex', message.role === 'user' ? 'justify-end' : 'justify-start')}
          >
            <Card
              className={clsx(
                'max-w-3xl',
                message.role === 'user' ? 'bg-primary text-white' : 'bg-white'
              )}
            >
              <CardContent>
                <div className="mb-1 flex items-center gap-2">
                  <span className="text-xs font-medium opacity-75">
                    {message.role === 'user' ? 'You' : message.model || 'Assistant'}
                  </span>
                  <span className="text-xs opacity-50">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                  {message.role === 'assistant' && (message.outputTokens || message.tokens) && (
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 cursor-help"
                      title={`Input: ${message.inputTokens ?? '–'} | Output: ${message.outputTokens ?? message.tokens ?? '–'} | Cost: ${message.cost !== undefined ? '$' + message.cost.toFixed(6) : '–'}`}
                    >
                      {message.outputTokens || message.tokens} tok
                      {message.cost !== undefined && (
                        <>
                          {' · $'}
                          {message.cost.toFixed(4)}
                        </>
                      )}
                    </span>
                  )}
                  {message.role === 'assistant' &&
                    isStreaming &&
                    inProgressAssistantId === message.id && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 animate-pulse">
                        Streaming…
                      </span>
                    )}
                </div>
                <div className="prose max-w-none prose-sm dark:prose-invert">
                  {message.role === 'assistant' ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        code({ className, children, ...props }) {
                          return (
                            <pre className="rounded bg-slate-900 text-slate-100 p-3 overflow-x-auto text-[11px] leading-relaxed">
                              <code className={className} {...props}>
                                {children}
                              </code>
                            </pre>
                          );
                        },
                        table({ children, ...props }) {
                          return (
                            <div className="overflow-x-auto mb-2">
                              <table
                                className="min-w-full text-xs border border-slate-300"
                                {...props}
                              >
                                {children}
                              </table>
                            </div>
                          );
                        },
                        th({ children, ...props }) {
                          return (
                            <th
                              className="border border-slate-300 bg-slate-100 px-2 py-1 text-left"
                              {...props}
                            >
                              {children}
                            </th>
                          );
                        },
                        td({ children, ...props }) {
                          return (
                            <td className="border border-slate-300 px-2 py-1 align-top" {...props}>
                              {children}
                            </td>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    <pre className="whitespace-pre-wrap font-sans text-sm m-0">
                      {message.content}
                    </pre>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        ))}
        {/* Footer summary */}
        <div className="mt-auto pt-2 text-xs text-slate-500 flex gap-4 border-t border-slate-200">
          <span>
            Tokens:{' '}
            {messages
              .filter((m) => m.role === 'assistant')
              .reduce((s, m) => s + (m.outputTokens || m.tokens || 0), 0)}
          </span>
          <span>
            Cost: $
            {messages
              .filter((m) => m.role === 'assistant')
              .reduce((s, m) => s + (m.cost || 0), 0)
              .toFixed(4)}
          </span>
        </div>
      </div>
    </AuroraBackground>
  );
}
