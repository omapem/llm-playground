import { Message } from '@/types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import clsx from 'clsx';
import { WavyBackground } from './ui/shadcn-io/wavy-background';
import { AuroraBackground } from './ui/shadcn-io/aurora-background';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { useChatStore } from '@/store/chatStore';
import { useEffect, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { Button } from './ui/button';

interface MessageDisplayProps {
  messages: Message[];
}

function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group">
      <Button
        variant="ghost"
        size="icon"
        className="absolute right-2 top-2 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-700/80 hover:bg-slate-600"
        onClick={handleCopy}
        title="Copy code"
      >
        {copied ? <Check className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3 text-slate-200" />}
      </Button>
      <pre className="rounded-lg bg-slate-900 text-slate-100 p-4 overflow-x-auto text-xs leading-relaxed">
        <code className={language ? `language-${language}` : ''}>{code}</code>
      </pre>
    </div>
  );
}

export default function MessageDisplay({ messages }: MessageDisplayProps) {
  const { isStreaming, inProgressAssistantId } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive or streaming updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

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
          <Card className="max-w-2xl text-center rounded-xl shadow-lg">
            <CardHeader className="space-y-3 pb-6">
              <CardTitle className="text-2xl">Welcome to LLM Playground</CardTitle>
              <CardDescription className="text-base">
                Select a model and start exploring AI capabilities
              </CardDescription>
            </CardHeader>
            <CardContent className="pb-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-left">
                <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100/50 rounded-lg border border-blue-200/50">
                  <h3 className="font-semibold text-sm text-blue-900 mb-1">💡 Get Help</h3>
                  <p className="text-xs text-blue-700">Ask me to explain code, debug issues, or provide guidance</p>
                </div>
                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100/50 rounded-lg border border-purple-200/50">
                  <h3 className="font-semibold text-sm text-purple-900 mb-1">✨ Generate Code</h3>
                  <p className="text-xs text-purple-700">Request code snippets, functions, or full implementations</p>
                </div>
                <div className="p-4 bg-gradient-to-br from-green-50 to-green-100/50 rounded-lg border border-green-200/50">
                  <h3 className="font-semibold text-sm text-green-900 mb-1">🔍 Analyze</h3>
                  <p className="text-xs text-green-700">Review code quality, security, or performance optimization</p>
                </div>
                <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100/50 rounded-lg border border-orange-200/50">
                  <h3 className="font-semibold text-sm text-orange-900 mb-1">🎓 Learn</h3>
                  <p className="text-xs text-orange-700">Explore concepts, patterns, and best practices</p>
                </div>
              </div>
              <p className="mt-6 text-xs text-slate-500">
                💡 Tip: Press <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-300 rounded text-slate-700 font-mono">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-300 rounded text-slate-700 font-mono">Shift+Enter</kbd> for new line
              </p>
            </CardContent>
          </Card>
        </WavyBackground>
      </div>
    );
  }

  return (
    <AuroraBackground>
      <div ref={containerRef} className="w-full h-full flex flex-col space-y-3 p-4 md:p-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx('flex', message.role === 'user' ? 'justify-end' : 'justify-start')}
          >
            <Card
              className={clsx(
                'max-w-[85%] md:max-w-[75%] shadow-sm transition-all hover:shadow-md rounded-lg',
                message.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-white/95 backdrop-blur-sm'
              )}
            >
              <CardContent className="pt-3 pb-3">
                <div className="mb-2 flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-semibold">
                    {message.role === 'user' ? 'You' : message.model || 'Assistant'}
                  </span>
                  <span className="text-xs opacity-60">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                  {message.role === 'assistant' && (message.outputTokens || message.tokens) && (
                    <span
                      className="text-[10px] px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 font-medium cursor-help transition-colors hover:bg-slate-200"
                      title={`Input: ${message.inputTokens ?? '–'} tokens | Output: ${message.outputTokens ?? message.tokens ?? '–'} tokens | Cost: ${message.cost !== undefined ? '$' + message.cost.toFixed(6) : '–'}`}
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
                      <span className="text-[10px] px-2 py-0.5 rounded-md bg-amber-50 text-amber-700 font-medium animate-pulse">
                        Streaming…
                      </span>
                    )}
                </div>
                <div className={clsx('prose max-w-none prose-sm', message.role === 'user' ? 'prose-invert' : '')}>
                  {message.role === 'assistant' ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        code({ inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          const codeString = String(children).replace(/\n$/, '');

                          if (!inline && match) {
                            return <CodeBlock code={codeString} language={match[1]} />;
                          }
                          if (!inline) {
                            return <CodeBlock code={codeString} />;
                          }

                          return (
                            <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800 text-xs font-mono" {...props}>
                              {children}
                            </code>
                          );
                        },
                        table({ children, ...props }) {
                          return (
                            <div className="overflow-x-auto my-4 rounded-lg border border-slate-200">
                              <table
                                className="min-w-full text-xs"
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
                              className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-left font-semibold text-slate-700"
                              {...props}
                            >
                              {children}
                            </th>
                          );
                        },
                        td({ children, ...props }) {
                          return (
                            <td className="border-b border-slate-100 px-3 py-2 align-top" {...props}>
                              {children}
                            </td>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    <p className="whitespace-pre-wrap text-sm m-0 leading-relaxed">
                      {message.content}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        ))}
        <div ref={messagesEndRef} />
        {/* Footer summary */}
        <div className="sticky bottom-0 pt-3 pb-2 text-xs text-slate-600 flex gap-6 border-t border-slate-200/80 bg-white/90 backdrop-blur-sm rounded-lg px-4 py-2.5 shadow-sm">
          <span className="font-semibold">
            Total Tokens:{' '}
            <span className="text-slate-900 font-bold">
              {messages
                .filter((m) => m.role === 'assistant')
                .reduce((s, m) => s + (m.outputTokens || m.tokens || 0), 0).toLocaleString()}
            </span>
          </span>
          <span className="font-semibold">
            Total Cost:{' '}
            <span className="text-slate-900 font-bold">
              ${messages
                .filter((m) => m.role === 'assistant')
                .reduce((s, m) => s + (m.cost || 0), 0)
                .toFixed(4)}
            </span>
          </span>
        </div>
      </div>
    </AuroraBackground>
  );
}
