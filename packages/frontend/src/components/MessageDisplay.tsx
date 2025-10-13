import { Message } from '@/types';
import clsx from 'clsx';
import { WavyBackground } from './ui/shadcn-io/wavy-background';
import { AuroraBackground } from './ui/shadcn-io/aurora-background';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';

interface MessageDisplayProps {
  messages: Message[];
}

export default function MessageDisplay({ messages }: MessageDisplayProps) {
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
                </div>
                <div className="whitespace-pre-wrap">{message.content}</div>
              </CardContent>
            </Card>
          </div>
        ))}
      </div>
    </AuroraBackground>
  );
}
