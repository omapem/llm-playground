import { useEffect, useState } from 'react';
import ConversationList from '@/components/ConversationList';
import MessageDisplay from '@/components/MessageDisplay';
import MessageInput from '@/components/MessageInput';
import ParameterControls from '@/components/ParameterControls';
import { useChatStore } from '@/store/chatStore';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from '@/components/ui/select';
import { apiClient, type ModelInfo } from '@/services/api';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [parametersOpen, setParametersOpen] = useState(false);
  const { currentConversation, messages, selectedModel, setSelectedModel, createConversation } =
    useChatStore();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 'b',
      ctrl: true,
      action: () => setSidebarOpen((prev) => !prev),
      description: 'Toggle sidebar',
    },
    {
      key: 'k',
      ctrl: true,
      action: () => setParametersOpen((prev) => !prev),
      description: 'Toggle parameters',
    },
    {
      key: 'n',
      ctrl: true,
      action: () => createConversation(),
      description: 'New conversation',
    },
    {
      key: 'Escape',
      action: () => {
        if (parametersOpen) setParametersOpen(false);
      },
      description: 'Close panels',
    },
  ]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setLoadingModels(true);
        const list = await apiClient.getModels();
        if (!mounted) return;
        setModels(list);
        // If the selected model isn't available, pick the first available
        if (list.length > 0) {
          const found = list.find((m) => m.id === selectedModel);
          if (!found) setSelectedModel(list[0].id);
        }
      } catch (e) {
        console.warn('Failed to load models:', e);
        setModels([]);
      } finally {
        setLoadingModels(false);
      }
    })();
    return () => {
      mounted = false;
    };
    // It's safe to include selectedModel/setSelectedModel; guard above prevents loops
  }, [selectedModel, setSelectedModel]);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* Sidebar */}
      {sidebarOpen && (
        <div className="w-64 border-r border-gray-200 bg-white flex flex-col min-h-0 shadow-sm">
          <div className="flex h-14 items-center justify-between border-b border-gray-200 px-4">
            <h1 className="text-base font-semibold text-gray-900">LLM Playground</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-lg p-1.5 hover:bg-gray-100 transition-colors"
              aria-label="Close sidebar"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
          <ConversationList />
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex flex-1 flex-col min-h-0">
        {/* Header */}
        <div className="flex h-14 items-center justify-between border-b border-gray-200 bg-white px-4 shadow-sm">
          {sidebarOpen ? (
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm font-medium text-gray-700 truncate">
                {currentConversation?.title || 'New Conversation'}
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 min-w-0">
              <button
                onClick={() => setSidebarOpen(true)}
                className="rounded-lg p-1.5 hover:bg-gray-100 transition-colors"
                aria-label="Open sidebar"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
              <span className="text-sm font-medium text-gray-700 truncate">
                {currentConversation?.title || 'New Conversation'}
              </span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <Select value={selectedModel} onValueChange={(v: string) => setSelectedModel(v)}>
              <SelectTrigger className="w-56 h-9 text-sm">
                <SelectValue placeholder={loadingModels ? 'Loading models…' : 'Select a model'} />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id} className="py-2.5">
                    <div className="flex flex-col gap-0.5">
                      <div className="font-medium text-sm">{m.name}</div>
                      <div className="text-xs text-slate-500">
                        {m.provider === 'anthropic' ? 'Anthropic' : 'OpenAI'}
                        {m.maxTokens && ` · ${(m.maxTokens / 1000).toFixed(0)}K context`}
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={() => setParametersOpen(!parametersOpen)}
              variant={parametersOpen ? 'default' : 'outline'}
              size="sm"
              className="h-9 transition-all"
            >
              <span className="mr-1">⚙️</span>
              Parameters
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex flex-1 overflow-hidden min-h-0">
          <div className="flex-1 overflow-y-auto">
            <MessageDisplay messages={messages} />
          </div>

          {/* Parameters Panel */}
          <div
            className={`w-80 overflow-y-auto border-l border-gray-200 bg-gray-50/95 backdrop-blur-sm p-4 transition-all duration-200 ease-out ${
              parametersOpen ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0 absolute right-0 pointer-events-none'
            }`}
          >
            <div className="space-y-1 mb-4">
              <h3 className="text-sm font-semibold text-gray-900">Model Parameters</h3>
              <p className="text-xs text-gray-500">Fine-tune the model behavior</p>
            </div>
            <ParameterControls />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 bg-white p-4">
          <MessageInput modelsAvailable={models.length > 0} />
        </div>
      </div>
    </div>
  );
}

export default App;
