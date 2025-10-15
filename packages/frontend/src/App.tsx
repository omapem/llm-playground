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

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [parametersOpen, setParametersOpen] = useState(false);
  const { currentConversation, messages, selectedModel, setSelectedModel } = useChatStore();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

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
        <div className="w-64 border-r border-gray-200 bg-white flex flex-col min-h-0">
          <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4">
            <h1 className="text-lg font-semibold text-gray-900">LLM Playground</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-lg p-1 hover:bg-gray-100"
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
        <div className="flex h-16 items-center justify-between border-b border-gray-200 bg-white px-4">
          {sidebarOpen ? (
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700">
                {currentConversation?.title || 'New Conversation'}
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSidebarOpen(true)}
                className="rounded-lg p-1 hover:bg-gray-100"
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
              <span className="text-sm font-medium text-gray-700">
                {currentConversation?.title || 'New Conversation'}
              </span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <Select value={selectedModel} onValueChange={(v: string) => setSelectedModel(v)}>
              <SelectTrigger className="w-56">
                <SelectValue placeholder={loadingModels ? 'Loading models…' : 'Select a model'} />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button onClick={() => setParametersOpen(!parametersOpen)} variant="outline">
              ⚙️ Parameters
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex flex-1 overflow-hidden min-h-0">
          <div className="flex-1 overflow-y-auto">
            <MessageDisplay messages={messages} />
          </div>

          {/* Parameters Panel */}
          {parametersOpen && (
            <div className="w-80 overflow-y-auto border-l border-gray-200 bg-gray-50 p-4">
              <ParameterControls />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 bg-white p-4">
          <MessageInput />
        </div>
      </div>
    </div>
  );
}

export default App;
