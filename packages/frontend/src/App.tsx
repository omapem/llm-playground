import { useState } from 'react';
import ConversationList from '@/components/ConversationList';
import MessageDisplay from '@/components/MessageDisplay';
import MessageInput from '@/components/MessageInput';
import ParameterControls from '@/components/ParameterControls';
import { useChatStore } from '@/store/chatStore';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [parametersOpen, setParametersOpen] = useState(false);
  const { currentConversation, messages, selectedModel, setSelectedModel } = useChatStore();

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      {sidebarOpen && (
        <div className="w-64 border-r border-gray-200 bg-white">
          <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4">
            <h1 className="text-lg font-semibold text-gray-900">LLM Playground</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-lg p-1 hover:bg-gray-100"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
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
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <div className="flex h-16 items-center justify-between border-b border-gray-200 bg-white px-4">
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg p-1 hover:bg-gray-100"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
          )}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">
              {currentConversation?.title || 'New Conversation'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="rounded-lg border border-gray-300 px-3 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            >
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
              <option value="claude-3-opus-20240229">Claude 3 Opus</option>
            </select>
            <button
              onClick={() => setParametersOpen(!parametersOpen)}
              className="rounded-lg border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50"
            >
              ⚙️ Parameters
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex flex-1 overflow-hidden">
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
