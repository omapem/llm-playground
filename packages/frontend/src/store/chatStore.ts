import { create } from 'zustand';
import { ChatState, Conversation, Message } from '@/types';

interface ChatStore extends ChatState {
  addMessage: (message: Message) => void;
  updateMessage: (id: string, content: string) => void;
  createConversation: () => void;
  setCurrentConversation: (conversation: Conversation) => void;
  setIsStreaming: (isStreaming: boolean) => void;
  updateParameters: (parameters: Partial<ChatState['parameters']>) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  conversations: [],
  currentConversation: null,
  messages: [],
  isStreaming: false,
  parameters: {
    temperature: 0.7,
    maxTokens: 2000,
    topP: 1,
  },

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  updateMessage: (id, content) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content } : msg
      ),
    })),

  createConversation: () => {
    const newConversation: Conversation = {
      id: crypto.randomUUID(),
      title: 'New Conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messageCount: 0,
    };

    set((state) => ({
      conversations: [newConversation, ...state.conversations],
      currentConversation: newConversation,
      messages: [],
    }));
  },

  setCurrentConversation: (conversation) =>
    set({ currentConversation: conversation }),

  setIsStreaming: (isStreaming) => set({ isStreaming }),

  updateParameters: (parameters) =>
    set((state) => ({
      parameters: { ...state.parameters, ...parameters },
    })),
}));
