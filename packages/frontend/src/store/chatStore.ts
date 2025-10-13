import { create, StateCreator } from 'zustand';
import { persist, PersistOptions } from 'zustand/middleware';
import { ChatState, Conversation, Message } from '@/types';

interface ChatStore extends ChatState {
  addMessage: (message: Message) => void;
  updateMessage: (id: string, content: string) => void;
  appendToMessage: (id: string, content: string) => void;
  createConversation: () => void;
  setCurrentConversation: (conversationId: string) => void;
  deleteConversation: (conversationId: string) => void;
  setIsStreaming: (isStreaming: boolean) => void;
  updateParameters: (parameters: Partial<ChatState['parameters']>) => void;
  setSelectedModel: (model: string) => void;
  loadConversationMessages: (conversationId: string) => void;
  selectedModel: string;
}

// Separate messages storage by conversation
const conversationMessages = new Map<string, Message[]>();

type ChatStoreCreator = StateCreator<
  ChatStore,
  [['zustand/persist', unknown]],
  [],
  ChatStore
>;

const chatStoreCreator: ChatStoreCreator = (set, get) => ({
      conversations: [],
      currentConversation: null,
      messages: [],
      isStreaming: false,
      selectedModel: 'gpt-3.5-turbo',
      parameters: {
        temperature: 0.7,
        maxTokens: 2000,
        topP: 1,
      },

      addMessage: (message: Message) => {
        const state = get();
        const conversationId = message.conversationId;

        // Store in memory map
        const existing = conversationMessages.get(conversationId) || [];
        conversationMessages.set(conversationId, [...existing, message]);

        // Update state if it's the current conversation
        if (state.currentConversation?.id === conversationId) {
          set({ messages: [...state.messages, message] });
        }

        // Update conversation metadata
        set((state) => ({
          conversations: state.conversations.map((conv) =>
            conv.id === conversationId
              ? {
                  ...conv,
                  messageCount: conv.messageCount + 1,
                  updatedAt: Date.now(),
                }
              : conv
          ),
        }));
      },

      updateMessage: (id: string, content: string) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, content } : msg
          ),
        })),

      appendToMessage: (id: string, content: string) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, content: msg.content + content } : msg
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

      setCurrentConversation: (conversationId: string) => {
        const state = get();
        const conversation = state.conversations.find(
          (c) => c.id === conversationId
        );

        if (conversation) {
          const messages = conversationMessages.get(conversationId) || [];
          set({
            currentConversation: conversation,
            messages,
          });
        }
      },

      deleteConversation: (conversationId: string) => {
        conversationMessages.delete(conversationId);
        set((state) => ({
          conversations: state.conversations.filter(
            (c) => c.id !== conversationId
          ),
          currentConversation:
            state.currentConversation?.id === conversationId
              ? null
              : state.currentConversation,
          messages:
            state.currentConversation?.id === conversationId
              ? []
              : state.messages,
        }));
      },

      loadConversationMessages: (conversationId: string) => {
        const messages = conversationMessages.get(conversationId) || [];
        set({ messages });
      },

      setIsStreaming: (isStreaming: boolean) => set({ isStreaming }),

      setSelectedModel: (model: string) => set({ selectedModel: model }),

      updateParameters: (parameters: Partial<ChatState['parameters']>) =>
        set((state) => ({
          parameters: { ...state.parameters, ...parameters },
        })),
});

const persistConfig: PersistOptions<ChatStore, Pick<ChatStore, 'conversations' | 'parameters' | 'selectedModel'>> = {
  name: 'llm-playground-storage',
  partialize: (state) => ({
    conversations: state.conversations,
    parameters: state.parameters,
    selectedModel: state.selectedModel,
  }),
};

export const useChatStore = create<ChatStore>()(
  persist(chatStoreCreator, persistConfig)
);
