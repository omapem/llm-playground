import { create } from 'zustand';

import { ChatState, Conversation, Message } from '@/types';

interface BackendConversation {
  id: string;
  title?: string | null;
  createdAt: string;
  updatedAt: string;
  messageCount?: number;
  totalCost?: number;
  messages?: { cost?: number | null }[];
}

interface BackendMessage {
  id: string;
  conversationId: string;
  role: string;
  content: string;
  model?: string;
  createdAt?: string;
  tokens?: number;
  cost?: number;
  parameters?: Record<string, unknown>;
}

function mapBackendConversation(conv: BackendConversation): Conversation {
  const totalCost =
    conv.totalCost ?? (conv.messages ? conv.messages.reduce((s, m) => s + (m.cost ?? 0), 0) : 0);
  return {
    id: conv.id,
    title: (conv.title ?? 'Untitled') as string,
    createdAt: new Date(conv.createdAt).getTime(),
    updatedAt: new Date(conv.updatedAt).getTime(),
    messageCount: conv.messageCount ?? (conv.messages ? conv.messages.length : 0),
    totalCost,
  };
}

function mapBackendMessage(msg: BackendMessage): Message {
  const role = (
    ['user', 'assistant', 'system'].includes(msg.role) ? msg.role : 'assistant'
  ) as Message['role'];
  return {
    id: msg.id,
    conversationId: msg.conversationId,
    role,
    content: msg.content,
    model: msg.model,
    parameters: msg.parameters,
    timestamp: msg.createdAt ? new Date(msg.createdAt).getTime() : Date.now(),
    tokens: msg.tokens,
    cost: msg.cost,
  };
}
import { apiClient } from '../services/api';

export interface ChatStore extends ChatState {
  addMessage: (message: Omit<Message, 'id'>) => Promise<Message>;
  updateMessage: (id: string, content: string) => void;
  appendToMessage: (id: string, content: string) => void;
  updateMessageMeta: (
    id: string,
    meta: Partial<Pick<Message, 'tokens' | 'inputTokens' | 'outputTokens' | 'cost'>>
  ) => void;
  createConversation: (title?: string) => Promise<Conversation>;
  setCurrentConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  renameConversation: (conversationId: string, title: string) => Promise<void>;
  setIsStreaming: (isStreaming: boolean) => void;
  updateParameters: (parameters: Partial<ChatState['parameters']>) => void;
  setSelectedModel: (model: string) => void;
  loadConversationMessages: (conversationId: string) => Promise<void>;
  selectedModel: string;
  /** Tracks the assistant message id currently receiving streamed content */
  inProgressAssistantId: string | null;
  setInProgressAssistantId: (id: string | null) => void;
}

export const useChatStore = create<ChatStore>((set, get) => ({
  conversations: [],
  currentConversation: null,
  messages: [],
  isStreaming: false,
  inProgressAssistantId: null,
  selectedModel: 'gpt-3.5-turbo',
  parameters: {
    temperature: 0.7,
    maxTokens: 2000,
    topP: 1,
  },
  addMessage: async (message: Omit<Message, 'id'>) => {
    const state = get();
    if (!state.currentConversation) throw new Error('No active conversation');
    const rawMsg = await apiClient.addMessage(state.currentConversation.id, message);
    const newMsg = mapBackendMessage(rawMsg);
    set((s) => ({ messages: [...s.messages, newMsg] }));
    const rawConvs = await apiClient.getConversations();
    const conversations = (rawConvs as BackendConversation[]).map(mapBackendConversation);
    set({ conversations });
    return newMsg;
  },
  updateMessage: (id, content) =>
    set((state) => ({
      messages: state.messages.map((msg) => (msg.id === id ? { ...msg, content } : msg)),
    })),
  updateMessageMeta: (id, meta) =>
    set((state) => ({
      messages: state.messages.map((msg) => (msg.id === id ? { ...msg, ...meta } : msg)),
    })),
  appendToMessage: (id, content) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content: msg.content + content } : msg
      ),
    })),
  createConversation: async (title?: string) => {
    const rawConv = await apiClient.createConversation(title || 'New Conversation');
    const newConv: Conversation = mapBackendConversation(rawConv);
    const rawConvs = await apiClient.getConversations();
    const conversations = (rawConvs as BackendConversation[]).map(mapBackendConversation);
    set({ conversations, currentConversation: newConv, messages: [] });
    return newConv;
  },
  setCurrentConversation: async (conversationId: string) => {
    const state = get();
    const conversation = state.conversations.find((c) => c.id === conversationId);
    if (conversation) {
      const rawMsgs = await apiClient.getMessages(conversationId);
      const messages = (rawMsgs as BackendMessage[]).map(mapBackendMessage);
      set({ currentConversation: conversation, messages });
    }
  },
  deleteConversation: async (conversationId: string) => {
    await apiClient.deleteConversation(conversationId);
    const rawConvs = await apiClient.getConversations();
    const conversations = (rawConvs as BackendConversation[]).map(mapBackendConversation);
    set((state) => ({
      conversations,
      currentConversation:
        state.currentConversation?.id === conversationId ? null : state.currentConversation,
      messages: state.currentConversation?.id === conversationId ? [] : state.messages,
    }));
  },
  renameConversation: async (conversationId: string, title: string) => {
    // Optimistic update
    const prev = get();
    const prevConversations = prev.conversations;
    const prevCurrent = prev.currentConversation;
    const updatedList = prevConversations.map((c) =>
      c.id === conversationId ? { ...c, title } : c
    );
    set({
      conversations: updatedList,
      currentConversation:
        prevCurrent?.id === conversationId ? { ...prevCurrent, title } : prevCurrent,
    });
    try {
      await apiClient.renameConversation(conversationId, title);
      const raw = await apiClient.getConversations();
      set({ conversations: (raw as BackendConversation[]).map(mapBackendConversation) });
    } catch (e) {
      // rollback
      set({ conversations: prevConversations, currentConversation: prevCurrent });
      throw e;
    }
  },
  loadConversationMessages: async (conversationId: string) => {
    const rawMsgs = await apiClient.getMessages(conversationId);
    const messages = (rawMsgs as BackendMessage[]).map(mapBackendMessage);
    set({ messages });
  },
  setIsStreaming: (isStreaming: boolean) => set({ isStreaming }),
  setInProgressAssistantId: (id: string | null) => set({ inProgressAssistantId: id }),
  setSelectedModel: (model: string) => set({ selectedModel: model }),
  updateParameters: (parameters: Partial<ChatState['parameters']>) =>
    set((state) => ({
      parameters: { ...state.parameters, ...parameters },
    })),
}));

// Initial load of existing conversations + (optional) preload first conversation's messages
(async () => {
  try {
    const rawConvs = await apiClient.getConversations();
    const mapped = (rawConvs as BackendConversation[]).map((conv) => ({
      id: conv.id,
      title: conv.title || 'Untitled',
      createdAt: new Date(conv.createdAt).getTime(),
      updatedAt: new Date(conv.updatedAt).getTime(),
      messageCount: conv.messages ? conv.messages.length : (conv.messageCount ?? 0),
      totalCost:
        conv.totalCost ??
        (conv.messages ? conv.messages.reduce((s: number, m) => s + (m.cost ?? 0), 0) : 0),
    }));
    useChatStore.setState({ conversations: mapped });
  } catch (e) {
    console.warn('Failed to preload conversations:', e);
  }
})();
