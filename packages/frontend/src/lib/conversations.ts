export type BackendConversation = {
  id: string;
  title?: string | null;
  createdAt: string;
  updatedAt: string;
  messageCount?: number;
  totalCost?: number;
  messages?: { cost?: number | null }[];
};

export function mapConversationsForStore(raw: BackendConversation[]) {
  return raw.map((c) => ({
    id: c.id,
    title: (c.title ?? 'Untitled') as string,
    createdAt: new Date(c.createdAt).getTime(),
    updatedAt: new Date(c.updatedAt).getTime(),
    messageCount: c.messageCount ?? (c.messages ? c.messages.length : 0),
    totalCost:
      c.totalCost ?? (c.messages ? c.messages.reduce((s, m) => s + (m.cost ?? 0), 0) : 0),
  }));
}
