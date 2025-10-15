import { useEffect, useRef, useState } from 'react';
import { useChatStore } from '@/store/chatStore';
import { Button, buttonVariants } from './ui/button';
import { Card } from './ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from './ui/dialog';
import { cn } from '@/lib/utils';
import { apiClient } from '@/services/api';
import { useToast } from '@/components/ui/use-toast';
import { Trash2, MoreVertical, Pencil } from 'lucide-react';

export default function ConversationList() {
  const {
    conversations,
    currentConversation,
    setCurrentConversation,
    createConversation,
    renameConversation,
  } = useChatStore();
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [pendingRenameId, setPendingRenameId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState<string>('');
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number } | null>(null);
  const { push } = useToast();
  const menuWrapperRef = useRef<HTMLDivElement | null>(null);
  const menuContentRef = useRef<HTMLDivElement | null>(null);

  // Close the kebab menu on outside click or Escape
  useEffect(() => {
    if (!menuOpenId) return;
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      const inButton = !!(
        menuWrapperRef.current &&
        target &&
        menuWrapperRef.current.contains(target)
      );
      const inMenu = !!(
        menuContentRef.current &&
        target &&
        menuContentRef.current.contains(target)
      );
      if (!inButton && !inMenu) {
        setMenuOpenId(null);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpenId(null);
    };
    const onScrollAny = () => setMenuOpenId(null);
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('scroll', onScrollAny, true);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('scroll', onScrollAny, true);
    };
  }, [menuOpenId]);

  return (
    <div className="flex flex-col min-h-0 flex-1">
      <div className="p-4">
        <Button className="w-full" onClick={() => createConversation()}>
          + New Conversation
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {conversations.length === 0 ? (
          <Card className="p-4 text-center text-sm text-gray-500">
            No conversations yet. Create one to get started!
          </Card>
        ) : (
          <div className="space-y-2">
            {conversations.map((conversation) => {
              const isSelected = currentConversation?.id === conversation.id;
              return (
                <div
                  key={conversation.id}
                  role="button"
                  tabIndex={0}
                  className={cn(
                    buttonVariants({ variant: isSelected ? 'default' : 'ghost', size: 'sm' }),
                    'group w-full grid grid-cols-[1fr_28px] gap-3 text-sm rounded-md text-left conv-list-padding items-start h-auto py-2'
                  )}
                  onClick={() => setCurrentConversation(conversation.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') setCurrentConversation(conversation.id);
                  }}
                >
                  <div className="flex-1 text-left min-w-0">
                    <div className="truncate">{conversation.title}</div>
                    <div className="mt-0.5 text-xs opacity-60 flex gap-2">
                      <span>{conversation.messageCount} msgs</span>
                      <span>· {new Date(conversation.updatedAt).toLocaleDateString()}</span>
                      {conversation.totalCost !== undefined && (
                        <span
                          title={`Total cost (server aggregated): $${(conversation.totalCost || 0).toFixed(6)}`}
                        >
                          · ${(conversation.totalCost || 0).toFixed(4)}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Kebab menu + dialogs */}
                  <div
                    className="relative justify-self-end self-start -mt-1"
                    ref={menuOpenId === conversation.id ? menuWrapperRef : undefined}
                  >
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 text-slate-400 hover:text-slate-700 hover:bg-slate-50 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                      onClick={(e) => {
                        e.stopPropagation();
                        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                        setMenuPos({ top: rect.top, left: rect.right + 8 });
                        setMenuOpenId((prev) =>
                          prev === conversation.id ? null : conversation.id
                        );
                      }}
                      title="Conversation actions"
                      aria-label="Conversation actions"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>

                    {menuOpenId === conversation.id && menuPos && (
                      <div
                        ref={menuContentRef}
                        className="fixed z-[9999] w-36 rounded-md border border-slate-200 bg-white shadow-md py-1 text-slate-800"
                        style={{ top: menuPos.top, left: menuPos.left }}
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          className="flex w-full items-center gap-2 px-3 py-1.5 text-sm hover:bg-slate-50"
                          onClick={() => {
                            setMenuOpenId(null);
                            setPendingRenameId(conversation.id);
                            setRenameDraft(conversation.title);
                          }}
                        >
                          <Pencil className="h-4 w-4" />
                          <span>Rename</span>
                        </button>
                        <button
                          className="flex w-full items-center gap-2 px-3 py-1.5 text-sm text-red-600 hover:bg-red-50"
                          onClick={() => {
                            setMenuOpenId(null);
                            setPendingDeleteId(conversation.id);
                          }}
                        >
                          <Trash2 className="h-4 w-4" />
                          <span>Delete</span>
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Delete confirmation dialog */}
                  <Dialog
                    open={pendingDeleteId === conversation.id}
                    onOpenChange={(open) => setPendingDeleteId(open ? conversation.id : null)}
                  >
                    <DialogContent onClick={(e) => e.stopPropagation()}>
                      <DialogHeader>
                        <DialogTitle>Delete conversation</DialogTitle>
                        <DialogDescription>
                          This action cannot be undone. This will permanently delete "
                          {conversation.title}" and all of its messages.
                        </DialogDescription>
                      </DialogHeader>
                      <DialogFooter>
                        <DialogClose asChild>
                          <Button variant="outline">Cancel</Button>
                        </DialogClose>
                        <Button
                          variant="destructive"
                          onClick={async (e) => {
                            e.stopPropagation();
                            setPendingDeleteId(null);
                            // Optimistic update
                            const prev = useChatStore.getState();
                            const prevConversations = prev.conversations;
                            const prevCurrent = prev.currentConversation;
                            const prevMessages = prev.messages;
                            const filtered = prevConversations.filter(
                              (c) => c.id !== conversation.id
                            );
                            useChatStore.setState({
                              conversations: filtered,
                              currentConversation:
                                prevCurrent?.id === conversation.id ? null : prevCurrent,
                              messages: prevCurrent?.id === conversation.id ? [] : prevMessages,
                            });
                            try {
                              await apiClient.deleteConversation(conversation.id);
                              // Reconcile with server
                              const raw = await apiClient.getConversations();
                              useChatStore.setState({
                                conversations: raw.map((c) => ({
                                  id: c.id,
                                  title: c.title || 'Untitled',
                                  createdAt: new Date(c.createdAt).getTime(),
                                  updatedAt: new Date(c.updatedAt).getTime(),
                                  messageCount: c.messageCount ?? 0,
                                  totalCost: c.totalCost ?? 0,
                                })),
                              });
                              push('Conversation deleted');
                            } catch {
                              // Rollback on failure
                              useChatStore.setState({
                                conversations: prevConversations,
                                currentConversation: prevCurrent,
                                messages: prevMessages,
                              });
                              push('Failed to delete conversation', 'destructive');
                            }
                          }}
                        >
                          Delete
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>

                  {/* Rename dialog */}
                  <Dialog
                    open={pendingRenameId === conversation.id}
                    onOpenChange={(open) => setPendingRenameId(open ? conversation.id : null)}
                  >
                    <DialogContent onClick={(e) => e.stopPropagation()}>
                      <DialogHeader>
                        <DialogTitle>Rename conversation</DialogTitle>
                        <DialogDescription>
                          Choose a new name for this conversation.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="mt-2">
                        <input
                          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-200"
                          value={renameDraft}
                          onChange={(e) => setRenameDraft(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>
                      <DialogFooter>
                        <DialogClose asChild>
                          <Button variant="outline">Cancel</Button>
                        </DialogClose>
                        <Button
                          onClick={async (e) => {
                            e.stopPropagation();
                            const title = (renameDraft || '').trim() || 'Untitled';
                            try {
                              await renameConversation(conversation.id, title);
                              push('Conversation renamed');
                            } catch {
                              push('Failed to rename conversation', 'destructive');
                            } finally {
                              setPendingRenameId(null);
                            }
                          }}
                        >
                          Save
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
