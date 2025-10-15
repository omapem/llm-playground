import * as React from 'react';
import { ToastContext } from './use-toast';

export type ToastVariant = 'default' | 'destructive';

type ToastItem = { id: string; message: string; variant: ToastVariant };

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = React.useState<ToastItem[]>([]);

  const push = React.useCallback((message: string, variant: ToastVariant = 'default') => {
    const id = crypto.randomUUID();
    setItems((prev) => [...prev, { id, message, variant }]);
    window.setTimeout(() => {
      setItems((prev) => prev.filter((i) => i.id !== id));
    }, 3000);
  }, []);

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        {items.map((t) => (
          <div
            key={t.id}
            className={`rounded-md px-3 py-2 text-sm shadow ${
              t.variant === 'destructive' ? 'bg-red-600 text-white' : 'bg-slate-900 text-white'
            }`}
          >
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
