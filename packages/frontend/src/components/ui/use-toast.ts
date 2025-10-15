import * as React from 'react'
import type { ToastVariant } from './toast'

type ToastContextType = {
  push: (message: string, variant?: ToastVariant) => void
} | null

export const ToastContext = React.createContext<ToastContextType>(null)

export function useToast() {
  const ctx = React.useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within a ToastProvider')
  return ctx
}
