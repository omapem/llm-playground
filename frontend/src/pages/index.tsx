/**
 * Home page
 */

import React from 'react';
import { TokenizationInspector } from '@/components/TokenizationInspector';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <nav className="bg-white shadow">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900">LLM Playground</h1>
        </div>
      </nav>

      <TokenizationInspector />
    </main>
  );
}
