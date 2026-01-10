'use client';

/**
 * TokenizationInspector Component
 * Main interface for visualizing and analyzing tokenization
 */

import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { TokenizationResponse } from '@/types/tokenization';

export function TokenizationInspector() {
  const [text, setText] = useState('');
  const [tokenizerType, setTokenizerType] = useState<'bpe' | 'huggingface'>('huggingface');
  const [result, setResult] = useState<TokenizationResponse | null>(null);

  const tokenizeMutation = useMutation({
    mutationFn: async () => {
      const response = await apiClient.tokenize(text, tokenizerType);
      return response;
    },
    onSuccess: (data) => {
      setResult(data);
    },
    onError: (error) => {
      console.error('Tokenization failed:', error);
    },
  });

  const handleTokenize = () => {
    if (text.trim()) {
      tokenizeMutation.mutate();
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Tokenization Inspector</h1>
        <p className="text-gray-600">
          Explore how text is converted into tokens and understand tokenization strategies
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Text to Tokenize</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to tokenize..."
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
            rows={4}
          />
        </div>

        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-2">Tokenizer Type</label>
            <select
              value={tokenizerType}
              onChange={(e) => setTokenizerType(e.target.value as 'bpe' | 'huggingface')}
              className="w-full p-2 border border-gray-300 rounded-lg"
            >
              <option value="huggingface">HuggingFace (Production)</option>
              <option value="bpe">BPE Educational</option>
            </select>
          </div>

          <button
            onClick={handleTokenize}
            disabled={tokenizeMutation.isPending || !text.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition"
          >
            {tokenizeMutation.isPending ? 'Tokenizing...' : 'Tokenize'}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Summary Stats */}
          <div className="bg-blue-50 rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatBox label="Tokens" value={result.summary.token_count} />
              <StatBox label="Characters" value={result.summary.character_count} />
              <StatBox
                label="Compression"
                value={result.summary.compression_ratio.toFixed(2)}
              />
              <StatBox
                label="Avg Token Length"
                value={result.summary.avg_token_length.toFixed(2)}
              />
            </div>
          </div>

          {/* Token Visualization */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Tokens</h2>
            <div className="flex flex-wrap gap-2">
              {result.tokens.map((token, idx) => (
                <TokenBadge key={idx} token={token} />
              ))}
            </div>
          </div>

          {/* Token IDs */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Token IDs</h2>
            <div className="bg-gray-100 p-4 rounded font-mono text-sm overflow-auto">
              {result.token_ids.join(', ')}
            </div>
          </div>

          {/* Token Details Table */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Token Details</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-4">ID</th>
                    <th className="text-left py-2 px-4">Token</th>
                    <th className="text-left py-2 px-4">Characters</th>
                    <th className="text-left py-2 px-4">Type</th>
                  </tr>
                </thead>
                <tbody>
                  {result.tokens.map((token, idx) => (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="py-2 px-4">{token.id}</td>
                      <td className="py-2 px-4 font-mono">{token.text}</td>
                      <td className="py-2 px-4">
                        {token.end - token.start}
                      </td>
                      <td className="py-2 px-4">
                        <span
                          className={
                            token.special ? 'text-orange-600' : 'text-green-600'
                          }
                        >
                          {token.special ? 'Special' : 'Regular'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {tokenizeMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          Error: {tokenizeMutation.error instanceof Error ? tokenizeMutation.error.message : 'Unknown error'}
        </div>
      )}
    </div>
  );
}

interface StatBoxProps {
  label: string;
  value: string | number;
}

function StatBox({ label, value }: StatBoxProps) {
  return (
    <div className="text-center">
      <p className="text-gray-600 text-sm">{label}</p>
      <p className="text-2xl font-bold text-blue-600">{value}</p>
    </div>
  );
}

interface TokenBadgeProps {
  token: {
    text: string;
    id: number;
    special?: boolean;
  };
}

function TokenBadge({ token }: TokenBadgeProps) {
  return (
    <span
      className={`px-3 py-1 rounded-full text-sm font-mono ${
        token.special
          ? 'bg-orange-100 text-orange-800'
          : 'bg-blue-100 text-blue-800'
      }`}
      title={`ID: ${token.id}`}
    >
      {token.text}
    </span>
  );
}
