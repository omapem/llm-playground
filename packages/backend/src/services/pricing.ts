export interface ModelPricing {
  inputPer1K: number; // USD per 1K input tokens
  outputPer1K: number; // USD per 1K output tokens
}

// Simplified pricing table (update with real values as needed)
export const PRICING: Record<string, ModelPricing> = {
  'gpt-3.5-turbo': { inputPer1K: 0.0005, outputPer1K: 0.0015 },
  'gpt-4': { inputPer1K: 0.03, outputPer1K: 0.06 },
  'claude-3-5-sonnet-20241022': { inputPer1K: 0.003, outputPer1K: 0.015 },
  'claude-3-opus-20240229': { inputPer1K: 0.015, outputPer1K: 0.075 },
};

export function estimateCost(model: string, inputTokens: number, outputTokens: number): number {
  const pricing = PRICING[model];
  if (!pricing) return 0;
  return (inputTokens / 1000) * pricing.inputPer1K + (outputTokens / 1000) * pricing.outputPer1K;
}
