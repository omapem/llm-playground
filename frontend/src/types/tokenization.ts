/**
 * Type definitions for tokenization API responses
 */

export interface Token {
  id: number;
  text: string;
  start: number;
  end: number;
  special?: boolean;
}

export interface TokenizationSummary {
  token_count: number;
  character_count: number;
  special_tokens: number;
  regular_tokens: number;
  avg_token_length: number;
  compression_ratio: number;
}

export interface TokenizationResponse {
  original_text: string;
  tokens: Token[];
  token_ids: number[];
  summary: TokenizationSummary;
}

export interface TokenComparisonResponse {
  text: string;
  tokenizer1_tokens: string[];
  tokenizer2_tokens: string[];
  tokenizer1_count: number;
  tokenizer2_count: number;
  difference: number;
  shared_tokens: number;
}

export interface CostEstimate {
  token_count: number;
  cost_per_token: number;
  estimated_cost: number;
  characters: number;
  avg_characters_per_token: number;
}

export interface VocabularyCoverage {
  total_tokens: number;
  unique_tokens: number;
  unknown_tokens: number;
  coverage: number;
  oov_rate: number;
}
