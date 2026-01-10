/**
 * API client for communication with backend
 */

import axios, { AxiosInstance } from 'axios';
import {
  TokenizationResponse,
  TokenComparisonResponse,
  CostEstimate,
  VocabularyCoverage,
} from '@/types/tokenization';

class APIClient {
  private client: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000/api/v1') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Tokenize text using specified tokenizer
   */
  async tokenize(
    text: string,
    tokenizerType: string = 'huggingface'
  ): Promise<TokenizationResponse> {
    const response = await this.client.post('/tokenization/encode', {
      text,
      tokenizer_type: tokenizerType,
    });
    return response.data;
  }

  /**
   * Compare tokenization results from two tokenizers
   */
  async compareTokenizers(
    text: string,
    tokenizer1: string = 'bpe',
    tokenizer2: string = 'huggingface'
  ): Promise<TokenComparisonResponse> {
    const response = await this.client.post('/tokenization/compare', {
      text,
      tokenizer1,
      tokenizer2,
    });
    return response.data;
  }

  /**
   * Train a tokenizer on provided texts
   */
  async trainTokenizer(
    texts: string[],
    vocabSize: number = 50257,
    tokenizerType: string = 'huggingface'
  ) {
    const response = await this.client.post('/tokenization/train', {
      texts,
      vocab_size: vocabSize,
      tokenizer_type: tokenizerType,
      min_frequency: 2,
    });
    return response.data;
  }

  /**
   * Analyze vocabulary coverage on corpus
   */
  async analyzeCoverage(
    texts: string[],
    tokenizerType: string = 'huggingface'
  ): Promise<VocabularyCoverage> {
    const response = await this.client.post('/tokenization/coverage', {
      texts,
      tokenizer_type: tokenizerType,
    });
    return response.data;
  }

  /**
   * Estimate tokenization cost
   */
  async estimateCost(
    text: string,
    costPerToken: number = 0.0001,
    tokenizerType: string = 'huggingface'
  ): Promise<CostEstimate> {
    const response = await this.client.post('/tokenization/estimate-cost', {
      text,
      cost_per_token: costPerToken,
      tokenizer_type: tokenizerType,
    });
    return response.data;
  }

  /**
   * Health check for tokenization service
   */
  async healthCheck() {
    const response = await this.client.get('/tokenization/health');
    return response.data;
  }
}

export const apiClient = new APIClient();
