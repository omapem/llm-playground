// Tokenizer utilities: prefer tiktoken for OpenAI models, fallback heuristic otherwise
import { createRequire } from 'module';
// Attempt to load tiktoken encoding lazily; safe in ESM using createRequire
let cl100kEncoding: { encode: (s: string) => number[] } | null = null;
let initTried = false;
function initTiktoken() {
  if (initTried) return;
  initTried = true;
  try {
    const require = createRequire(import.meta.url);
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { encoding_for_model } = require('@dqbd/tiktoken');
    cl100kEncoding = encoding_for_model('gpt-3.5-turbo');
    if (process.env.DEBUG_TOKENS) {
      // eslint-disable-next-line no-console
      console.log('[tokenizer] Loaded tiktoken encoding');
    }
  } catch (e) {
    if (process.env.DEBUG_TOKENS) {
      console.warn(
        '[tokenizer] Failed to load tiktoken, using heuristic fallback:',
        (e as Error).message
      );
    }
    cl100kEncoding = null;
  }
}

function isOpenAIModel(model?: string) {
  if (!model) return false;
  return /gpt-|o1|text-davinci|gpt4/i.test(model);
}

export function estimateTokens(text: string, model?: string): number {
  if (!text) return 0;
  if (isOpenAIModel(model)) {
    initTiktoken();
    if (cl100kEncoding) {
      try {
        const tokens = cl100kEncoding.encode(text);
        return tokens.length;
      } catch (e) {
        if (process.env.DEBUG_TOKENS) {
          console.warn('[tokenizer] encode failed, fallback heuristic:', (e as Error).message);
        }
      }
    }
  }
  // Heuristic fallback: ~4 chars/token
  return Math.max(1, Math.ceil(text.length / 4));
}

export function estimateMessageTokens(
  messages: { role: string; content: string }[],
  model?: string
): number {
  return messages.reduce((sum, m) => sum + estimateTokens(m.content, model), 0);
}
