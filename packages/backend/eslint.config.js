import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import path from 'node:path';
import globals from 'globals';

export default tseslint.config(
  // Base recommended rules
  js.configs.recommended,
  ...tseslint.configs.recommended,

  // Global configuration
  {
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        tsconfigRootDir: path.resolve(new URL('.', import.meta.url).pathname),
        project: ['./tsconfig.json'],
      },
      globals: {
        ...globals.node,
        ...globals.es2022,
      },
    },
  },

  // Custom rules
  {
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },

  // Ignore patterns
  {
    ignores: ['dist/**', 'node_modules/**'],
  }
);
