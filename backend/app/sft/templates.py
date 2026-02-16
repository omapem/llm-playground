"""Prompt templates for supervised fine-tuning.

This module provides template classes for formatting training examples into
instruction-following prompts. Different templates support different formats
like Alpaca-style instructions and multi-turn chat conversations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class PromptTemplate(ABC):
    """Abstract base class for prompt templates.

    A prompt template defines how to format a training example
    (instruction, input, output) into a text string suitable for
    language model training.
    """

    @abstractmethod
    def format(self, example: Dict[str, Any]) -> str:
        """Format a training example into a prompt string.

        Args:
            example: Dictionary containing the training example fields.
                     The expected keys depend on the specific template.

        Returns:
            Formatted prompt string ready for tokenization.

        Raises:
            ValueError: If required fields are missing.
        """
        pass


class AlpacaTemplate(PromptTemplate):
    """Template for Alpaca-style instruction-following format.

    Format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    If input is empty, the Input section is omitted.

    Example:
        >>> template = AlpacaTemplate()
        >>> example = {
        ...     "instruction": "Translate to French",
        ...     "input": "Hello world",
        ...     "output": "Bonjour le monde"
        ... }
        >>> print(template.format(example))
        ### Instruction:
        Translate to French

        ### Input:
        Hello world

        ### Response:
        Bonjour le monde
    """

    def format(self, example: Dict[str, Any]) -> str:
        """Format an example in Alpaca instruction format.

        Args:
            example: Dictionary with keys 'instruction', 'input', 'output'.

        Returns:
            Formatted prompt string.

        Raises:
            ValueError: If required fields are missing.
        """
        # Validate required fields
        required_fields = ["instruction", "output"]
        missing_fields = [f for f in required_fields if f not in example]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        # Build prompt parts
        parts = [
            "### Instruction:",
            instruction,
            "",
        ]

        # Only include input section if input is non-empty
        if input_text and input_text.strip():
            parts.extend([
                "### Input:",
                input_text,
                "",
            ])

        parts.extend([
            "### Response:",
            output,
        ])

        return "\n".join(parts)


class ChatTemplate(PromptTemplate):
    """Template for multi-turn chat conversation format.

    Supports multiple chat formats:
    - ChatML: OpenAI's format with <|im_start|> and <|im_end|> tags
    - Llama-2: Meta's format with [INST] and [/INST] tags

    Example (ChatML):
        >>> template = ChatTemplate(format="chatml")
        >>> example = {
        ...     "messages": [
        ...         {"role": "user", "content": "Hello!"},
        ...         {"role": "assistant", "content": "Hi there!"}
        ...     ]
        ... }
        >>> print(template.format(example))
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>

    Example (Llama-2):
        >>> template = ChatTemplate(format="llama2")
        >>> example = {
        ...     "messages": [
        ...         {"role": "user", "content": "Hello!"},
        ...         {"role": "assistant", "content": "Hi there!"}
        ...     ]
        ... }
        >>> print(template.format(example))
        [INST] Hello! [/INST]
        Hi there!
    """

    def __init__(self, format: str = "chatml"):
        """Initialize chat template.

        Args:
            format: Chat format to use ('chatml' or 'llama2').

        Raises:
            ValueError: If format is not supported.
        """
        if format not in ["chatml", "llama2"]:
            raise ValueError(f"Unsupported chat format: {format}. Use 'chatml' or 'llama2'")
        self.chat_format = format

    def format(self, example: Dict[str, Any]) -> str:
        """Format a chat conversation.

        Args:
            example: Dictionary with key 'messages' containing a list of
                     message dicts with 'role' and 'content' keys.

        Returns:
            Formatted chat string.

        Raises:
            ValueError: If required fields are missing or messages is empty.
        """
        if "messages" not in example:
            raise ValueError("Missing required field: messages")

        messages = example["messages"]
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if self.chat_format == "chatml":
            return self._format_chatml(messages)
        else:  # llama2
            return self._format_llama2(messages)

    def _format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in ChatML format.

        Args:
            messages: List of message dictionaries.

        Returns:
            ChatML formatted string.
        """
        formatted_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(formatted_parts)

    def _format_llama2(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Llama-2 format.

        Args:
            messages: List of message dictionaries.

        Returns:
            Llama-2 formatted string.
        """
        formatted_parts = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "user":
                user_content = msg["content"]
                # Check if there's an assistant response
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    assistant_content = messages[i + 1]["content"]
                    formatted_parts.append(f"[INST] {user_content} [/INST]\n{assistant_content}")
                    i += 2  # Skip both user and assistant messages
                else:
                    # User message without assistant response
                    formatted_parts.append(f"[INST] {user_content} [/INST]")
                    i += 1
            else:
                # Standalone assistant message (unusual but handle it)
                formatted_parts.append(msg["content"])
                i += 1

        return "\n".join(formatted_parts)


class TemplateRegistry:
    """Registry for managing prompt templates.

    Provides a centralized way to register and retrieve prompt templates
    by name. Includes default templates (alpaca, chatml, llama2) and
    supports custom template registration.

    Example:
        >>> registry = TemplateRegistry()
        >>> template = registry.get("alpaca")
        >>> example = {"instruction": "Say hello", "output": "Hello!"}
        >>> print(template.format(example))
        ### Instruction:
        Say hello

        ### Response:
        Hello!
    """

    def __init__(self):
        """Initialize the template registry with default templates."""
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default templates."""
        self._templates["alpaca"] = AlpacaTemplate()
        self._templates["chatml"] = ChatTemplate(format="chatml")
        self._templates["llama2"] = ChatTemplate(format="llama2")

    def register(self, name: str, template: PromptTemplate) -> None:
        """Register a custom template.

        Args:
            name: Name to register the template under.
            template: PromptTemplate instance to register.
        """
        self._templates[name] = template

    def get(self, name: str) -> PromptTemplate:
        """Get a template by name.

        Args:
            name: Name of the template to retrieve.

        Returns:
            The registered PromptTemplate.

        Raises:
            KeyError: If template name is not registered.
        """
        if name not in self._templates:
            raise KeyError(
                f"Template '{name}' not found. "
                f"Available templates: {list(self._templates.keys())}"
            )
        return self._templates[name]

    def list_templates(self) -> List[str]:
        """List all registered template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())
