"""Tests for SFT prompt templates."""

import pytest
from app.sft.templates import (
    PromptTemplate,
    AlpacaTemplate,
    ChatTemplate,
    TemplateRegistry,
)


def test_prompt_template_is_abstract():
    """Test PromptTemplate cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PromptTemplate()


def test_alpaca_template_format_with_input():
    """Test AlpacaTemplate formats instruction with input."""
    template = AlpacaTemplate()
    example = {
        "instruction": "Translate to French",
        "input": "Hello world",
        "output": "Bonjour le monde",
    }

    result = template.format(example)

    assert "### Instruction:" in result
    assert "Translate to French" in result
    assert "### Input:" in result
    assert "Hello world" in result
    assert "### Response:" in result
    assert "Bonjour le monde" in result


def test_alpaca_template_format_without_input():
    """Test AlpacaTemplate formats instruction without input."""
    template = AlpacaTemplate()
    example = {
        "instruction": "What is AI?",
        "input": "",
        "output": "AI is artificial intelligence",
    }

    result = template.format(example)

    assert "### Instruction:" in result
    assert "What is AI?" in result
    assert "### Input:" not in result  # Should skip empty input
    assert "### Response:" in result
    assert "AI is artificial intelligence" in result


def test_alpaca_template_missing_fields():
    """Test AlpacaTemplate raises error for missing required fields."""
    template = AlpacaTemplate()

    # Missing instruction
    with pytest.raises(ValueError, match="Missing required fields"):
        template.format({"output": "test"})

    # Missing output
    with pytest.raises(ValueError, match="Missing required fields"):
        template.format({"instruction": "test"})


def test_chat_template_chatml_format():
    """Test ChatTemplate formats multi-turn conversation in ChatML format."""
    template = ChatTemplate(format="chatml")
    example = {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]
    }

    result = template.format(example)

    assert "<|im_start|>user" in result
    assert "Hello!<|im_end|>" in result
    assert "<|im_start|>assistant" in result
    assert "Hi there!<|im_end|>" in result
    assert "How are you?<|im_end|>" in result
    assert "I'm doing well!<|im_end|>" in result


def test_chat_template_llama2_format():
    """Test ChatTemplate formats conversation in Llama-2 format."""
    template = ChatTemplate(format="llama2")
    example = {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]
    }

    result = template.format(example)

    assert "[INST] Hello! [/INST]" in result
    assert "Hi there!" in result
    assert "[INST] How are you? [/INST]" in result
    assert "I'm doing well!" in result


def test_chat_template_unsupported_format():
    """Test ChatTemplate raises error for unsupported format."""
    with pytest.raises(ValueError, match="Unsupported chat format"):
        ChatTemplate(format="invalid_format")


def test_chat_template_empty_messages():
    """Test ChatTemplate raises error for empty messages list."""
    template = ChatTemplate(format="chatml")

    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        template.format({"messages": []})


def test_chat_template_missing_messages_field():
    """Test ChatTemplate raises error when messages field is missing."""
    template = ChatTemplate(format="chatml")

    with pytest.raises(ValueError, match="Missing required field: messages"):
        template.format({"conversation": []})


def test_template_registry_get_default_templates():
    """Test TemplateRegistry provides default templates."""
    registry = TemplateRegistry()

    # Check default templates exist
    alpaca = registry.get("alpaca")
    chatml = registry.get("chatml")
    llama2 = registry.get("llama2")

    assert isinstance(alpaca, AlpacaTemplate)
    assert isinstance(chatml, ChatTemplate)
    assert isinstance(llama2, ChatTemplate)


def test_template_registry_register_custom():
    """Test registering a custom template."""
    registry = TemplateRegistry()
    custom_template = AlpacaTemplate()

    registry.register("my_custom_template", custom_template)

    retrieved = registry.get("my_custom_template")
    assert retrieved is custom_template


def test_template_registry_get_nonexistent():
    """Test getting non-existent template raises KeyError."""
    registry = TemplateRegistry()

    with pytest.raises(KeyError, match="Template 'nonexistent' not found"):
        registry.get("nonexistent")


def test_template_registry_list_templates():
    """Test listing all registered templates."""
    registry = TemplateRegistry()

    templates = registry.list_templates()

    assert "alpaca" in templates
    assert "chatml" in templates
    assert "llama2" in templates
    assert len(templates) == 3
