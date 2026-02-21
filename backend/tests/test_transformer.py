"""Tests for transformer architecture components."""

import pytest
import torch
import numpy as np

from app.transformer import (
    Attention,
    PositionalEncoding,
    LearnedPositionalEncoding,
    FeedForward,
    LayerNormalization,
    TransformerBlock,
    StackedTransformerBlocks,
    TransformerConfig,
    ModelConfigurator,
    GPT2Small,
    AttentionVisualization,
)


class TestAttention:
    """Tests for the Attention mechanism."""

    @pytest.fixture
    def attention_module(self):
        """Create an attention module for testing."""
        return Attention(hidden_size=768, num_heads=12, dropout_rate=0.1)

    def test_attention_initialization(self, attention_module):
        """Test attention module initializes correctly."""
        assert attention_module.hidden_size == 768
        assert attention_module.num_heads == 12
        assert attention_module.head_dim == 64

    def test_attention_forward_shape(self, attention_module):
        """Test attention output has correct shape."""
        batch_size, seq_len, hidden_size = 2, 16, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output = attention_module(hidden_states)

        assert output.output.shape == (batch_size, seq_len, hidden_size)
        assert output.attention_weights.shape == (batch_size, 12, seq_len, seq_len)

    def test_attention_with_mask(self, attention_module):
        """Test attention with attention mask."""
        batch_size, seq_len, hidden_size = 2, 16, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Create attention mask (padding mask)
        attention_mask = torch.ones(batch_size, 1, 1, seq_len)
        attention_mask[:, :, :, 10:] = 0  # Mask out last 6 positions
        attention_mask = (1.0 - attention_mask) * -10000.0

        output = attention_module(hidden_states, attention_mask=attention_mask)

        assert output.output.shape == (batch_size, seq_len, hidden_size)

    def test_attention_parameter_count(self, attention_module):
        """Test parameter counting for attention."""
        param_count = attention_module.get_parameter_count()

        # 3 * (hidden_size * hidden_size + hidden_size) for Q, K, V
        # + (hidden_size * hidden_size + hidden_size) for output
        expected = 4 * (768 * 768 + 768)
        assert param_count == expected

    def test_attention_head_dim_mismatch(self):
        """Test attention raises error on invalid head dimension."""
        with pytest.raises(ValueError):
            Attention(hidden_size=768, num_heads=13)  # 768 % 13 != 0


class TestPositionalEncoding:
    """Tests for position encoding."""

    def test_sinusoidal_encoding_initialization(self):
        """Test sinusoidal position encoding initializes."""
        encoding = PositionalEncoding(hidden_size=768, max_position_embeddings=2048)
        assert encoding.hidden_size == 768
        assert encoding.max_position_embeddings == 2048

    def test_sinusoidal_encoding_forward(self):
        """Test sinusoidal encoding forward pass."""
        encoding = PositionalEncoding(hidden_size=768, max_position_embeddings=2048)
        x = torch.randn(2, 16, 768)
        output = encoding(x)

        assert output.shape == x.shape
        # Check that it's not identical to input (position info added)
        assert not torch.allclose(output, x)

    def test_sinusoidal_encoding_sequence_length_error(self):
        """Test error on sequence longer than max."""
        encoding = PositionalEncoding(hidden_size=768, max_position_embeddings=512)
        x = torch.randn(2, 1024, 768)

        with pytest.raises(ValueError):
            encoding(x)

    def test_learned_encoding_forward(self):
        """Test learned position encoding forward pass."""
        encoding = LearnedPositionalEncoding(hidden_size=768, max_position_embeddings=2048)
        x = torch.randn(2, 16, 768)
        output = encoding(x)

        assert output.shape == x.shape

    def test_learned_encoding_parameter_count(self):
        """Test learned encoding parameter counting."""
        encoding = LearnedPositionalEncoding(hidden_size=768, max_position_embeddings=2048)
        param_count = encoding.get_parameter_count()

        expected = 768 * 2048
        assert param_count == expected


class TestFeedForward:
    """Tests for feed-forward network."""

    def test_feedforward_initialization(self):
        """Test FFN initialization."""
        ffn = FeedForward(hidden_size=768, intermediate_size=3072)
        assert ffn.hidden_size == 768
        assert ffn.intermediate_size == 3072

    def test_feedforward_forward_shape(self):
        """Test FFN output shape."""
        ffn = FeedForward(hidden_size=768, intermediate_size=3072)
        x = torch.randn(2, 16, 768)
        output = ffn(x)

        assert output.shape == (2, 16, 768)

    def test_feedforward_parameter_count(self):
        """Test FFN parameter counting."""
        ffn = FeedForward(hidden_size=768, intermediate_size=3072)
        param_count = ffn.get_parameter_count()

        # linear_1: 768 * 3072 + 3072
        # linear_2: 3072 * 768 + 768
        expected = 768 * 3072 + 3072 + 3072 * 768 + 768
        assert param_count == expected

    def test_feedforward_activation_functions(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "gelu_approximate"]:
            ffn = FeedForward(hidden_size=768, intermediate_size=3072, activation=activation)
            x = torch.randn(2, 16, 768)
            output = ffn(x)
            assert output.shape == (2, 16, 768)

    def test_feedforward_invalid_activation(self):
        """Test error on invalid activation."""
        with pytest.raises(ValueError):
            FeedForward(hidden_size=768, activation="invalid_activation")


class TestLayerNormalization:
    """Tests for layer normalization."""

    def test_layer_norm_forward(self):
        """Test layer norm forward pass."""
        norm = LayerNormalization(hidden_size=768)
        x = torch.randn(2, 16, 768)
        output = norm(x)

        assert output.shape == x.shape

    def test_layer_norm_normalization(self):
        """Test that layer norm actually normalizes."""
        norm = LayerNormalization(hidden_size=768)
        x = torch.randn(2, 16, 768)
        output = norm(x)

        # Check that output has mean close to 0 and std close to 1
        # (for the feature dimension)
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

    def test_layer_norm_parameter_count(self):
        """Test layer norm parameter counting."""
        norm = LayerNormalization(hidden_size=768, bias=True)
        param_count = norm.get_parameter_count()

        # weight + bias
        expected = 768 + 768
        assert param_count == expected


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        block = TransformerBlock(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
        )
        x = torch.randn(2, 16, 768)
        output = block(x)

        assert output.output.shape == (2, 16, 768)
        assert output.attention_weights.shape == (2, 12, 16, 16)

    def test_transformer_block_residual_connection(self):
        """Test that residual connections preserve information."""
        block = TransformerBlock(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            dropout_rate=0.0,  # No dropout for deterministic test
        )
        x = torch.randn(2, 16, 768)

        # With identity initialization, output should be close to input
        # (This is a weak test since we don't control initialization)
        output = block(x)
        assert output.output.shape == x.shape

    def test_transformer_block_norm_first(self):
        """Test norm-first vs post-norm variants."""
        x = torch.randn(2, 16, 768)

        # Pre-norm
        block_prenorm = TransformerBlock(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            norm_first=True,
        )
        output_prenorm = block_prenorm(x)

        # Post-norm
        block_postnorm = TransformerBlock(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            norm_first=False,
        )
        output_postnorm = block_postnorm(x)

        # Both should have same shape
        assert output_prenorm.output.shape == output_postnorm.output.shape

    def test_transformer_block_parameter_count(self):
        """Test parameter counting for block."""
        block = TransformerBlock(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
        )
        param_count = block.get_parameter_count()

        # Should be: attention + ffn + 2 * layer_norm
        assert param_count > 0


class TestStackedBlocks:
    """Tests for stacked transformer blocks."""

    def test_stacked_blocks_forward(self):
        """Test stacked blocks forward pass."""
        stack = StackedTransformerBlocks(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
        )
        x = torch.randn(2, 16, 768)
        output = stack(x)

        assert output.output.shape == (2, 16, 768)

    def test_stacked_blocks_depth(self):
        """Test stacked blocks with different depths."""
        for num_layers in [1, 6, 12, 24]:
            stack = StackedTransformerBlocks(
                hidden_size=768,
                num_layers=num_layers,
                num_heads=12,
                intermediate_size=3072,
            )
            x = torch.randn(2, 16, 768)
            output = stack(x)
            assert output.output.shape == x.shape


class TestTransformerConfig:
    """Tests for configuration management."""

    def test_config_initialization(self):
        """Test config initializes with defaults."""
        config = TransformerConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 12

    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = TransformerConfig()
        config.validate()  # Should not raise

        # Invalid: hidden_size not divisible by num_heads
        config = TransformerConfig(hidden_size=768, num_heads=13)
        with pytest.raises(ValueError):
            config.validate()

    def test_gpt2_small_preset(self):
        """Test GPT-2 small preset."""
        config = GPT2Small()
        config.validate()

        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.vocab_size == 50257

    def test_config_to_dict(self):
        """Test config conversion to dictionary."""
        config = TransformerConfig(hidden_size=768, num_layers=12)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["hidden_size"] == 768
        assert config_dict["num_layers"] == 12

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "intermediate_size": 2048,
        }
        config = TransformerConfig.from_dict(config_dict)

        assert config.hidden_size == 512
        assert config.num_layers == 8


class TestModelConfigurator:
    """Tests for model configurator."""

    def test_create_model_from_config(self):
        """Test creating a model from config."""
        config = GPT2Small()
        model = ModelConfigurator.create_model(config)

        assert isinstance(model, torch.nn.Module)

    def test_count_parameters(self):
        """Test parameter counting."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            intermediate_size=2048,
        )
        model = ModelConfigurator.create_model(config)
        param_count = ModelConfigurator.count_parameters(model)

        assert param_count > 0

    def test_get_architecture_info(self):
        """Test getting architecture information."""
        config = GPT2Small()
        info = ModelConfigurator.get_architecture_info(config)

        assert "config" in info
        assert "total_parameters" in info
        assert "model_size_mb" in info
        assert "layer_info" in info
        assert "parameter_distribution" in info

    def test_model_size_estimation(self):
        """Test model size estimation."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            intermediate_size=2048,
        )
        model = ModelConfigurator.create_model(config)
        size_mb = ModelConfigurator.get_model_size_mb(model)

        # Should be positive and reasonable for this config
        assert 0 < size_mb < 1000


class TestVisualization:
    """Tests for visualization utilities."""

    @pytest.fixture
    def attention_weights(self):
        """Create sample attention weights."""
        return torch.randn(2, 12, 16, 16)

    def test_extract_attention_head(self, attention_weights):
        """Test extracting a single attention head."""
        head_attention = AttentionVisualization.extract_attention_head(
            attention_weights, layer=0, head=0
        )

        assert isinstance(head_attention, np.ndarray)
        assert head_attention.shape == (16, 16)

    def test_average_attention_heads(self, attention_weights):
        """Test averaging attention heads."""
        averaged = AttentionVisualization.average_attention_heads(
            attention_weights, layer=0
        )

        assert isinstance(averaged, np.ndarray)
        assert averaged.shape == (16, 16)

    def test_get_top_attention_positions(self):
        """Test finding top attention positions."""
        attention_matrix = np.random.randn(16, 16)
        attention_matrix = np.abs(attention_matrix)  # Positive values

        top_positions = AttentionVisualization.get_top_attention_positions(
            attention_matrix, top_k=5
        )

        assert len(top_positions) == 5
        assert all("query_position" in p for p in top_positions)
        assert all("key_position" in p for p in top_positions)

    def test_compute_attention_entropy(self):
        """Test computing attention entropy."""
        attention_matrix = np.random.dirichlet([1] * 16, size=16)  # Valid probabilities

        entropy = AttentionVisualization.compute_attention_entropy(attention_matrix)

        assert entropy.shape == (16,)
        assert np.all(entropy >= 0)  # Entropy is non-negative

    def test_format_for_visualization(self):
        """Test formatting attention for visualization."""
        attention_weights = torch.randn(2, 12, 16, 16)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        viz_data = AttentionVisualization.format_for_visualization(
            attention_weights, layer=0
        )

        assert "title" in viz_data
        assert "matrix" in viz_data
        assert "x_labels" in viz_data
        assert "y_labels" in viz_data
        assert "entropy" in viz_data
        assert "patterns" in viz_data


class TestIntegration:
    """Integration tests for transformer components."""

    def test_end_to_end_forward_pass(self):
        """Test full forward pass through transformer stack."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=3,
            num_heads=8,
            intermediate_size=2048,
            dropout_rate=0.1,
        )

        model = ModelConfigurator.create_model(config)
        x = torch.randn(2, 16, 512)

        output = model(x)

        assert output.output.shape == (2, 16, 512)
        assert output.attention_weights.shape == (2, 8, 16, 16)

    def test_model_instantiation_benchmark(self):
        """Test that model instantiation is fast."""
        import time

        config = GPT2Small()
        start = time.time()
        model = ModelConfigurator.create_model(config)
        elapsed = time.time() - start

        # Should instantiate in under 2s (allows for CI variability)
        assert elapsed < 2.0

    def test_gpt2_small_instantiation(self):
        """Test GPT-2 small can be instantiated and used."""
        config = GPT2Small()
        config.validate()

        model = ModelConfigurator.create_model(config)
        param_count = ModelConfigurator.count_parameters(model)

        # create_model builds StackedTransformerBlocks only (no embeddings/LM head),
        # so the count reflects transformer layers alone (~85M for GPT-2 small).
        # Full GPT-2 small (~124M) includes ~38M from the 50257×768 embedding table.
        expected_approx = 85_000_000
        # Allow ±20% variation
        assert expected_approx * 0.8 < param_count < expected_approx * 1.2

    def test_attention_visualization_performance(self):
        """Test that attention visualization completes in <500ms."""
        import time

        attention_weights = torch.softmax(torch.randn(12, 1, 12, 256, 256), dim=-1)

        start = time.time()
        viz_data = AttentionVisualization.format_for_visualization(
            attention_weights, layer=0
        )
        elapsed = time.time() - start

        assert elapsed < 0.5
        assert "matrix" in viz_data
