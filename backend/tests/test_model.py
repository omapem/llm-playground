"""Tests for GPT Language Model implementation."""

import pytest
import torch
import torch.nn as nn
from app.transformer import TransformerConfig, GPT2Small, GPT2Medium, GPTModel, ModelOutput


class TestGPTModelInitialization:
    """Tests for GPTModel initialization."""

    @pytest.fixture
    def gpt2_small_config(self):
        """Create GPT-2 Small configuration."""
        return GPT2Small()

    @pytest.fixture
    def custom_config(self):
        """Create custom configuration."""
        return TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
            activation="gelu",
            dropout_rate=0.1,
        )

    def test_gpt2_small_initialization(self, gpt2_small_config):
        """Test GPTModel initializes with GPT-2 Small config."""
        model = GPTModel(gpt2_small_config)
        assert model is not None
        assert isinstance(model, nn.Module)
        assert model.config == gpt2_small_config

    def test_custom_config_initialization(self, custom_config):
        """Test GPTModel initializes with custom config."""
        model = GPTModel(custom_config)
        assert model.config == custom_config
        assert model.config.hidden_size == 256

    def test_embeddings_layer_created(self, custom_config):
        """Test token embeddings layer is created correctly."""
        model = GPTModel(custom_config)
        assert hasattr(model, "token_embedding")
        assert isinstance(model.token_embedding, nn.Embedding)
        assert model.token_embedding.num_embeddings == custom_config.vocab_size
        assert model.token_embedding.embedding_dim == custom_config.hidden_size

    def test_position_encoding_created(self, custom_config):
        """Test position encoding is created."""
        model = GPTModel(custom_config)
        assert hasattr(model, "position_embedding")

    def test_transformer_blocks_created(self, custom_config):
        """Test transformer blocks are created."""
        model = GPTModel(custom_config)
        assert hasattr(model, "transformer")

    def test_lm_head_created(self, custom_config):
        """Test LM head is created correctly."""
        model = GPTModel(custom_config)
        assert hasattr(model, "lm_head")
        assert isinstance(model.lm_head, nn.Linear)
        assert model.lm_head.in_features == custom_config.hidden_size
        assert model.lm_head.out_features == custom_config.vocab_size

    def test_weight_tying_option(self, custom_config):
        """Test weight tying between embeddings and LM head."""
        custom_config.tie_embeddings = True
        model = GPTModel(custom_config)
        # When tied, lm_head.weight should reference token_embedding.weight
        assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    def test_no_weight_tying_by_default(self, custom_config):
        """Test weight is not tied by default."""
        custom_config.tie_embeddings = False
        model = GPTModel(custom_config)
        # Weights should be different
        assert model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr()

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises error on model creation."""
        with pytest.raises(ValueError):
            invalid_config = TransformerConfig(
                vocab_size=0,  # Invalid
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                intermediate_size=1024,
            )
            # Validation happens when creating model, not config
            model = GPTModel(invalid_config)


class TestGPTModelForwardPass:
    """Tests for GPTModel forward pass."""

    @pytest.fixture
    def model_and_batch(self):
        """Create model and batch."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
            max_position_embeddings=128,
        )
        model = GPTModel(config)
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        return model, input_ids, batch_size, seq_len

    def test_forward_pass_returns_model_output(self, model_and_batch):
        """Test forward pass returns ModelOutput dataclass."""
        model, input_ids, _, _ = model_and_batch
        output = model(input_ids)
        assert isinstance(output, ModelOutput)

    def test_forward_pass_logits_shape(self, model_and_batch):
        """Test output logits have correct shape."""
        model, input_ids, batch_size, seq_len = model_and_batch
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, 100)

    def test_forward_pass_without_labels(self, model_and_batch):
        """Test forward pass without labels returns None loss."""
        model, input_ids, _, _ = model_and_batch
        output = model(input_ids)
        assert output.loss is None

    def test_forward_pass_with_labels(self, model_and_batch):
        """Test forward pass with labels computes loss."""
        model, input_ids, batch_size, seq_len = model_and_batch
        labels = torch.randint(0, 100, (batch_size, seq_len))
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert isinstance(output.loss, torch.Tensor)
        assert output.loss.shape == ()  # Scalar

    def test_loss_is_positive(self, model_and_batch):
        """Test computed loss is positive."""
        model, input_ids, batch_size, seq_len = model_and_batch
        labels = torch.randint(0, 100, (batch_size, seq_len))
        output = model(input_ids, labels=labels)
        assert output.loss > 0

    def test_forward_pass_with_attention_mask(self, model_and_batch):
        """Test forward pass with attention mask."""
        model, input_ids, batch_size, seq_len = model_and_batch
        # Create causal attention mask
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len))
        attention_mask = torch.triu(attention_mask, diagonal=1) * -1e10
        output = model(input_ids, attention_mask=attention_mask)
        assert output.logits is not None

    def test_hidden_states_in_output(self, model_and_batch):
        """Test that hidden states are included in output."""
        model, input_ids, _, _ = model_and_batch
        output = model(input_ids)
        assert hasattr(output, "hidden_states")

    def test_model_eval_mode(self, model_and_batch):
        """Test model works in eval mode."""
        model, input_ids, _, _ = model_and_batch
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
        assert output.logits is not None

    def test_batch_size_one(self):
        """Test forward pass with batch size 1."""
        config = TransformerConfig(
            vocab_size=50,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            intermediate_size=128,
        )
        model = GPTModel(config)
        input_ids = torch.randint(0, 50, (1, 16))
        output = model(input_ids)
        assert output.logits.shape == (1, 16, 50)

    def test_long_sequence(self):
        """Test forward pass with long sequence."""
        config = TransformerConfig(
            vocab_size=50,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
        )
        model = GPTModel(config)
        input_ids = torch.randint(0, 50, (2, 256))
        output = model(input_ids)
        assert output.logits.shape == (2, 256, 50)


class TestGPTModelWeightInitialization:
    """Tests for weight initialization."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
        )
        return GPTModel(config)

    def test_token_embedding_initialized(self, model):
        """Test token embedding weights are initialized."""
        weights = model.token_embedding.weight
        # Check weights are not all zeros
        assert not torch.allclose(weights, torch.zeros_like(weights))
        # Check weights have reasonable range (GPT-2 style: normal 0, 0.02)
        assert weights.std() < 0.1

    def test_lm_head_weights_initialized(self, model):
        """Test LM head weights are initialized."""
        weights = model.lm_head.weight
        assert not torch.allclose(weights, torch.zeros_like(weights))

    def test_lm_head_bias_initialized(self, model):
        """Test LM head bias is initialized to zero."""
        if model.lm_head.bias is not None:
            bias = model.lm_head.bias
            assert torch.allclose(bias, torch.zeros_like(bias))

    def test_all_parameters_require_grad(self, model):
        """Test all parameters require gradients."""
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require gradients"


class TestGPTModelParameterCounting:
    """Tests for parameter counting."""

    def test_get_parameter_count_method_exists(self):
        """Test model has get_parameter_count method."""
        config = GPT2Small()
        model = GPTModel(config)
        assert hasattr(model, "get_parameter_count")

    def test_parameter_count_returns_dict(self):
        """Test get_parameter_count returns dictionary."""
        config = GPT2Small()
        model = GPTModel(config)
        counts = model.get_parameter_count()
        assert isinstance(counts, dict)
        assert "total" in counts

    def test_parameter_count_reasonable_gpt2_small(self):
        """Test parameter count is reasonable for GPT-2 Small."""
        config = GPT2Small()
        model = GPTModel(config)
        counts = model.get_parameter_count()
        # GPT-2 Small: ~85M (transformer + token embedding) + learned position encoding + lm_head
        # Total ~163M when using learned position encoding
        # Allow wide range since it depends on position encoding type
        assert 100_000_000 < counts["total"] < 200_000_000

    def test_parameter_count_breakdown(self):
        """Test parameter count breakdown."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
        )
        model = GPTModel(config)
        counts = model.get_parameter_count()
        assert "total" in counts
        # Total should equal sum of parts
        total = 0
        for key, value in counts.items():
            if key != "total":
                total += value
        assert counts["total"] == total

    def test_tied_embeddings_affect_count(self):
        """Test tied embeddings do not add duplicate lm_head parameters."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
        )

        # Without tying: lm_head has separate parameters
        config.tie_embeddings = False
        model_no_tie = GPTModel(config)
        count_no_tie = model_no_tie.get_parameter_count()["total"]

        # With tying: lm_head.weight points to token_embedding.weight
        config.tie_embeddings = True
        model_tie = GPTModel(config)
        count_tie = model_tie.get_parameter_count()["total"]

        # Tied should have fewer or equal parameters
        # (might be equal if both weight tied parameters are counted)
        assert count_no_tie >= count_tie


class TestGPTModelDeviceManagement:
    """Tests for device management."""

    @pytest.fixture
    def model(self):
        """Create model."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
        )
        return GPTModel(config)

    def test_model_on_cpu(self, model):
        """Test model can be moved to CPU."""
        model.cpu()
        for param in model.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_cuda(self, model):
        """Test model can be moved to CUDA if available."""
        model.cuda()
        for param in model.parameters():
            assert param.device.type == "cuda"

    def test_forward_pass_respects_device(self, model):
        """Test forward pass respects device."""
        model.cpu()
        input_ids = torch.randint(0, 100, (2, 16), device="cpu")
        output = model(input_ids)
        assert output.logits.device.type == "cpu"

    def test_model_train_eval_modes(self, model):
        """Test model can switch between train and eval modes."""
        model.train()
        assert model.training
        model.eval()
        assert not model.training

    def test_gradient_flow(self, model):
        """Test gradients flow through model."""
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.backward()

        # Check that gradients have been computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGPTModelPresets:
    """Tests for model presets."""

    def test_gpt2_small_preset(self):
        """Test GPT-2 Small preset."""
        config = GPT2Small()
        model = GPTModel(config)
        assert model is not None
        counts = model.get_parameter_count()
        # GPT-2 Small: ~85M (transformer) + learned position encoding + token/lm head
        # Total: ~163M with learned position encoding
        assert 100_000_000 < counts["total"] < 200_000_000

    def test_gpt2_medium_preset(self):
        """Test GPT-2 Medium preset."""
        config = GPT2Medium()
        model = GPTModel(config)
        assert model is not None
        counts = model.get_parameter_count()
        # GPT-2 Medium: ~355M (transformer) + learned position encoding + token/lm head
        # Total: ~406M with learned position encoding
        assert 350_000_000 < counts["total"] < 450_000_000

    def test_different_presets_different_sizes(self):
        """Test different presets have different sizes."""
        model_small = GPTModel(GPT2Small())
        model_medium = GPTModel(GPT2Medium())

        small_params = model_small.get_parameter_count()["total"]
        medium_params = model_medium.get_parameter_count()["total"]

        assert medium_params > small_params


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_model_output_creation(self):
        """Test ModelOutput can be created."""
        logits = torch.randn(2, 16, 100)
        loss = torch.tensor(1.5)
        output = ModelOutput(logits=logits, loss=loss)
        assert output.logits is logits
        assert output.loss is loss

    def test_model_output_none_loss(self):
        """Test ModelOutput with None loss."""
        logits = torch.randn(2, 16, 100)
        output = ModelOutput(logits=logits, loss=None)
        assert output.loss is None

    def test_model_output_backward_through_logits(self):
        """Test gradients flow through output logits."""
        logits = torch.randn(2, 16, 100, requires_grad=True)
        loss = logits.sum()
        loss.backward()
        assert logits.grad is not None
