"""API routes for transformer architecture components."""

from typing import Optional

from fastapi import APIRouter, HTTPException
import torch

from ..transformer import (
    TransformerConfig,
    ModelConfigurator,
    GPT2Small,
    GPT2Medium,
    Llama7B,
    AttentionVisualization,
)
from .models import (
    TransformerConfigRequest,
    ParameterCountResponse,
    AttentionVisualizationRequest,
    AttentionVisualizationResponse,
    ArchitectureInfoResponse,
)

router = APIRouter(prefix="/api/v1/transformer", tags=["transformer"])


@router.get("/presets", tags=["architecture"])
async def get_presets():
    """Get available architecture presets.

    Returns:
        Available preset configurations
    """
    presets = {
        "gpt2_small": {
            "name": "GPT-2 Small",
            "description": "GPT-2 small configuration (12 layers, 768 hidden)",
            "params": ModelConfigurator.get_architecture_info(GPT2Small()),
        },
        "gpt2_medium": {
            "name": "GPT-2 Medium",
            "description": "GPT-2 medium configuration (24 layers, 1024 hidden)",
            "params": ModelConfigurator.get_architecture_info(GPT2Medium()),
        },
        "llama_7b": {
            "name": "Llama 7B",
            "description": "Llama 7B-like configuration",
            "params": ModelConfigurator.get_architecture_info(Llama7B()),
        },
    }
    return {"presets": presets}


@router.post("/configure", response_model=ArchitectureInfoResponse, tags=["architecture"])
async def configure_model(request: TransformerConfigRequest):
    """Configure a transformer model and get architecture details.

    Args:
        request: Configuration request with model parameters

    Returns:
        Detailed architecture information
    """
    try:
        # Handle preset configurations
        if request.preset:
            if request.preset == "gpt2_small":
                config = GPT2Small()
            elif request.preset == "gpt2_medium":
                config = GPT2Medium()
            elif request.preset == "llama_7b":
                config = Llama7B()
            else:
                raise ValueError(f"Unknown preset: {request.preset}")
        else:
            # Create custom configuration
            config = TransformerConfig(
                hidden_size=request.hidden_size,
                num_layers=request.num_layers,
                num_heads=request.num_heads,
                intermediate_size=request.intermediate_size,
                vocab_size=request.vocab_size,
                max_position_embeddings=request.max_position_embeddings,
                activation=request.activation,
                dropout_rate=request.dropout_rate,
                position_encoding_type=request.position_encoding_type,
            )

        # Validate configuration
        config.validate()

        # Get architecture info
        info = ModelConfigurator.get_architecture_info(config)

        return ArchitectureInfoResponse(**info)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")


@router.post("/parameter-count", response_model=ParameterCountResponse, tags=["architecture"])
async def count_parameters(request: TransformerConfigRequest):
    """Count parameters for a configuration.

    Args:
        request: Configuration request

    Returns:
        Parameter count and breakdown
    """
    try:
        # Handle preset configurations
        if request.preset:
            if request.preset == "gpt2_small":
                config = GPT2Small()
            elif request.preset == "gpt2_medium":
                config = GPT2Medium()
            elif request.preset == "llama_7b":
                config = Llama7B()
            else:
                raise ValueError(f"Unknown preset: {request.preset}")
        else:
            config = TransformerConfig(
                hidden_size=request.hidden_size,
                num_layers=request.num_layers,
                num_heads=request.num_heads,
                intermediate_size=request.intermediate_size,
                vocab_size=request.vocab_size,
                max_position_embeddings=request.max_position_embeddings,
                activation=request.activation,
                dropout_rate=request.dropout_rate,
                position_encoding_type=request.position_encoding_type,
            )

        config.validate()

        # Create model and count parameters
        model = ModelConfigurator.create_model(config)
        total_params = ModelConfigurator.count_parameters(model)
        model_size_mb = ModelConfigurator.get_model_size_mb(model)

        # Get layer info
        head_dim = config.get_head_dim()
        layer_info = {
            "num_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "head_dim": head_dim,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
        }

        # Parameter distribution
        param_dist = {
            "attention_params_per_layer": 3 * (config.hidden_size ** 2 + config.hidden_size)
            + config.hidden_size ** 2
            + config.hidden_size,
            "ffn_params_per_layer": 2 * config.hidden_size * config.intermediate_size
            + config.hidden_size
            + config.intermediate_size,
            "norm_params_per_layer": 4 * config.hidden_size,  # 2 LayerNorms with weight + bias
            "total_attention_params": config.num_layers
            * (3 * (config.hidden_size ** 2 + config.hidden_size)
               + config.hidden_size ** 2
               + config.hidden_size),
            "total_ffn_params": config.num_layers
            * (2 * config.hidden_size * config.intermediate_size
               + config.hidden_size
               + config.intermediate_size),
            "total_norm_params": config.num_layers * 4 * config.hidden_size,
        }

        seq_info = {
            "max_position_embeddings": config.max_position_embeddings,
            "vocabulary_size": config.vocab_size,
        }

        return ParameterCountResponse(
            total_parameters=total_params,
            model_size_mb=model_size_mb,
            layer_info=layer_info,
            parameter_distribution=param_dist,
            sequence_info=seq_info,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parameter counting error: {str(e)}")


@router.post("/visualize/attention", response_model=AttentionVisualizationResponse, tags=["visualization"])
async def visualize_attention(request: AttentionVisualizationRequest):
    """Generate attention visualization data.

    Creates synthetic attention patterns for visualization purposes.

    Args:
        request: Visualization request with layer and head info

    Returns:
        Visualization data with attention matrix and patterns
    """
    try:
        # Generate synthetic attention weights
        # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        num_layers = 12
        batch_size = 1
        num_heads = 12
        seq_len = 16

        if request.tokens:
            seq_len = len(request.tokens)

        # Create synthetic attention weights
        # Initialize with uniform attention + some learned patterns
        attention_weights = torch.randn(
            num_layers, batch_size, num_heads, seq_len, seq_len
        )
        # Normalize to get attention probabilities
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # Generate labels
        if request.tokens:
            labels = request.tokens[:seq_len]
        else:
            labels = [f"pos_{i}" for i in range(seq_len)]

        # Format visualization
        viz_data = AttentionVisualization.format_for_visualization(
            attention_weights,
            layer=request.layer,
            head=request.head,
            sequence=labels,
        )

        return AttentionVisualizationResponse(**viz_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@router.get("/compare", tags=["architecture"])
async def compare_architectures(configs: Optional[str] = None):
    """Compare multiple architecture configurations.

    Args:
        configs: Optional comma-separated preset names (default: all presets)

    Returns:
        Comparison of architectures
    """
    try:
        if configs:
            preset_names = [c.strip() for c in configs.split(",")]
        else:
            preset_names = ["gpt2_small", "gpt2_medium", "llama_7b"]

        comparison = {}
        for name in preset_names:
            if name == "gpt2_small":
                config = GPT2Small()
            elif name == "gpt2_medium":
                config = GPT2Medium()
            elif name == "llama_7b":
                config = Llama7B()
            else:
                continue

            comparison[name] = ModelConfigurator.get_architecture_info(config)

        return {
            "comparison": comparison,
            "total_configs": len(comparison),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")


@router.get("/health", tags=["health"])
async def transformer_health():
    """Health check for transformer module."""
    return {
        "status": "ok",
        "module": "transformer",
        "capabilities": [
            "attention_mechanism",
            "transformer_blocks",
            "position_encoding",
            "model_configuration",
            "visualization",
            "parameter_counting",
        ],
    }
