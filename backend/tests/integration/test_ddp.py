"""Tests for Distributed Data Parallel (DDP) training support.

Tests DDP utilities (distributed.py), Trainer DDP integration, and
the standalone train_script module. All tests run without multiple GPUs
by mocking distributed primitives where needed.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset

from app.training.distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    create_distributed_dataloader,
    reduce_mean,
)
from app.training import Trainer, TrainingConfig
from app.transformer import TransformerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 100, seq_len: int = 128, vocab_size: int = 50257):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def training_dataset():
    """Create training dataset."""
    return SimpleDataset(size=100, seq_len=128)


@pytest.fixture
def small_training_config(temp_checkpoint_dir):
    """Create a minimal training config for fast tests."""
    return TrainingConfig(
        model_config=TransformerConfig(
            vocab_size=50257,
            hidden_size=64,
            num_layers=1,
            num_heads=1,
            max_position_embeddings=128,
        ),
        batch_size=4,
        learning_rate=1e-3,
        max_steps=5,
        warmup_steps=2,
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=5,
        logging_steps=5,
    )


# ---------------------------------------------------------------------------
# Tests for distributed.py — DistributedConfig
# ---------------------------------------------------------------------------


class TestDistributedConfig:
    """Tests for the DistributedConfig dataclass."""

    def test_is_main_process_rank_zero(self):
        """is_main_process should return True when rank is 0."""
        config = DistributedConfig(world_size=4, rank=0, local_rank=0)
        assert config.is_main_process is True

    def test_is_main_process_rank_nonzero(self):
        """is_main_process should return False when rank > 0."""
        config = DistributedConfig(world_size=4, rank=1, local_rank=1)
        assert config.is_main_process is False

    def test_default_backend_is_nccl(self):
        """Default backend should be nccl."""
        config = DistributedConfig(world_size=2, rank=0, local_rank=0)
        assert config.backend == "nccl"

    def test_custom_backend(self):
        """Backend can be set to gloo."""
        config = DistributedConfig(world_size=2, rank=0, local_rank=0, backend="gloo")
        assert config.backend == "gloo"


# ---------------------------------------------------------------------------
# Tests for distributed.py — setup_distributed
# ---------------------------------------------------------------------------


class TestSetupDistributed:
    """Tests for the setup_distributed function."""

    def test_returns_none_when_env_vars_missing(self):
        """setup_distributed should return None when torchrun env vars are absent."""
        # Ensure env vars are not set
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
        with patch.dict(os.environ, env, clear=True):
            result = setup_distributed()
        assert result is None

    def test_returns_none_when_partial_env_vars(self):
        """setup_distributed should return None when only some env vars are set."""
        env = {"RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            result = setup_distributed()
        assert result is None

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.is_available", return_value=False)
    def test_setup_with_env_vars_gloo(self, mock_cuda, mock_init_pg):
        """setup_distributed should initialise with gloo when CUDA is unavailable."""
        env = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            result = setup_distributed()

        assert result is not None
        assert result.rank == 0
        assert result.world_size == 2
        assert result.local_rank == 0
        assert result.backend == "gloo"
        mock_init_pg.assert_called_once_with(backend="gloo", rank=0, world_size=2)

    def test_raises_on_invalid_env_vars(self):
        """setup_distributed should raise RuntimeError on non-integer env vars."""
        env = {"RANK": "not_a_number", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="Invalid distributed training env vars"):
                setup_distributed()

    @patch("torch.cuda.set_device")
    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.is_available", return_value=True)
    def test_setup_with_env_vars_nccl(self, mock_cuda, mock_init_pg, mock_set_device):
        """setup_distributed should initialise with nccl when CUDA is available."""
        env = {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            result = setup_distributed()

        assert result is not None
        assert result.rank == 1
        assert result.world_size == 4
        assert result.local_rank == 1
        assert result.backend == "nccl"
        mock_init_pg.assert_called_once_with(backend="nccl", rank=1, world_size=4)
        mock_set_device.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# Tests for distributed.py — cleanup_distributed
# ---------------------------------------------------------------------------


class TestCleanupDistributed:
    """Tests for the cleanup_distributed function."""

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_when_initialized(self, mock_destroy, mock_is_init):
        """cleanup_distributed should call destroy_process_group when initialized."""
        cleanup_distributed()
        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_when_not_initialized(self, mock_destroy, mock_is_init):
        """cleanup_distributed should not call destroy_process_group when not initialized."""
        cleanup_distributed()
        mock_destroy.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for distributed.py — create_distributed_dataloader
# ---------------------------------------------------------------------------


class TestCreateDistributedDataloader:
    """Tests for the create_distributed_dataloader function."""

    def test_no_dist_config_returns_regular_dataloader(self, training_dataset):
        """Without dist_config, should create a regular DataLoader."""
        loader = create_distributed_dataloader(
            training_dataset, batch_size=8, dist_config=None, shuffle=True,
        )
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 8
        # No DistributedSampler should be attached
        assert not isinstance(loader.sampler, DistributedSampler)

    @patch("torch.distributed.is_initialized", return_value=True)
    def test_with_dist_config_uses_distributed_sampler(self, mock_is_init, training_dataset):
        """With dist_config, DataLoader should use a DistributedSampler."""
        dist_config = DistributedConfig(world_size=2, rank=0, local_rank=0, backend="gloo")

        loader = create_distributed_dataloader(
            training_dataset, batch_size=4, dist_config=dist_config, shuffle=True,
        )
        assert isinstance(loader, DataLoader)
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.batch_size == 4

    @patch("torch.distributed.is_initialized", return_value=True)
    def test_distributed_sampler_uses_correct_rank(self, mock_is_init, training_dataset):
        """DistributedSampler should be configured with the correct rank and world_size."""
        dist_config = DistributedConfig(world_size=4, rank=2, local_rank=2, backend="gloo")

        loader = create_distributed_dataloader(
            training_dataset, batch_size=4, dist_config=dist_config, shuffle=False,
        )
        sampler = loader.sampler
        assert isinstance(sampler, DistributedSampler)
        assert sampler.num_replicas == 4
        assert sampler.rank == 2
        assert sampler.shuffle is False

    def test_pin_memory_passed_through(self, training_dataset):
        """pin_memory parameter should be forwarded to the DataLoader."""
        loader = create_distributed_dataloader(
            training_dataset, batch_size=4, pin_memory=True,
        )
        assert loader.pin_memory is True


# ---------------------------------------------------------------------------
# Tests for distributed.py — reduce_mean
# ---------------------------------------------------------------------------


class TestReduceMean:
    """Tests for the reduce_mean function."""

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_returns_unchanged_when_not_distributed(self, mock_is_init):
        """reduce_mean should return the tensor unchanged when dist is not initialized."""
        tensor = torch.tensor(3.14)
        result = reduce_mean(tensor)
        assert torch.allclose(result, torch.tensor(3.14))

    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_reduce_mean_divides_by_world_size(self, mock_is_init, mock_all_reduce, mock_ws):
        """reduce_mean should divide the tensor by world_size after all_reduce."""
        tensor = torch.tensor(8.0)
        result = reduce_mean(tensor)
        mock_all_reduce.assert_called_once()
        # After all_reduce (which is mocked and does nothing), tensor is divided by 4
        assert torch.allclose(result, torch.tensor(2.0))

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_reduce_mean_does_not_mutate_input(self, mock_is_init, mock_all_reduce, mock_ws):
        """reduce_mean should not modify the original input tensor."""
        original_value = 4.0
        tensor = torch.tensor(original_value)
        _ = reduce_mean(tensor)
        # Original tensor should be unchanged
        assert torch.allclose(tensor, torch.tensor(original_value))


# ---------------------------------------------------------------------------
# Tests for Trainer DDP integration
# ---------------------------------------------------------------------------


class TestTrainerDDPIntegration:
    """Tests for DDP support in the Trainer class."""

    def test_trainer_accepts_dist_config_none(self, small_training_config, training_dataset):
        """Trainer should accept dist_config=None (default) without error."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
            dist_config=None,
        )
        assert trainer.dist_config is None

    def test_trainer_default_dist_config_is_none(self, small_training_config, training_dataset):
        """Trainer should default dist_config to None when not provided."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
        )
        assert trainer.dist_config is None

    def test_is_main_process_without_dist(self, small_training_config, training_dataset):
        """is_main_process should return True when dist_config is None."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
        )
        assert trainer.is_main_process is True

    def test_is_main_process_rank_zero(self, small_training_config, training_dataset):
        """is_main_process should return True when rank is 0."""
        dist_config = DistributedConfig(world_size=2, rank=0, local_rank=0, backend="gloo")

        # Mock DDP so it returns the model unchanged (no process group needed)
        mock_ddp = MagicMock(side_effect=lambda model, **kw: model)
        with patch("torch.nn.parallel.DistributedDataParallel", mock_ddp):
            trainer = Trainer(
                config=small_training_config,
                train_dataset=training_dataset,
                dist_config=dist_config,
            )
        assert trainer.is_main_process is True

    def test_is_main_process_rank_nonzero(self, small_training_config, training_dataset):
        """is_main_process should return False when rank > 0."""
        dist_config = DistributedConfig(world_size=2, rank=1, local_rank=1, backend="gloo")

        mock_ddp = MagicMock(side_effect=lambda model, **kw: model)
        with patch("torch.nn.parallel.DistributedDataParallel", mock_ddp):
            trainer = Trainer(
                config=small_training_config,
                train_dataset=training_dataset,
                dist_config=dist_config,
            )
        assert trainer.is_main_process is False

    def test_get_unwrapped_model_no_ddp(self, small_training_config, training_dataset):
        """_get_unwrapped_model should return the model directly when no DDP."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
        )
        unwrapped = trainer._get_unwrapped_model()
        assert unwrapped is trainer.model

    def test_get_unwrapped_model_with_ddp_mock(self, small_training_config, training_dataset):
        """_get_unwrapped_model should return model.module when wrapped in DDP."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
        )
        # Simulate DDP wrapping by attaching a .module attribute
        original_model = trainer.model
        wrapper = MagicMock()
        wrapper.module = original_model
        trainer.model = wrapper

        unwrapped = trainer._get_unwrapped_model()
        assert unwrapped is original_model

    def test_trainer_trains_without_dist(self, small_training_config, training_dataset):
        """Trainer should still train normally with dist_config=None."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
            dist_config=None,
        )
        trainer.train()
        assert trainer.current_step == small_training_config.max_steps

    def test_trainer_uses_distributed_dataloader(self, small_training_config, training_dataset):
        """With dist_config, Trainer should use DistributedSampler in its DataLoader."""
        dist_config = DistributedConfig(world_size=2, rank=0, local_rank=0, backend="gloo")

        mock_ddp = MagicMock(side_effect=lambda model, **kw: model)
        with patch("torch.nn.parallel.DistributedDataParallel", mock_ddp):
            trainer = Trainer(
                config=small_training_config,
                train_dataset=training_dataset,
                dist_config=dist_config,
            )
        assert isinstance(trainer.train_loader.sampler, DistributedSampler)

    def test_trainer_calls_set_epoch_on_distributed_sampler(self, small_training_config, training_dataset):
        """Trainer should call set_epoch on DistributedSampler when iterating epochs."""
        dist_config = DistributedConfig(world_size=2, rank=0, local_rank=0, backend="gloo")

        mock_ddp = MagicMock(side_effect=lambda model, **kw: model)
        with patch("torch.nn.parallel.DistributedDataParallel", mock_ddp):
            trainer = Trainer(
                config=small_training_config,
                train_dataset=training_dataset,
                dist_config=dist_config,
            )

        # Verify sampler has set_epoch capability
        assert hasattr(trainer.train_loader.sampler, "set_epoch")

        # Spy on set_epoch
        original_set_epoch = trainer.train_loader.sampler.set_epoch
        call_args = []
        def spy_set_epoch(epoch):
            call_args.append(epoch)
            return original_set_epoch(epoch)
        trainer.train_loader.sampler.set_epoch = spy_set_epoch

        trainer.train()

        # set_epoch should have been called at least once (epoch 0 at start)
        assert len(call_args) >= 1
        assert call_args[0] == 0

    def test_trainer_checkpoint_uses_unwrapped_model(self, small_training_config, training_dataset):
        """_save_checkpoint should use the unwrapped model (no 'module.' prefix)."""
        trainer = Trainer(
            config=small_training_config,
            train_dataset=training_dataset,
        )
        # Run a single step so we have a loss
        trainer.train()

        # Load the last checkpoint and verify keys have no 'module.' prefix
        from pathlib import Path
        checkpoints = list(Path(small_training_config.checkpoint_dir).glob("checkpoint_step_*.pt"))
        assert len(checkpoints) >= 1
        ckpt = torch.load(str(checkpoints[0]), weights_only=False)
        for key in ckpt["model_state_dict"].keys():
            assert not key.startswith("module."), (
                f"Checkpoint key should not have 'module.' prefix: {key}"
            )


# ---------------------------------------------------------------------------
# Tests for train_script module
# ---------------------------------------------------------------------------


class TestTrainScript:
    """Tests for the standalone train_script module."""

    def test_train_script_is_importable(self):
        """train_script.py should be importable as a module."""
        import app.training.train_script as ts
        assert hasattr(ts, "main")

    def test_train_script_has_main_function(self):
        """train_script.main should be a callable."""
        from app.training.train_script import main
        assert callable(main)
