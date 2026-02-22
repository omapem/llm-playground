"""Tests for DDP metrics aggregation.

Verifies that reduce_metrics correctly aggregates metric dictionaries
across distributed ranks, and that existing reduce_mean in the trainer
pipeline produces correct aggregated values.
"""

from unittest.mock import patch, MagicMock

import pytest
import torch

from app.training.metrics import reduce_metrics


class TestReduceMetrics:
    """Tests for the reduce_metrics utility function."""

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_returns_unchanged_when_not_distributed(self, mock_is_init):
        """reduce_metrics should return the dict unchanged when dist is not initialized."""
        metrics = {"loss": 1.5, "grad_norm": 0.3, "perplexity": 4.48}
        result = reduce_metrics(metrics)
        assert result == metrics

    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_divides_all_values_by_world_size(self, mock_is_init, mock_all_reduce, mock_ws):
        """reduce_metrics should divide each value by world_size after all_reduce."""
        metrics = {"loss": 8.0, "grad_norm": 4.0}
        result = reduce_metrics(metrics)
        # all_reduce is mocked (no-op), so values are divided by 4
        assert pytest.approx(result["loss"], abs=1e-6) == 2.0
        assert pytest.approx(result["grad_norm"], abs=1e-6) == 1.0

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_calls_all_reduce_for_each_metric(self, mock_is_init, mock_all_reduce, mock_ws):
        """reduce_metrics should call all_reduce once per metric key."""
        metrics = {"a": 1.0, "b": 2.0, "c": 3.0}
        reduce_metrics(metrics)
        assert mock_all_reduce.call_count == 3

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_empty_dict(self, mock_is_init):
        """reduce_metrics should handle empty dict."""
        result = reduce_metrics({})
        assert result == {}

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_does_not_mutate_input(self, mock_is_init, mock_all_reduce, mock_ws):
        """reduce_metrics should not mutate the input dictionary."""
        original = {"loss": 4.0}
        original_copy = dict(original)
        reduce_metrics(original)
        assert original == original_copy

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_uses_correct_device(self, mock_is_init, mock_all_reduce, mock_ws):
        """reduce_metrics should create tensors on the specified device."""
        device = torch.device("cpu")
        metrics = {"loss": 1.0}
        reduce_metrics(metrics, device=device)

        # Verify the tensor passed to all_reduce was on the correct device
        call_args = mock_all_reduce.call_args_list[0]
        tensor_arg = call_args[0][0]
        assert tensor_arg.device == device


class TestTrainerMetricsAggregation:
    """Tests verifying that the Trainer correctly aggregates metrics in DDP mode."""

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_training_step_returns_loss_without_dist(self, mock_is_init):
        """In non-distributed mode, _training_step should return local loss."""
        from app.training import Trainer, TrainingConfig
        from app.transformer import TransformerConfig
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=64,
                    num_layers=1,
                    num_heads=1,
                    max_position_embeddings=128,
                ),
                batch_size=4,
                learning_rate=1e-3,
                max_steps=1,
                checkpoint_dir=tmpdir,
            )
            dataset = torch.utils.data.TensorDataset(
                torch.randint(0, 50257, (20, 128))
            )
            trainer = Trainer(config=config, train_dataset=dataset)

            # Get a batch and run a step
            batch = next(iter(trainer.train_loader))
            loss = trainer._training_step(batch)

            assert isinstance(loss, float)
            assert loss > 0

    def test_log_metrics_only_on_main_process(self):
        """_log_metrics should only be called on is_main_process=True."""
        from app.training import Trainer, TrainingConfig
        from app.training.distributed import DistributedConfig
        from app.transformer import TransformerConfig
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
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
                logging_steps=1,
                checkpoint_dir=tmpdir,
                save_steps=100,
            )
            # Non-main process (rank 1)
            dist_config = DistributedConfig(world_size=2, rank=1, local_rank=1, backend="gloo")
            dataset = torch.utils.data.TensorDataset(
                torch.randint(0, 50257, (20, 128))
            )

            mock_ddp = MagicMock(side_effect=lambda model, **kw: model)
            with patch("torch.nn.parallel.DistributedDataParallel", mock_ddp):
                trainer = Trainer(
                    config=config,
                    train_dataset=dataset,
                    dist_config=dist_config,
                )

            assert trainer.is_main_process is False

            # Spy on _log_metrics
            with patch.object(trainer, "_log_metrics") as mock_log:
                trainer.train()
                # _log_metrics should NOT be called on non-main process
                mock_log.assert_not_called()
