"""Tests for SFT auto-validation integration.

Verifies that SFTTrainer automatically adds a ValidationCallback
when a validation dataset is available and none was manually provided.
"""

from unittest.mock import patch, MagicMock

import pytest

from app.sft.trainer import SFTTrainer
from app.sft.config import SFTConfig
from app.sft.callbacks import ValidationCallback, SFTCallback


@pytest.fixture
def minimal_sft_config(tmp_path):
    """Create a minimal SFT config for testing."""
    return SFTConfig(
        base_model="gpt2",
        output_dir=str(tmp_path / "output"),
        dataset_name="test",
        dataset_format="alpaca",
        validation_split=0.1,
        eval_steps=50,
        max_steps=10,
        batch_size=2,
    )


class TestAutoValidationIntegration:
    """Tests for automatic ValidationCallback integration in SFTTrainer."""

    def test_auto_adds_validation_callback_when_eval_dataset_exists(
        self, minimal_sft_config
    ):
        """SFTTrainer should auto-add ValidationCallback when eval_dataset is available."""
        trainer = SFTTrainer(config=minimal_sft_config, callbacks=[])

        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__ = MagicMock(return_value=10)
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=90)

        # Mock the internal methods to avoid real model loading
        with (
            patch.object(trainer, "_load_model") as mock_load,
            patch.object(trainer, "_prepare_datasets") as mock_prepare,
            patch.object(trainer, "_load_template") as mock_template,
            patch.object(trainer, "_create_training_args") as mock_args,
            patch.object(trainer, "_create_trl_trainer") as mock_trl,
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            mock_prepare.return_value = (mock_train_dataset, mock_eval_dataset)
            mock_template.return_value = MagicMock()
            mock_args.return_value = MagicMock()

            mock_trl_instance = MagicMock()
            mock_trl_instance.train.return_value = MagicMock(
                training_loss=0.5, metrics={}
            )
            mock_trl.return_value = mock_trl_instance

            trainer.train()

        # ValidationCallback should have been auto-added
        validation_cbs = [
            cb for cb in trainer.callbacks if isinstance(cb, ValidationCallback)
        ]
        assert len(validation_cbs) == 1
        assert validation_cbs[0].eval_steps == 50
        assert validation_cbs[0].val_dataset is mock_eval_dataset

    def test_does_not_add_duplicate_validation_callback(self, minimal_sft_config):
        """SFTTrainer should NOT add ValidationCallback if one already exists."""
        existing_cb = ValidationCallback(val_dataset=MagicMock(), eval_steps=200)
        trainer = SFTTrainer(config=minimal_sft_config, callbacks=[existing_cb])

        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__ = MagicMock(return_value=10)
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=90)

        with (
            patch.object(trainer, "_load_model") as mock_load,
            patch.object(trainer, "_prepare_datasets") as mock_prepare,
            patch.object(trainer, "_load_template") as mock_template,
            patch.object(trainer, "_create_training_args") as mock_args,
            patch.object(trainer, "_create_trl_trainer") as mock_trl,
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            mock_prepare.return_value = (mock_train_dataset, mock_eval_dataset)
            mock_template.return_value = MagicMock()
            mock_args.return_value = MagicMock()

            mock_trl_instance = MagicMock()
            mock_trl_instance.train.return_value = MagicMock(
                training_loss=0.5, metrics={}
            )
            mock_trl.return_value = mock_trl_instance

            trainer.train()

        # Should still have only the original ValidationCallback
        validation_cbs = [
            cb for cb in trainer.callbacks if isinstance(cb, ValidationCallback)
        ]
        assert len(validation_cbs) == 1
        assert validation_cbs[0] is existing_cb
        assert validation_cbs[0].eval_steps == 200

    def test_no_validation_callback_when_no_eval_dataset(self, minimal_sft_config):
        """SFTTrainer should NOT add ValidationCallback when eval_dataset is None."""
        trainer = SFTTrainer(config=minimal_sft_config, callbacks=[])

        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=100)

        with (
            patch.object(trainer, "_load_model") as mock_load,
            patch.object(trainer, "_prepare_datasets") as mock_prepare,
            patch.object(trainer, "_load_template") as mock_template,
            patch.object(trainer, "_create_training_args") as mock_args,
            patch.object(trainer, "_create_trl_trainer") as mock_trl,
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            mock_prepare.return_value = (mock_train_dataset, None)
            mock_template.return_value = MagicMock()
            mock_args.return_value = MagicMock()

            mock_trl_instance = MagicMock()
            mock_trl_instance.train.return_value = MagicMock(
                training_loss=0.5, metrics={}
            )
            mock_trl.return_value = mock_trl_instance

            trainer.train()

        # No ValidationCallback should be present
        validation_cbs = [
            cb for cb in trainer.callbacks if isinstance(cb, ValidationCallback)
        ]
        assert len(validation_cbs) == 0

    def test_uses_default_eval_steps_when_not_configured(self, minimal_sft_config):
        """SFTTrainer should use 500 as default eval_steps when config.eval_steps is None."""
        minimal_sft_config.eval_steps = None
        trainer = SFTTrainer(config=minimal_sft_config, callbacks=[])

        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__ = MagicMock(return_value=10)
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=90)

        with (
            patch.object(trainer, "_load_model") as mock_load,
            patch.object(trainer, "_prepare_datasets") as mock_prepare,
            patch.object(trainer, "_load_template") as mock_template,
            patch.object(trainer, "_create_training_args") as mock_args,
            patch.object(trainer, "_create_trl_trainer") as mock_trl,
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            mock_prepare.return_value = (mock_train_dataset, mock_eval_dataset)
            mock_template.return_value = MagicMock()
            mock_args.return_value = MagicMock()

            mock_trl_instance = MagicMock()
            mock_trl_instance.train.return_value = MagicMock(
                training_loss=0.5, metrics={}
            )
            mock_trl.return_value = mock_trl_instance

            trainer.train()

        validation_cbs = [
            cb for cb in trainer.callbacks if isinstance(cb, ValidationCallback)
        ]
        assert len(validation_cbs) == 1
        assert validation_cbs[0].eval_steps == 500

    def test_preserves_other_callbacks(self, minimal_sft_config):
        """Auto-adding ValidationCallback should not disturb other callbacks."""

        class CustomCallback(SFTCallback):
            pass

        custom = CustomCallback()
        trainer = SFTTrainer(config=minimal_sft_config, callbacks=[custom])

        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__ = MagicMock(return_value=10)
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=90)

        with (
            patch.object(trainer, "_load_model") as mock_load,
            patch.object(trainer, "_prepare_datasets") as mock_prepare,
            patch.object(trainer, "_load_template") as mock_template,
            patch.object(trainer, "_create_training_args") as mock_args,
            patch.object(trainer, "_create_trl_trainer") as mock_trl,
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            mock_prepare.return_value = (mock_train_dataset, mock_eval_dataset)
            mock_template.return_value = MagicMock()
            mock_args.return_value = MagicMock()

            mock_trl_instance = MagicMock()
            mock_trl_instance.train.return_value = MagicMock(
                training_loss=0.5, metrics={}
            )
            mock_trl.return_value = mock_trl_instance

            trainer.train()

        # Both custom callback and auto-added validation should be present
        assert custom in trainer.callbacks
        assert len(trainer.callbacks) == 2
        assert any(isinstance(cb, ValidationCallback) for cb in trainer.callbacks)
