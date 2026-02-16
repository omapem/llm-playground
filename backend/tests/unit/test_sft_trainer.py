"""Unit tests for SFT trainer.

Tests the main SFTTrainer orchestrator that coordinates model loading,
dataset preparation, and training execution using TRL's SFTTrainer.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

from app.sft.config import SFTConfig
from app.sft.templates import AlpacaTemplate, ChatTemplate, TemplateRegistry
from app.sft.dataset import SFTDatasetProcessor
from app.sft.lora_config import LoRAConfigManager
from app.sft.callbacks import SFTCallback, ValidationCallback
from app.sft.trainer import SFTTrainer


@pytest.fixture
def mock_config():
    """Create a minimal SFT config for testing."""
    return SFTConfig(
        base_model="gpt2",
        dataset_name="tatsu-lab/alpaca",
        dataset_format="alpaca",
        output_dir="./test_output",
        batch_size=2,
        num_epochs=1,
        max_seq_length=512,
    )


@pytest.fixture
def mock_config_with_template():
    """Create config with explicit template name."""
    return SFTConfig(
        base_model="gpt2",
        dataset_name="tatsu-lab/alpaca",
        dataset_format="alpaca",
        template_name="alpaca",
        output_dir="./test_output",
    )


@pytest.fixture
def mock_config_chat():
    """Create config for chat format."""
    return SFTConfig(
        base_model="gpt2",
        dataset_name="openchat/openchat",
        dataset_format="chat",
        output_dir="./test_output",
    )


@pytest.fixture
def mock_callbacks():
    """Create mock callbacks."""
    return [Mock(spec=SFTCallback), Mock(spec=ValidationCallback)]


class TestSFTTrainerInit:
    """Tests for SFTTrainer initialization."""

    def test_init_basic(self, mock_config):
        """Test basic trainer initialization."""
        trainer = SFTTrainer(mock_config)
        assert trainer.config == mock_config
        assert trainer.callbacks == []
        assert isinstance(trainer.dataset_processor, SFTDatasetProcessor)

    def test_init_with_callbacks(self, mock_config, mock_callbacks):
        """Test initialization with callbacks."""
        trainer = SFTTrainer(mock_config, callbacks=mock_callbacks)
        assert trainer.config == mock_config
        assert trainer.callbacks == mock_callbacks

    def test_init_empty_callbacks_list(self, mock_config):
        """Test initialization with empty callbacks list."""
        trainer = SFTTrainer(mock_config, callbacks=[])
        assert trainer.callbacks == []


class TestTemplateLoading:
    """Tests for template loading functionality."""

    def test_load_template_explicit_name(self, mock_config_with_template):
        """Test loading template with explicit name."""
        trainer = SFTTrainer(mock_config_with_template)
        template = trainer._load_template()
        assert isinstance(template, AlpacaTemplate)

    def test_load_template_auto_detect_alpaca(self, mock_config):
        """Test auto-detection of Alpaca template from dataset format."""
        trainer = SFTTrainer(mock_config)
        template = trainer._load_template()
        assert isinstance(template, AlpacaTemplate)

    def test_load_template_auto_detect_chat(self, mock_config_chat):
        """Test auto-detection of ChatML template from dataset format."""
        trainer = SFTTrainer(mock_config_chat)
        template = trainer._load_template()
        assert isinstance(template, ChatTemplate)
        assert template.chat_format == "chatml"

    def test_load_template_invalid_name(self, mock_config):
        """Test error when template name is invalid."""
        mock_config.template_name = "nonexistent_template"
        trainer = SFTTrainer(mock_config)
        with pytest.raises(KeyError, match="Template.*not found"):
            trainer._load_template()

    def test_load_template_invalid_format(self, mock_config):
        """Test error when dataset format is unknown."""
        mock_config.dataset_format = "unknown_format"
        mock_config.template_name = None
        trainer = SFTTrainer(mock_config)
        with pytest.raises(ValueError, match="Unknown dataset format"):
            trainer._load_template()


class TestModelLoading:
    """Tests for model loading functionality."""

    @patch("app.sft.trainer.AutoTokenizer")
    @patch("app.sft.trainer.AutoModelForCausalLM")
    @patch("app.sft.trainer.get_peft_model")
    def test_load_model_basic(
        self, mock_get_peft, mock_auto_model, mock_auto_tokenizer, mock_config
    ):
        """Test basic model loading with LoRA."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model
        mock_peft_model = Mock()
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_config)
        model, tokenizer = trainer._load_model()

        # Verify calls
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_auto_model.from_pretrained.assert_called_once()
        mock_get_peft.assert_called_once()

        # Verify returns
        assert model == mock_peft_model
        assert tokenizer == mock_tokenizer

    @patch("app.sft.trainer.AutoTokenizer")
    @patch("app.sft.trainer.AutoModelForCausalLM")
    @patch("app.sft.trainer.get_peft_model")
    def test_load_model_with_qlora(
        self, mock_get_peft, mock_auto_model, mock_auto_tokenizer, mock_config
    ):
        """Test model loading with QLoRA quantization."""
        mock_config.use_qlora = True

        # Setup mocks
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model
        mock_peft_model = Mock()
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_config)
        model, tokenizer = trainer._load_model()

        # Verify quantization config was passed
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs
        assert call_kwargs["quantization_config"] is not None

    @patch("app.sft.trainer.AutoTokenizer")
    @patch("app.sft.trainer.AutoModelForCausalLM")
    @patch("app.sft.trainer.get_peft_model")
    def test_load_model_without_lora(
        self, mock_get_peft, mock_auto_model, mock_auto_tokenizer, mock_config
    ):
        """Test model loading when LoRA is disabled."""
        mock_config.use_lora = False

        # Setup mocks
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        trainer = SFTTrainer(mock_config)
        model, tokenizer = trainer._load_model()

        # When LoRA is disabled, PEFT should not be applied
        mock_get_peft.assert_not_called()
        assert model == mock_base_model

    @patch("app.sft.trainer.get_peft_model")
    @patch("app.sft.trainer.AutoTokenizer")
    @patch("app.sft.trainer.AutoModelForCausalLM")
    def test_load_model_device_map(
        self, mock_auto_model, mock_auto_tokenizer, mock_get_peft, mock_config
    ):
        """Test that device_map is passed correctly."""
        mock_config.device_map = "auto"

        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = Mock()
        mock_auto_model.from_pretrained.return_value = Mock()
        mock_get_peft.return_value = Mock()

        trainer = SFTTrainer(mock_config)
        trainer._load_model()

        # Verify device_map was passed
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["device_map"] == "auto"


class TestDatasetPreparation:
    """Tests for dataset preparation functionality."""

    @patch.object(SFTDatasetProcessor, "load_and_format")
    def test_prepare_datasets_single_dataset(self, mock_load_and_format, mock_config):
        """Test dataset preparation without validation split."""
        mock_dataset = Mock()
        mock_load_and_format.return_value = mock_dataset

        trainer = SFTTrainer(mock_config)
        train_ds, val_ds = trainer._prepare_datasets()

        # Verify processor was called
        mock_load_and_format.assert_called_once()
        call_kwargs = mock_load_and_format.call_args[1]
        assert call_kwargs["dataset_name"] == mock_config.dataset_name
        # Check template is correct type (instance comparison won't work)
        assert isinstance(call_kwargs["template"], AlpacaTemplate)

        # Should return single dataset as train, None for val
        assert train_ds == mock_dataset
        assert val_ds is None

    @patch.object(SFTDatasetProcessor, "load_and_format")
    def test_prepare_datasets_with_validation_split(
        self, mock_load_and_format, mock_config
    ):
        """Test dataset preparation with validation split."""
        mock_train = Mock()
        mock_val = Mock()
        mock_load_and_format.return_value = (mock_train, mock_val)

        mock_config.validation_split = 0.1

        trainer = SFTTrainer(mock_config)
        train_ds, val_ds = trainer._prepare_datasets()

        # Should return both datasets
        assert train_ds == mock_train
        assert val_ds == mock_val


class TestTrainingArgsCreation:
    """Tests for HuggingFace TrainingArguments creation."""

    def test_create_training_args_basic(self, mock_config):
        """Test creation of TrainingArguments with all config fields."""
        trainer = SFTTrainer(mock_config)
        training_args = trainer._create_training_args()

        # Verify key fields are mapped
        assert training_args.output_dir == mock_config.output_dir
        assert training_args.num_train_epochs == mock_config.num_epochs
        assert (
            training_args.per_device_train_batch_size == mock_config.batch_size
        )
        assert (
            training_args.gradient_accumulation_steps
            == mock_config.gradient_accumulation_steps
        )
        assert training_args.learning_rate == mock_config.learning_rate
        assert training_args.warmup_ratio == mock_config.warmup_ratio

    def test_create_training_args_mixed_precision_fp16(self, mock_config):
        """Test training args with fp16 mixed precision."""
        mock_config.mixed_precision = "fp16"
        trainer = SFTTrainer(mock_config)
        training_args = trainer._create_training_args()

        assert training_args.fp16 is True
        assert training_args.bf16 is False

    def test_create_training_args_mixed_precision_bf16(self, mock_config):
        """Test training args with bf16 mixed precision."""
        mock_config.mixed_precision = "bf16"
        trainer = SFTTrainer(mock_config)
        training_args = trainer._create_training_args()

        assert training_args.bf16 is True
        assert training_args.fp16 is False

    def test_create_training_args_optimizer(self, mock_config):
        """Test optimizer type is passed correctly."""
        mock_config.optimizer_type = "paged_adamw_32bit"
        trainer = SFTTrainer(mock_config)
        training_args = trainer._create_training_args()

        assert training_args.optim == "paged_adamw_32bit"

    def test_create_training_args_scheduler(self, mock_config):
        """Test scheduler type is passed correctly."""
        mock_config.scheduler_type = "cosine"
        trainer = SFTTrainer(mock_config)
        training_args = trainer._create_training_args()

        assert training_args.lr_scheduler_type == "cosine"


class TestTRLTrainerCreation:
    """Tests for TRL SFTTrainer instantiation."""

    @patch("app.sft.trainer.TRLSFTTrainer")
    def test_create_trl_trainer(self, mock_trl_trainer, mock_config):
        """Test TRL trainer instantiation."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_training_args = Mock()

        trainer = SFTTrainer(mock_config)
        trl_trainer = trainer._create_trl_trainer(
            mock_model,
            mock_tokenizer,
            mock_train_dataset,
            mock_eval_dataset,
            mock_training_args,
        )

        # Verify TRL trainer was created with correct arguments
        mock_trl_trainer.assert_called_once_with(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            args=mock_training_args,
            max_seq_length=mock_config.max_seq_length,
            dataset_text_field="text",
        )


class TestMergedModelSaving:
    """Tests for merged model saving functionality."""

    @patch("app.sft.trainer.Path")
    def test_save_merged_model(self, mock_path, mock_config):
        """Test saving merged model."""
        # Setup mock model with PEFT methods
        mock_model = Mock()
        mock_model.merge_and_unload.return_value = Mock()

        # Setup mock path
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir = Mock()

        trainer = SFTTrainer(mock_config)
        output_dir = "./merged_model"
        trainer._save_merged_model(mock_model, output_dir)

        # Verify merge was called
        mock_model.merge_and_unload.assert_called_once()


class TestEndToEndTraining:
    """Tests for complete training workflow."""

    @patch("app.sft.trainer.TRLSFTTrainer")
    @patch.object(SFTTrainer, "_save_merged_model")
    @patch.object(SFTTrainer, "_prepare_datasets")
    @patch.object(SFTTrainer, "_load_model")
    def test_train_complete_workflow(
        self,
        mock_load_model,
        mock_prepare_datasets,
        mock_save_merged,
        mock_trl_trainer,
        mock_config,
    ):
        """Test complete training workflow."""
        # Setup mocks
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_train_ds = Mock()
        mock_train_ds.__len__ = Mock(return_value=100)
        mock_val_ds = Mock()
        mock_val_ds.__len__ = Mock(return_value=10)
        mock_prepare_datasets.return_value = (mock_train_ds, mock_val_ds)

        mock_train_result = Mock()
        mock_train_result.training_loss = 1.5
        mock_train_result.metrics = {"train_loss": 1.5}

        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = mock_train_result
        mock_trainer_instance.add_callback = Mock()
        mock_trl_trainer.return_value = mock_trainer_instance

        # Execute training
        trainer = SFTTrainer(mock_config)
        result = trainer.train()

        # Verify workflow
        mock_load_model.assert_called_once()
        mock_prepare_datasets.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

        # Verify result
        assert "train_loss" in result
        assert "metrics" in result
        assert "output_dir" in result

    @patch("app.sft.trainer.TRLSFTTrainer")
    @patch.object(SFTTrainer, "_prepare_datasets")
    @patch.object(SFTTrainer, "_load_model")
    def test_train_with_callbacks(
        self, mock_load_model, mock_prepare_datasets, mock_trl_trainer, mock_config
    ):
        """Test training with custom callbacks."""
        # Setup mocks
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_train_ds = Mock()
        mock_train_ds.__len__ = Mock(return_value=100)
        mock_prepare_datasets.return_value = (mock_train_ds, None)

        mock_train_result = Mock()
        mock_train_result.training_loss = 1.5
        mock_train_result.metrics = {}

        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = mock_train_result
        mock_trainer_instance.add_callback = Mock()
        mock_trl_trainer.return_value = mock_trainer_instance

        # Create trainer with callbacks
        mock_callback = Mock(spec=SFTCallback)
        trainer = SFTTrainer(mock_config, callbacks=[mock_callback])
        result = trainer.train()

        # Verify callback was added
        mock_trainer_instance.add_callback.assert_called_once_with(mock_callback)

    @patch("app.sft.trainer.TRLSFTTrainer")
    @patch.object(SFTTrainer, "_save_merged_model")
    @patch.object(SFTTrainer, "_prepare_datasets")
    @patch.object(SFTTrainer, "_load_model")
    def test_train_with_merged_model_saving(
        self,
        mock_load_model,
        mock_prepare_datasets,
        mock_save_merged,
        mock_trl_trainer,
        mock_config,
    ):
        """Test training with merged model saving enabled."""
        mock_config.save_merged_model = True

        # Setup mocks
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_train_ds = Mock()
        mock_train_ds.__len__ = Mock(return_value=100)
        mock_prepare_datasets.return_value = (mock_train_ds, None)

        mock_train_result = Mock()
        mock_train_result.training_loss = 1.5
        mock_train_result.metrics = {}

        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = mock_train_result
        mock_trl_trainer.return_value = mock_trainer_instance

        # Execute training
        trainer = SFTTrainer(mock_config)
        trainer.train()

        # Verify merged model was saved
        mock_save_merged.assert_called_once()
