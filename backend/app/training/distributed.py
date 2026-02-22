"""Distributed training utilities for DDP (DistributedDataParallel).

Provides setup/cleanup for PyTorch DDP, distributed sampler creation,
and utilities for coordinating across ranks.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Populated from environment variables set by torchrun/torch.distributed.launch.

    Attributes:
        world_size: Total number of processes (GPUs)
        rank: Global rank of this process
        local_rank: Local rank on this node (used for device assignment)
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
    """

    world_size: int
    rank: int
    local_rank: int
    backend: str = "nccl"

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0


def setup_distributed() -> Optional[DistributedConfig]:
    """Initialize distributed training from environment variables.

    Reads RANK, WORLD_SIZE, LOCAL_RANK set by torchrun. If these are not
    present, returns None (non-distributed mode).

    Returns:
        DistributedConfig if distributed env detected, None otherwise
    """
    # Check if torchrun set the environment
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")

    if rank is None or world_size is None or local_rank is None:
        return None

    try:
        rank = int(rank)
        world_size = int(world_size)
        local_rank = int(local_rank)
    except ValueError as e:
        raise RuntimeError(
            f"Invalid distributed training env vars: "
            f"RANK={os.environ.get('RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
        ) from e

    # Choose backend based on CUDA availability
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize process group
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except RuntimeError:
        logger.error(
            f"Failed to init process group: backend={backend}, "
            f"rank={rank}, world_size={world_size}"
        )
        raise

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=backend,
    )

    logger.info(
        f"Distributed training initialized: rank={rank}/{world_size}, "
        f"local_rank={local_rank}, backend={backend}"
    )

    return config


def cleanup_distributed() -> None:
    """Clean up distributed training resources.

    Should be called at the end of training to properly shut down
    the process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleanup complete")


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    dist_config: Optional[DistributedConfig] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader with optional distributed sampling.

    When dist_config is provided, wraps the dataset with a DistributedSampler
    to partition data across ranks. The sampler handles shuffling, so the
    DataLoader's shuffle parameter is set to False.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size per process (NOT total batch size)
        dist_config: Distributed config (None for single-process)
        shuffle: Whether to shuffle data (handled by sampler in distributed mode)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Configured DataLoader
    """
    sampler = None

    if dist_config is not None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_config.world_size,
            rank=dist_config.rank,
            shuffle=shuffle,
        )
        # When using a sampler, DataLoader shuffle must be False
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor by averaging across all processes.

    In non-distributed mode (dist not initialized), returns the tensor unchanged.
    Returns a new tensor to avoid in-place mutation that could break autograd.

    Args:
        tensor: Tensor to reduce

    Returns:
        Averaged tensor across all processes
    """
    if not dist.is_initialized():
        return tensor

    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / dist.get_world_size()
