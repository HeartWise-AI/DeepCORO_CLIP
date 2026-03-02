import math
import random
from typing import Iterator, List, Optional

import torch.distributed as dist
from torch.utils.data import Sampler


class ClassAwareDistributedBatchSampler(Sampler[int]):
    """
    Distributed sampler that enforces a target abnormal ratio within every batch.

    The sampler precomputes balanced batches for the entire epoch, then shards
    them across distributed ranks. Sampling is done with replacement to avoid
    depletion when one class is underrepresented.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        abnormal_ratio: float = 0.5,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        label_source = getattr(dataset, "labels_indexed", None)
        if not label_source:
            label_source = getattr(dataset, "labels", None)
        if not label_source:
            raise ValueError("Dataset must expose non-empty 'labels' for class-aware sampling.")

        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive for ClassAwareDistributedBatchSampler.")

        self.abnormal_ratio = max(0.0, min(1.0, float(abnormal_ratio)))
        self.drop_last = drop_last
        self.world_size = max(1, int(world_size))
        self.rank = max(0, int(rank))
        self.seed = seed
        self.epoch = 0

        labels = [str(label).lower() for label in label_source]
        self.abnormal_indices: List[int] = [idx for idx, label in enumerate(labels) if label == "abnormal"]
        self.normal_indices: List[int] = [idx for idx, label in enumerate(labels) if label == "normal"]

        if not self.abnormal_indices or not self.normal_indices:
            raise ValueError("Class-aware sampler requires both abnormal and normal samples in the dataset.")

        self.abn_per_batch = max(1, int(round(self.batch_size * self.abnormal_ratio)))
        self.norm_per_batch = max(0, self.batch_size - self.abn_per_batch)

        if self.norm_per_batch == 0 and self.abn_per_batch < self.batch_size:
            self.abn_per_batch = self.batch_size

        self._rank_indices: List[int] = []
        self._epoch_batches: List[List[int]] = []
        self._needs_rebuild = True

    def __iter__(self) -> Iterator[int]:
        if self._needs_rebuild:
            self._build_epoch_indices()
            self._needs_rebuild = False
        return iter(self._rank_indices)

    def __len__(self) -> int:
        if self._needs_rebuild:
            self._build_epoch_indices()
            self._needs_rebuild = False
        return len(self._rank_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._needs_rebuild = True

    def _expand_pool(self, indices: List[int], required: int, rng: random.Random) -> List[int]:
        if required <= 0 or not indices:
            return []
        repeats = math.ceil(required / len(indices))
        pool = indices * repeats
        rng.shuffle(pool)
        return pool[:required]

    def _build_epoch_indices(self) -> None:
        rng = random.Random(self.seed + self.epoch)

        dataset_len = len(self.dataset)
        if dataset_len == 0:
            self._rank_indices = []
            self._epoch_batches = []
            return

        num_batches = math.ceil(dataset_len / self.batch_size)
        abn_needed = num_batches * self.abn_per_batch
        norm_needed = num_batches * self.norm_per_batch

        abn_pool = self._expand_pool(self.abnormal_indices, abn_needed, rng)
        norm_pool = self._expand_pool(self.normal_indices, norm_needed, rng)

        batches: List[List[int]] = []
        for batch_idx in range(num_batches):
            start_abn = batch_idx * self.abn_per_batch
            start_norm = batch_idx * self.norm_per_batch

            batch: List[int] = []
            batch.extend(abn_pool[start_abn:start_abn + self.abn_per_batch])
            batch.extend(norm_pool[start_norm:start_norm + self.norm_per_batch])

            if len(batch) < self.batch_size:
                combined = self.abnormal_indices + self.normal_indices
                while len(batch) < self.batch_size and combined:
                    batch.append(rng.choice(combined))

            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch[: self.batch_size])

        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]

        self._epoch_batches = batches
        rank_batches = batches[self.rank::self.world_size]
        self._rank_indices = [index for batch in rank_batches for index in batch]
