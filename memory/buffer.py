"""
memory/buffer.py

Fixed-size reservoir replay buffer.

Strategy: reservoir sampling (Vitter, 1985).
Each incoming sample has probability k/n of entering the buffer,
where k = capacity and n = total samples seen for that task.
This guarantees a statistically uniform sample of each task
regardless of how many samples arrive.

Usage:
    buffer = ReplayBuffer(capacity_per_task=200)
    buffer.add_task(task_id=0, dataset=train_tasks[0])
    replay_samples = buffer.sample(n_per_task=32)
    # replay_samples is a list of (x, y) tensors
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class ReplayBuffer:
    """
    Maintains one reservoir per task, each of fixed capacity.

    Args:
        capacity_per_task:  max samples stored per task
        seed:               random seed for reproducibility
    """

    def __init__(self, capacity_per_task: int = 200, seed: int = 42):
        self.capacity = capacity_per_task
        self.rng = random.Random(seed)

        # task_id -> list of (x: Tensor, y: int)
        self._storage: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
        self._counts:  Dict[int, int] = {}  # total samples seen per task

    def add_task(self, task_id: int, dataset: Dataset) -> None:
        """
        Populate the buffer for a new task using reservoir sampling.
        Call this once per task, right after the task's training finishes
        (or before — just before training gives you a clean uniform sample).

        Args:
            task_id:  integer identifier for this task
            dataset:  the full training dataset for this task
        """
        reservoir: List[Tuple[torch.Tensor, int]] = []
        n_seen = 0

        for i in range(len(dataset)):
            x, y = dataset[i]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            n_seen += 1

            if len(reservoir) < self.capacity:
                reservoir.append((x, y))
            else:
                # Replace a random existing element with probability k/n
                j = self.rng.randint(0, n_seen - 1)
                if j < self.capacity:
                    reservoir[j] = (x, y)

        self._storage[task_id] = reservoir
        self._counts[task_id]  = n_seen

        print(
            f"  Buffer: stored {len(reservoir)} samples for task {task_id} "
            f"(sampled from {n_seen} total)"
        )

    def sample(
        self,
        n_per_task: int,
        exclude_task: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Sample up to n_per_task items from each stored task.

        Args:
            n_per_task:    how many samples to draw per task
            exclude_task:  if set, skip this task (e.g. skip current task
                           to avoid duplicating samples already in the batch)

        Returns:
            Flat list of (x, y) tuples — shuffle before making a DataLoader.
        """
        samples = []
        for task_id, reservoir in self._storage.items():
            if task_id == exclude_task:
                continue
            n = min(n_per_task, len(reservoir))
            drawn = self.rng.sample(reservoir, n)
            samples.extend(drawn)
        return samples

    def collate(
        self,
        samples: List[Tuple[torch.Tensor, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of (x, y) tuples into a batch of tensors.
        Use this to turn buffer.sample() output into something
        you can feed directly to the model.

        Returns:
            x: (N, C, H, W) float tensor
            y: (N,) long tensor
        """
        xs = torch.stack([s[0] for s in samples])
        ys = torch.tensor([s[1] for s in samples], dtype=torch.long)
        return xs, ys

    @property
    def n_tasks_stored(self) -> int:
        return len(self._storage)

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self._storage.values())

    def __repr__(self) -> str:
        task_info = ", ".join(
            f"task {k}: {len(v)}" for k, v in self._storage.items()
        )
        return f"ReplayBuffer(capacity={self.capacity}, stored=[{task_info}])"


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    print("Testing ReplayBuffer...")
    fake_x = torch.randn(500, 3, 32, 32)
    fake_y = torch.zeros(500, dtype=torch.long)
    ds = TensorDataset(fake_x, fake_y)

    buf = ReplayBuffer(capacity_per_task=100)
    buf.add_task(0, ds)
    buf.add_task(1, ds)

    samples = buf.sample(n_per_task=32)
    x_batch, y_batch = buf.collate(samples)
    print(f"Sampled batch — x: {x_batch.shape}, y: {y_batch.shape}")
    print(buf)
    print("Buffer OK.")