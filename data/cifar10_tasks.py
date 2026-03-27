"""
data/cifar10_tasks.py

Splits CIFAR-10 into 5 sequential binary classification tasks:
  Task 0: classes {0, 1}  — airplane vs automobile
  Task 1: classes {2, 3}  — bird vs cat
  Task 2: classes {4, 5}  — deer vs dog
  Task 3: classes {6, 7}  — frog vs horse
  Task 4: classes {8, 9}  — ship vs truck

Labels are remapped to {0, 1} within each task so the head
always has 2 outputs. The task_id tells the trainer which
head to use (or you can use a single 10-class head — see NOTE).

NOTE on output head strategy:
  - Single 10-class head (default here): simpler, avoids routing logic.
    Less realistic but fine for benchmarking forgetting.
  - Per-task 2-class heads: more realistic, requires task ID at inference.
  We use the 10-class head so task ID is NOT needed at test time.
  Labels are kept as original CIFAR-10 labels (0-9) for this reason.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from typing import List, Tuple


TASK_CLASSES: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])


class TaskSubset(Dataset):
    """
    Wraps a CIFAR-10 dataset and exposes only the samples
    belonging to a given pair of classes.

    Labels are kept as original CIFAR-10 labels (0-9).
    """

    def __init__(self, cifar_dataset, class_pair: Tuple[int, int]):
        self.dataset = cifar_dataset
        self.class_pair = class_pair

        self.indices = [
            i for i, (_, label) in enumerate(cifar_dataset)
            if label in class_pair
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def get_task_datasets(
    data_root: str = "./data/cifar10",
    n_tasks: int = 5,
    sanity_samples: int = None,
) -> Tuple[List[TaskSubset], List[TaskSubset]]:
    """
    Returns (train_tasks, test_tasks) — two lists of length n_tasks.
    Each element is a TaskSubset for that task.

    Args:
        data_root:        where to download/cache CIFAR-10
        n_tasks:          how many tasks to use (1-5)
        sanity_samples:   if set, truncate each task to this many
                          samples for fast local testing
    """
    assert 1 <= n_tasks <= 5, "n_tasks must be between 1 and 5"

    train_base = datasets.CIFAR10(
        root=data_root, train=True, download=True,
        transform=get_transforms(train=True)
    )
    test_base = datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=get_transforms(train=False)
    )

    train_tasks, test_tasks = [], []

    for task_id in range(n_tasks):
        class_pair = TASK_CLASSES[task_id]

        train_task = TaskSubset(train_base, class_pair)
        test_task  = TaskSubset(test_base,  class_pair)

        if sanity_samples is not None:
            train_task = Subset(train_task, range(min(sanity_samples, len(train_task))))
            test_task  = Subset(test_task,  range(min(sanity_samples // 5, len(test_task))))

        train_tasks.append(train_task)
        test_tasks.append(test_task)

    return train_tasks, test_tasks


def get_task_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience wrapper — returns a DataLoader for a single task dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


if __name__ == "__main__":
    print("Verifying Split CIFAR-10 data loader...")
    train_tasks, test_tasks = get_task_datasets(n_tasks=5, sanity_samples=200)

    for i, (tr, te) in enumerate(zip(train_tasks, test_tasks)):
        c0, c1 = TASK_CLASSES[i]
        print(
            f"  Task {i} ({CIFAR10_CLASSES[c0]} vs {CIFAR10_CLASSES[c1]}): "
            f"{len(tr)} train, {len(te)} test samples"
        )

    loader = get_task_loader(train_tasks[0], batch_size=32)
    x, y = next(iter(loader))
    print(f"\nSample batch — x: {x.shape}, y: {y.shape}, labels: {y.unique().tolist()}")
    print("Data loader OK.")