"""
train/naive.py

Naive sequential training — no replay, no memory.
Trains the model on each task in order, then evaluates on all seen tasks.
This is your forgetting floor — how bad things get without any intervention.

After Task 2 you should see Task 1 accuracy collapse to near-random (50%
for binary, ~10% for 10-class). That's the catastrophic forgetting signal
this whole project is built around.
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Optional

from data.cifar10_tasks  import get_task_loader, TaskSubset
from models.resnet       import CIFARResNet18
from eval.metrics        import MetricsTracker


def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """Runs one epoch, returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   torch.utils.data.DataLoader,
    device:   torch.device,
) -> float:
    """Returns accuracy in [0, 1] on the given loader."""
    model.eval()
    correct = 0
    total   = 0

    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        logits  = model(x)
        preds   = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

    return correct / max(total, 1)


def run_naive(
    model:       CIFARResNet18,
    train_tasks: List[TaskSubset],
    test_tasks:  List[TaskSubset],
    n_epochs:    int   = 10,
    batch_size:  int   = 64,
    lr:          float = 0.05,
    device:      Optional[torch.device] = None,
    verbose:     bool  = True,
) -> MetricsTracker:
    """
    Trains sequentially on each task with no replay.

    Returns a MetricsTracker populated with the full accuracy matrix.

    Args:
        model:       freshly initialised CIFARResNet18
        train_tasks: list of training datasets, one per task
        test_tasks:  list of test datasets, one per task
        n_epochs:    epochs per task
        batch_size:  training batch size
        lr:          initial SGD learning rate
        device:      torch.device (autodetects if None)
        verbose:     print progress
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    n_tasks   = len(train_tasks)
    tracker   = MetricsTracker(n_tasks=n_tasks)

    if verbose:
        print(f"\nNaive baseline — {n_tasks} tasks, {n_epochs} epochs each")
        print(f"Device: {device}")

    for task_id in range(n_tasks):
        if verbose:
            print(f"\n--- Task {task_id} ---")

        train_loader = get_task_loader(
            train_tasks[task_id], batch_size=batch_size, shuffle=True
        )

        # Fresh optimiser + scheduler per task
        # SGD + cosine LR is standard for CIFAR — Adam tends to overfit here
        optimizer = SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — loss: {loss:.4f}")

        # Evaluate on all tasks seen so far
        if verbose:
            print(f"  Evaluating on tasks 0..{task_id}:")
        for eval_task_id in range(task_id + 1):
            test_loader = get_task_loader(
                test_tasks[eval_task_id], batch_size=128, shuffle=False
            )
            acc = evaluate(model, test_loader, device)
            tracker.record(
                after_task=task_id,
                task_tested=eval_task_id,
                accuracy=acc,
            )
            if verbose:
                print(f"    Task {eval_task_id} accuracy: {acc*100:.1f}%")

    return tracker


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.cifar10_tasks import get_task_datasets
    from models.resnet import build_model

    print("Running naive baseline (sanity mode — 3 tasks, 2 epochs, 300 samples)...")
    train_tasks, test_tasks = get_task_datasets(
        n_tasks=3, sanity_samples=300
    )
    model   = build_model()
    tracker = run_naive(
        model, train_tasks, test_tasks,
        n_epochs=2, batch_size=32,
    )
    tracker.print_summary("naive")