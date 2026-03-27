import torch
import torch.nn as nn
import random
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Optional

from data.cifar10_tasks import get_task_loader, TaskSubset
from models.resnet import CIFARResNet18
from memory.embedding_buffer import EmbeddingBuffer
from eval.metrics import MetricsTracker
from train.naive import evaluate


def train_one_epoch_embedding_replay(
    model,
    current_loader,
    buffer,
    optimizer,
    criterion,
    device,
    batch_size,
):
    model.train()
    total_loss = 0

    replay_size = batch_size // 2
    current_size = batch_size - replay_size

    for x_cur, y_cur in current_loader:

        x_cur = x_cur[:current_size].to(device)
        y_cur = y_cur[:current_size].to(device)

        # ---- CURRENT EMBEDDINGS ----
        h_cur = model.extract_features(x_cur)

        # ---- REPLAY EMBEDDINGS ----
        if buffer.n_tasks_stored > 0:
            all_samples = buffer.get_all()

            if len(all_samples) > 0:
                samples = random.sample(
                    all_samples,
                    min(replay_size, len(all_samples))
                )

                h_rep, y_rep = buffer.collate(samples)
                h_rep = h_rep.to(device)
                y_rep = y_rep.to(device)

                # combine embeddings
                h = torch.cat([h_cur, h_rep], dim=0)
                y = torch.cat([y_cur, y_rep], dim=0)
            else:
                h, y = h_cur, y_cur
        else:
            h, y = h_cur, y_cur

        # ---- CLASSIFICATION ----
        logits = model.fc(h)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(current_loader)


def run_embedding_replay(
    model: CIFARResNet18,
    train_tasks: List[TaskSubset],
    test_tasks: List[TaskSubset],
    capacity_per_task: int = 200,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> MetricsTracker:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    tracker = MetricsTracker(n_tasks=len(train_tasks))
    buffer = EmbeddingBuffer(capacity_per_task=capacity_per_task)

    if verbose:
        print(f"\nEmbedding Replay — {len(train_tasks)} tasks, {n_epochs} epochs each")
        print(f"Buffer: {capacity_per_task} embeddings/task")
        print(f"Device: {device}")

    for task_id in range(len(train_tasks)):

        if verbose:
            print(f"\n--- Task {task_id} ---")

        current_loader = get_task_loader(
            train_tasks[task_id], batch_size=batch_size, shuffle=True
        )

        optimizer = SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            loss = train_one_epoch_embedding_replay(
                model,
                current_loader,
                buffer,
                optimizer,
                criterion,
                device,
                batch_size,
            )
            scheduler.step()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — loss: {loss:.4f}")

        # 🔥 store embeddings AFTER training
        buffer.add_task(task_id, model, train_tasks[task_id], device)

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