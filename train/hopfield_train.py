import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Optional

from data.cifar10_tasks import get_task_loader, TaskSubset, TASK_CLASSES
from models.resnet import CIFARResNet18
from eval.metrics import MetricsTracker


def remap_labels(y: torch.Tensor, task_id: int) -> torch.Tensor:
    """
    Remaps original CIFAR-10 labels (0-9) to binary {0, 1} for the task head.
    e.g. Task 0 has classes (0,1): label 0 -> 0, label 1 -> 1
         Task 1 has classes (2,3): label 2 -> 0, label 3 -> 1
    """
    c0, c1 = TASK_CLASSES[task_id]
    out = torch.zeros_like(y)
    out[y == c1] = 1
    return out


@torch.no_grad()
def evaluate(
    model: CIFARResNet18,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_id: int,
) -> float:
    """Evaluates accuracy for a specific task using that task's head."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y = remap_labels(y, task_id)
        logits = model(x, task_id=task_id)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_one_epoch_hopfield(
    model: CIFARResNet18,
    current_loader: torch.utils.data.DataLoader,
    hopfield,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task_id: int,
    lambda_mem: float = 0.5,
    sim_threshold: float = 0.3,
) -> float:
    """
    One training epoch with Hopfield memory regularization.

    Loss = classification loss + lambda_mem * memory distillation loss

    The memory loss pulls the current embeddings towards the retrieved
    Hopfield memories for samples that are sufficiently similar (above
    sim_threshold). This prevents the backbone from drifting away from
    representations that encode old tasks.

    lambda_mem: weight of memory loss relative to classification loss.
                0.5 is a good starting point — increase if forgetting persists,
                decrease if new task learning is too slow.
    sim_threshold: only apply memory loss to samples whose current embedding
                   is similar enough to a retrieved memory. Filters out
                   samples that are genuinely new and shouldn't be constrained.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_cur, y_cur in current_loader:
        x_cur = x_cur.to(device)
        y_cur = y_cur.to(device)
        y_cur = remap_labels(y_cur, task_id)

        # Forward pass — get features and logits
        h_cur = model.get_features(x_cur)
        logits = model.heads[str(task_id)](h_cur)
        loss_cls = criterion(logits, y_cur)

        # Memory regularization — only if Hopfield has stored memories
        h_ret = hopfield.retrieve(h_cur)
        if h_ret is not None:
            similarity = F.cosine_similarity(h_cur, h_ret, dim=1)
            mask = similarity > sim_threshold
            if mask.sum() > 0:
                # MSE between current features and retrieved memories
                # for samples that are close to something we've seen before
                loss_mem = torch.mean((h_cur[mask] - h_ret[mask].detach()) ** 2)
                loss = loss_cls + lambda_mem * loss_mem
            else:
                loss = loss_cls
        else:
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_hopfield(
    model: CIFARResNet18,
    train_tasks: List[TaskSubset],
    test_tasks: List[TaskSubset],
    hopfield,
    capacity_per_task: int = 200,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.05,
    lambda_mem: float = 0.5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> MetricsTracker:
    """
    Full continual learning loop with Hopfield episodic memory.

    Per task:
      1. Add a fresh per-task output head
      2. Train on current task data with Hopfield memory regularization
      3. Extract embeddings from trained model and store in Hopfield memory
      4. Evaluate on all seen tasks using their respective heads
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    n_tasks = len(train_tasks)
    tracker = MetricsTracker(n_tasks=n_tasks)

    if verbose:
        print(f"\nHopfield training — {n_tasks} tasks, {n_epochs} epochs each")
        print(f"lambda_mem={lambda_mem} | capacity={capacity_per_task}")
        print(f"Device: {device}")

    for task_id in range(n_tasks):
        if verbose:
            print(f"\n--- Task {task_id} ({TASK_CLASSES[task_id]}) ---")

        # Add fresh head for this task
        model.add_head(task_id, num_classes=2)

        current_loader = get_task_loader(
            train_tasks[task_id], batch_size=batch_size, shuffle=True
        )

        optimizer = SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            loss = train_one_epoch_hopfield(
                model=model,
                current_loader=current_loader,
                hopfield=hopfield,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task_id=task_id,
                lambda_mem=lambda_mem,
            )
            scheduler.step()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — loss: {loss:.4f}")

        # Store embeddings from current task into Hopfield memory
        if verbose:
            print(f"  Storing embeddings in Hopfield memory...")
        model.eval()
        embeddings = []
        store_loader = get_task_loader(
            train_tasks[task_id], batch_size=128, shuffle=False
        )
        with torch.no_grad():
            for x, _ in store_loader:
                h = model.get_features(x.to(device))
                embeddings.append(h)

        # Reservoir sample down to capacity before storing
        all_emb = torch.cat(embeddings, dim=0)
        if all_emb.size(0) > capacity_per_task:
            idx = torch.randperm(all_emb.size(0))[:capacity_per_task]
            all_emb = all_emb[idx]

        hopfield.store(all_emb)
        if verbose:
            print(f"  Hopfield memory size: {hopfield.memory.size(0)} embeddings")

        # Evaluate on all seen tasks
        if verbose:
            print(f"  Evaluating on tasks 0..{task_id}:")
        for eval_task_id in range(task_id + 1):
            test_loader = get_task_loader(
                test_tasks[eval_task_id], batch_size=128, shuffle=False
            )
            acc = evaluate(model, test_loader, device, task_id=eval_task_id)
            tracker.record(task_id, eval_task_id, acc)
            if verbose:
                print(f"    Task {eval_task_id} accuracy: {acc*100:.1f}%")

    return tracker


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.cifar10_tasks import get_task_datasets
    from models.resnet import build_model

    # Import your HopfieldMemory class
    sys.path.insert(0, ".")
    from memory.hopfield import HopfieldMemory

    print("Running Hopfield training (sanity mode)...")
    train_tasks, test_tasks = get_task_datasets(n_tasks=2, sanity_samples=300)
    model = build_model()
    hopfield = HopfieldMemory()

    tracker = run_hopfield(
        model=model,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        hopfield=hopfield,
        n_epochs=3,
        batch_size=32,
        capacity_per_task=100,
        lambda_mem=0.5,
    )
    tracker.print_summary("hopfield")