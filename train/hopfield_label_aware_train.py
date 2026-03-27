import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Optional

from data.cifar10_tasks import get_task_loader, TaskSubset, TASK_CLASSES
from models.resnet import CIFARResNet18
from eval.metrics import MetricsTracker
from train.hopfield_train import evaluate, remap_labels
from memory.hopfield_label_aware import LabelAwareHopfieldMemory


def train_one_epoch_label_aware(
    model, current_loader, hopfield, optimizer, criterion,
    device, task_id, lambda_mem=0.1, sim_threshold=0.2,
    alpha=1.0, beta=0.5, top_k=16, temperature=0.3,
):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x_cur, y_cur in current_loader:
        x_cur = x_cur.to(device)
        y_cur = remap_labels(y_cur.to(device), task_id)

        h_cur    = model.get_features(x_cur)
        logits   = model.heads[str(task_id)](h_cur)
        loss_cls = criterion(logits, y_cur)

        h_ret = hopfield.retrieve(
            query=h_cur.detach(),
            current_task_id=task_id,
            top_k=top_k,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
        )

        if h_ret is not None:
            similarity = F.cosine_similarity(h_cur, h_ret, dim=1)
            mask = similarity > sim_threshold
            if mask.sum() > 0:
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
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def run_label_aware_hopfield(
    model, train_tasks, test_tasks, hopfield,
    capacity_per_task=500, n_epochs=10, batch_size=64,
    lr=0.05, lambda_mem=0.1, alpha=1.0, beta=0.5,
    top_k=16, temperature=0.3, sim_threshold=0.2,
    device=None, verbose=True,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    n_tasks   = len(train_tasks)
    tracker   = MetricsTracker(n_tasks=n_tasks)

    if verbose:
        print(f"\nLabel-Aware Hopfield — {n_tasks} tasks, {n_epochs} epochs each")
        print(f"λ={lambda_mem} | α={alpha} | β={beta} | cap={capacity_per_task}")
        print(f"top_k={top_k} | temp={temperature} | sim_threshold={sim_threshold}")
        print(f"Device: {device}")

    for task_id in range(n_tasks):
        if verbose:
            print(f"\n--- Task {task_id} ({TASK_CLASSES[task_id]}) ---")

        model.add_head(task_id, num_classes=2)

        current_loader = get_task_loader(
            train_tasks[task_id], batch_size=batch_size, shuffle=True
        )
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            loss = train_one_epoch_label_aware(
                model=model, current_loader=current_loader,
                hopfield=hopfield, optimizer=optimizer,
                criterion=criterion, device=device,
                task_id=task_id, lambda_mem=lambda_mem,
                sim_threshold=sim_threshold, alpha=alpha,
                beta=beta, top_k=top_k, temperature=temperature,
            )
            scheduler.step()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — loss: {loss:.4f}")

        # Extract embeddings + remap labels
        if verbose:
            print(f"  Storing embeddings + centroids...")
        model.eval()
        all_emb, all_lab = [], []
        with torch.no_grad():
            for x, y in get_task_loader(train_tasks[task_id], batch_size=128, shuffle=False):
                all_emb.append(model.get_features(x.to(device)))
                all_lab.append(remap_labels(y, task_id))

        all_emb = torch.cat(all_emb, dim=0)
        all_lab = torch.cat(all_lab, dim=0)

        if all_emb.size(0) > capacity_per_task:
            idx     = torch.randperm(all_emb.size(0))[:capacity_per_task]
            all_emb = all_emb[idx]
            all_lab = all_lab[idx]

        hopfield.store(all_emb, all_lab, task_id)

        # Centroid drift diagnostic
        if verbose and task_id > 0:
            print(f"  Centroid drift:")
            for prev in range(task_id):
                drift = hopfield.centroid_drift(model, train_tasks[prev], prev, device)
                status = 'HIGH — forgetting' if drift > 0.15 else 'low — stable'
                print(f"    Task {prev}: {drift:.4f} ({status})")

        # Evaluate
        if verbose:
            print(f"  Evaluating on tasks 0..{task_id}:")
        for eval_id in range(task_id + 1):
            acc = evaluate(
                model,
                get_task_loader(test_tasks[eval_id], batch_size=128, shuffle=False),
                device, task_id=eval_id
            )
            tracker.record(task_id, eval_id, acc)
            if verbose:
                print(f"    Task {eval_id} accuracy: {acc*100:.1f}%")

    return tracker
