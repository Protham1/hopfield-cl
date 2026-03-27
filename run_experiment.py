import torch
import torch.nn as nn
import numpy as np
import random
from data.cifar10_tasks import get_task_datasets, get_task_loader, TASK_CLASSES
from models.resnet import build_model, CIFARResNet18
from train.hopfield_train import evaluate, remap_labels
from train.hopfield_label_aware_train import run_label_aware_hopfield
from memory.hopfield_label_aware import LabelAwareHopfieldMemory
from eval.metrics import MetricsTracker
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def patched_add_head(self, task_id, num_classes=2):
    head = nn.Linear(512, num_classes)
    nn.init.normal_(head.weight, 0, 0.01)
    nn.init.constant_(head.bias, 0)
    self.heads[str(task_id)] = head.to(device)
CIFARResNet18.add_head = patched_add_head

SEEDS = [42, 123, 456]
naive_scores = []
la_scores = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed} — NAIVE")
    print(f"{'='*60}")
    set_seed(seed)
    train_tasks, test_tasks = get_task_datasets(n_tasks=5)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    tracker = MetricsTracker(n_tasks=5)

    for task_id in range(5):
        model.add_head(task_id, num_classes=2)
        loader = get_task_loader(train_tasks[task_id], batch_size=64, shuffle=True)
        opt = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        sch = CosineAnnealingLR(opt, T_max=10)
        for epoch in range(10):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                y = remap_labels(y, task_id)
                opt.zero_grad()
                criterion(model(x, task_id=task_id), y).backward()
                opt.step()
            sch.step()
        for eval_id in range(task_id + 1):
            acc = evaluate(model, get_task_loader(test_tasks[eval_id], batch_size=128, shuffle=False), device, eval_id)
            tracker.record(task_id, eval_id, acc)
        print(f"  Task {task_id} done — current task acc: {tracker.R[task_id][task_id]*100:.1f}%")

    score = tracker.average_accuracy * 100
    naive_scores.append(score)
    print(f"Naive seed={seed} avg accuracy: {score:.1f}%")
    tracker.plot(f'/content/drive/MyDrive/HOP/results/naive_seed{seed}.png', f'naive seed={seed}')

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed} — LABEL-AWARE HOPFIELD")
    print(f"{'='*60}")
    set_seed(seed)
    train_tasks, test_tasks = get_task_datasets(n_tasks=5)
    model    = build_model().to(device)
    hopfield = LabelAwareHopfieldMemory()

    tracker = run_label_aware_hopfield(
        model=model,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        hopfield=hopfield,
        n_epochs=10,
        batch_size=64,
        capacity_per_task=500,
        lambda_mem=0.15,
        alpha=1.0,
        beta=0.8,
        top_k=16,
        temperature=0.3,
        sim_threshold=0.05,
        device=device,
        verbose=False,
    )

    score = tracker.average_accuracy * 100
    la_scores.append(score)
    print(f"Label-aware seed={seed} avg accuracy: {score:.1f}%")
    tracker.plot(f'/content/drive/MyDrive/HOP/results/la_seed{seed}.png', f'label-aware seed={seed}')

# Final summary
print(f"\n{'='*60}")
print(f"FINAL 3-SEED COMPARISON")
print(f"{'='*60}")
print(f"\nNaive results:       {[f'{s:.1f}%' for s in naive_scores]}")
print(f"Label-aware results: {[f'{s:.1f}%' for s in la_scores]}")
print(f"\nNaive      mean ± std: {np.mean(naive_scores):.1f}% ± {np.std(naive_scores):.1f}%")
print(f"Label-aware mean ± std: {np.mean(la_scores):.1f}% ± {np.std(la_scores):.1f}%")
print(f"\nMean improvement: {np.mean(la_scores) - np.mean(naive_scores):+.1f}%")
print(f"Label-aware wins: {sum(l > n for l, n in zip(la_scores, naive_scores))}/3 seeds")

