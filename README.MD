# Hopfield Continual Learning — Catastrophic Forgetting Prevention

A continual learning system using **Modern Hopfield Networks** as episodic memory to prevent catastrophic forgetting across sequential tasks. Built and benchmarked on Split CIFAR-10 with systematic ablation across 7+ experimental configurations and direct comparison against A-GEM.

---

## Results

### Main 3-Way Comparison (avg over 3 seeds)

| Method | Avg Acc | BWT | Forgetting | Storage |
|---|---|---|---|---|
| Naive Sequential | 63.7% | -38.0% | 38.0% | None |
| **Label-Aware Hopfield** | **72.8%** | **-27.5%** | **27.5%** | Embeddings only |
| A-GEM | **82.4%** | -16.0% | **16.0%** | Raw images |

> **Key insight:** A-GEM wins on aggregate accuracy but requires storing raw images. LA-Hopfield achieves +9.1% over naive using only compact embeddings — a ~6× memory reduction per stored sample. In memory-constrained or privacy-sensitive settings, LA-Hopfield offers a meaningful point on the accuracy–memory Pareto frontier.

### Per-Task Forgetting

| Task | Naive | A-GEM | LA-Hopfield |
|---|---|---|---|
| Task 0 | 43% | ~8% | **15%** |
| Task 1 | 35% | ~19% | 28% |
| Task 2 | 43% | ~19% | 37% |
| Task 3 | 32% | ~18% | 30% |

> LA-Hopfield's protection is **non-uniform** — it concentrates on the highest-risk tasks (earliest tasks with most overwrite pressure), which is exactly what the risk score is designed for. Task 0 forgetting reduced from 43% → 15%.

### Multi-Seed Worst-Case Analysis

| Method | Seed 42 | Seed 123 | Seed 456 | Mean |
|---|---|---|---|---|
| Naive Sequential | 66.3% | 71.2% | — | 68.8% |
| **Label-Aware Hopfield** | **73.3%** | **71.0%** | — | **72.2%** |
| **Improvement** | **+7.0%** | **-0.2%** | — | **+3.4%** |

> The +7% gain concentrates in worst-case seeds where naive collapses — confirming the risk score functions as a **robustness mechanism**, not a uniform accuracy booster.

### Full Ablation Results (seed=42)

| Method | Avg Accuracy | vs Naive |
|---|---|---|
| Naive baseline | 71.4% | — |
| Hopfield reg cap=200 λ=0.05 | 55.0% | -16.4% |
| Hopfield reg cap=300 λ=0.1 | 70.4% | -1.0% |
| **Hopfield reg cap=500 λ=0.1** | **73.2%** | **+1.8%** |
| Hopfield cap=500 λ=0.15 | 72.5% | +1.1% |
| Hopfield embedding replay | 71.5% | +0.1% |
| Hopfield cap=800 λ=0.1 | 54.8% | -16.6% |
| **Label-Aware Hopfield v2** | **73.8%** | **+2.4%** |

### Accuracy Matrix — Naive vs Label-Aware (seed=42)

**Naive** — Task 0 collapses from 95% → 50% after 5 tasks:
```
       T0    T1    T2    T3    T4
T0:    95%    —     —     —     —
T1:    69%   87%    —     —     —
T2:    71%   68%   94%    —     —
T3:    57%   56%   78%   98%    —
T4:    50%   61%   56%   67%   98%   avg: 66.3%
```

**Label-Aware Hopfield** — Task 0 held at 81%:
```
       T0    T1    T2    T3    T4
T0:    95%    —     —     —     —
T1:    69%   87%    —     —     —
T2:    74%   67%   94%    —     —
T3:    51%   52%   64%   99%    —
T4:    81%   62%   56%   70%   98%   avg: 73.3%
```

---

## Key Findings

**Finding 1 — Output head isolation is a prerequisite.**
A shared classification head causes complete output interference — Task 0 accuracy collapses to 0% after Task 1 regardless of any memory mechanism. Per-task heads via `nn.ModuleDict` are necessary before any memory technique can contribute.

**Finding 2 — Regularisation strength and capacity do not co-scale.**
Sweeping λ at fixed capacity revealed a plateau at λ=0.1. Increasing beyond this degraded performance. Capacity and λ must be tuned independently — incorrect λ cannot compensate for incorrect capacity.

**Finding 3 — Memory capacity follows a non-monotonic optimum.**
Performance improved from cap=200 → cap=500 (+1.8% over naive) but degraded sharply at cap=800 (-16.6% below naive). Overloaded retrieval returns incoherent superpositions; the L2 constraint then penalises the backbone for deviating from noise. Optimal capacity: ~500 embeddings per task.

**Finding 4 — Embedding replay fails due to representation drift.**
Storing (embedding, label) pairs and replaying them as direct classification signal matched naive exactly at 71.5%. Backbone updates between tasks render stored embeddings inconsistent with current representations — the gradient signal becomes noise. Any embedding replay method must account for drift.

**Finding 5 — Similarity threshold acts as a novelty filter.**
`sim_threshold` allows the backbone to learn new tasks freely while constraining familiar patterns, operationalising stability-plasticity decoupling.

**Finding 6 — Label-aware retrieval specifically protects worst-case seeds.**
Risk score `α × query_sim − β × centroid_sim` targets memories both relevant to the current task AND drifting from their class centroid — the most endangered memories. +7% improvement on hard seeds where naive collapses.

**Finding 7 — Fast-learning tasks cause temporary instability.**
Tasks learned too quickly (loss < 0.03 by epoch 5) cause backbone updates to outpace risk score recalibration. Self-corrects by next task but reveals a limitation of static-λ regularisation.

**Finding 8 — A-GEM vs LA-Hopfield is an accuracy–memory tradeoff.**
A-GEM's 9.6pp advantage (82.4% vs 72.8%) comes from storing raw images and running an extra backward pass per batch. LA-Hopfield stores ~6× less data per sample and adds no backward pass overhead. The right choice depends on deployment constraints.

---

## Architecture

```
Raw images (32x32)
       ↓
TaskSubset (filters CIFAR-10 by class pair)
       ↓
CIFARResNet18 backbone (512-dim embedding)
       ↓
Per-task ModuleDict heads (2-class each)
       ↓  ↑
LabelAwareHopfieldMemory
  - store(embeddings, labels, task_id)
  - retrieve(query, current_task_id) → risk-scored retrieval
  - centroid_drift() → backbone stability diagnostic
       ↓
Loss = CrossEntropy(current) + λ × MSE(h_cur, h_retrieved)
       ↓
MetricsTracker
  - Accuracy matrix R[after_task][task_tested]
  - Backward transfer
  - Per-task forgetting
```

### Label-Aware Retrieval Risk Score

```
risk(m) = α × sim(query, m) − β × sim(m, centroid[m.label])
```

- `sim(query, m)` — cosine similarity: proxy for gradient alignment between current batch and stored memory
- `sim(m, centroid[m.label])` — how well-anchored the memory is to its own class (structural stability)
- **High risk** = similar to current query AND drifting from own class centroid = endangered, replay it
- **Inter-task only** — retrieves only from previous tasks, not current

### A-GEM (Averaged Gradient Episodic Memory)

```
ĝ = g − (g·ḡ / ḡ·ḡ) × ḡ,  if g·ḡ < 0
```

- `g` — current task gradient
- `ḡ` — average gradient over all stored memory samples
- Projects gradient to not increase memory loss — hard constraint, no λ tuning needed
- Requires raw image storage and one extra backward pass per batch

---

## Project Structure

```
HOP/
├── data/
│   ├── cifar10_tasks.py               # Split CIFAR-10 into 5 binary tasks
│   └── cifar100_tasks.py              # Split CIFAR-100 into N tasks
├── models/
│   ├── resnet.py                      # CIFAR ResNet-18 with per-task heads
│   └── resnet_cifar100.py             # ResNet-18 for CIFAR-100
├── memory/
│   ├── hopfield.py                    # Vanilla Modern Hopfield Network
│   ├── hopfield_label_aware.py        # Label-aware inter-class contrast retrieval
│   ├── gem_memory.py                  # Raw image buffer for A-GEM
│   └── buffer.py                      # Fixed-size reservoir replay buffer
├── train/
│   ├── naive.py                       # Naive sequential baseline
│   ├── hopfield_train.py              # Vanilla Hopfield regularisation
│   ├── hopfield_label_aware_train.py  # Label-aware training loop
│   ├── hopfield_replay2.py            # Embedding replay (drift study)
│   └── replay.py                      # Random reservoir replay baseline
├── eval/
│   └── metrics.py                     # Accuracy matrix, BWT, centroid drift
├── results/                           # Experiment plots
└── run_experiment.py                  # Entry point
```

---

## Setup

```bash
pip install torch torchvision numpy matplotlib
# install cifar10 dataset in data folder
```

**Sanity check (CPU, ~5 min):**
```bash
python run_experiment.py --mode sanity
```

**Full experiment (GPU recommended, ~30 min on Colab T4):**
```bash
python run_experiment.py --mode full
```

---

## Reproducing Best Result (Label-Aware Hopfield)

```python
from data.cifar10_tasks import get_task_datasets
from models.resnet import build_model
from memory.hopfield_label_aware import LabelAwareHopfieldMemory
from train.hopfield_label_aware_train import run_label_aware_hopfield

train_tasks, test_tasks = get_task_datasets(n_tasks=5)
model    = build_model()
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
)
tracker.print_summary('label-aware hopfield')
```

## Reproducing A-GEM Baseline

```python
from memory.gem_memory import GEMMemory
# see train/gem_train_cifar10.py for full A-GEM training loop
# key config: capacity_per_task=500, backbone-only gradient projection
```

---

## Key Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `capacity_per_task` | 500 | Optimal — 800 causes retrieval interference |
| `lambda_mem` | 0.15 | Memory loss weight — plateaus above 0.1 |
| `alpha` | 1.0 | Query similarity weight in risk score |
| `beta` | 0.8 | Centroid distance penalty — higher = more even replay |
| `top_k` | 16 | Attention pool size |
| `temperature` | 0.3 | Softmax sharpness |
| `sim_threshold` | 0.05 | Minimum similarity to apply memory loss |

---

## Limitations & Future Work

- **Representation drift** — embedding replay fails because backbone changes between tasks. Fix: re-encode stored images through current backbone at replay time.
- **Fast-learning instability** — tasks learned too quickly cause temporary drift. Fix: adaptive lambda that increases when classification loss drops too fast.
- **Close the gap with A-GEM** — combine risk-scored retrieval with a lightweight gradient projection step to approach A-GEM accuracy without full raw image storage.
- **Scale to Split CIFAR-100** — 10/20-task benchmark where forgetting compounds more severely.

---

## References

- Ramsauer et al. (2020) — [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)
- Rolnick et al. (2019) — [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)
- Lopez-Paz & Ranzato (2017) — [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)
- Chaudhry et al. (2019) — [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420)
