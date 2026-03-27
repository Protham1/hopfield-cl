"""
eval/metrics.py

Tracks two continual learning metrics:

1. Accuracy matrix R[i][j]:
   Accuracy on task j after training on task i.
   R is lower-triangular — you can only test tasks you've seen.

2. Average accuracy (AA):
   Mean of the diagonal and everything below it after the final task.
   This is the primary headline number.

3. Backward transfer (BWT):
   How much did earlier task accuracy change after seeing later tasks?
   BWT = (1 / T-1) * sum_{i<T} (R[T][i] - R[i][i])
   Negative BWT = forgetting. Positive = the network improved on old
   tasks after learning new ones (rare but possible with replay).

4. Per-task forgetting:
   R[i][i] - R[T][i] for each earlier task i.
   This is the per-task breakdown of BWT.

Usage:
    tracker = MetricsTracker(n_tasks=5)
    tracker.record(after_task=0, task_tested=0, accuracy=0.91)
    tracker.record(after_task=1, task_tested=0, accuracy=0.14)
    tracker.record(after_task=1, task_tested=1, accuracy=0.89)
    tracker.print_summary()
    tracker.plot("results/metrics.png")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


class MetricsTracker:
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # R[i][j] = accuracy on task j after training on tasks 0..i
        # -1.0 means not yet evaluated
        self.R = np.full((n_tasks, n_tasks), -1.0)

    def record(
        self,
        after_task: int,
        task_tested: int,
        accuracy: float,
    ) -> None:
        """
        Record one accuracy measurement.

        Args:
            after_task:   index of the most recently trained task (row)
            task_tested:  index of the task being evaluated (col)
            accuracy:     float in [0, 1]
        """
        assert task_tested <= after_task, (
            "Cannot test a task before training on it"
        )
        self.R[after_task][task_tested] = accuracy

    @property
    def average_accuracy(self) -> float:
        """
        Average accuracy across all (after_task, task_tested) pairs
        where after_task == n_tasks - 1 (i.e. after final task).
        """
        final_row = self.R[self.n_tasks - 1]
        valid = final_row[final_row >= 0]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    @property
    def backward_transfer(self) -> float:
        """
        BWT = mean degradation on all tasks except the last,
        measured from right after their own training vs after all tasks.
        Negative = forgetting.
        """
        if self.n_tasks < 2:
            return 0.0
        diffs = []
        for i in range(self.n_tasks - 1):
            r_at_training = self.R[i][i]
            r_at_end      = self.R[self.n_tasks - 1][i]
            if r_at_training >= 0 and r_at_end >= 0:
                diffs.append(r_at_end - r_at_training)
        return float(np.mean(diffs)) if diffs else 0.0

    def per_task_forgetting(self) -> dict:
        """
        Returns {task_id: forgetting_amount} for all tasks except last.
        Positive = forgot, negative = actually improved.
        """
        result = {}
        for i in range(self.n_tasks - 1):
            r_at_training = self.R[i][i]
            r_at_end      = self.R[self.n_tasks - 1][i]
            if r_at_training >= 0 and r_at_end >= 0:
                result[i] = r_at_training - r_at_end
        return result

    def print_summary(self, method_name: str = "") -> None:
        label = f" [{method_name}]" if method_name else ""
        print(f"\n{'='*50}")
        print(f"Results{label}")
        print(f"{'='*50}")
        print("\nAccuracy matrix R[after_task][task_tested]:")
        print("       " + "  ".join(f"T{j}" for j in range(self.n_tasks)))
        for i in range(self.n_tasks):
            row = []
            for j in range(self.n_tasks):
                if self.R[i][j] < 0:
                    row.append("  — ")
                else:
                    row.append(f"{self.R[i][j]*100:4.1f}")
            print(f"  T{i}:  " + "  ".join(row))

        print(f"\nAverage accuracy (after all tasks): {self.average_accuracy*100:.1f}%")
        print(f"Backward transfer:                  {self.backward_transfer*100:+.1f}%")
        print("\nPer-task forgetting:")
        for task_id, forgot in self.per_task_forgetting().items():
            direction = "forgot" if forgot > 0 else "improved"
            print(f"  Task {task_id}: {abs(forgot)*100:.1f}% {direction}")

    def plot(self, save_path: str, method_name: str = "") -> None:
        """
        Saves a figure with:
          Left:  accuracy matrix heatmap
          Right: per-task accuracy after final task (bar chart)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        label = f" — {method_name}" if method_name else ""
        fig.suptitle(f"Continual Learning Results{label}", fontsize=13)

        # Left: accuracy matrix
        ax = axes[0]
        masked = np.where(self.R >= 0, self.R, np.nan)
        im = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Task tested")
        ax.set_ylabel("After training task")
        ax.set_title("Accuracy matrix")
        ax.set_xticks(range(self.n_tasks))
        ax.set_yticks(range(self.n_tasks))
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if self.R[i][j] >= 0:
                    ax.text(j, i, f"{self.R[i][j]*100:.0f}",
                            ha="center", va="center", fontsize=9,
                            color="black")

        # Right: final accuracy per task
        ax2 = axes[1]
        final = self.R[self.n_tasks - 1]
        colors = ["#4CAF50" if v >= 0.7 else "#FF9800" if v >= 0.4 else "#F44336"
                  for v in final]
        bars = ax2.bar(range(self.n_tasks), [max(v, 0) for v in final], color=colors)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Task")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Accuracy after all tasks\n(avg: {self.average_accuracy*100:.1f}%)")
        ax2.set_xticks(range(self.n_tasks))
        for bar, val in zip(bars, final):
            if val >= 0:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                         f"{val*100:.0f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    t = MetricsTracker(n_tasks=3)
    # Simulated naive forgetting
    t.record(0, 0, 0.92)
    t.record(1, 0, 0.14); t.record(1, 1, 0.91)
    t.record(2, 0, 0.08); t.record(2, 1, 0.11); t.record(2, 2, 0.90)
    t.print_summary("naive")
    t.plot("/tmp/test_metrics.png", "naive")
    print("MetricsTracker OK.")