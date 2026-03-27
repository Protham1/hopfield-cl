import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class LabelAwareHopfieldMemory:
    """
    Label-aware Hopfield episodic memory with inter-class contrast retrieval.

    risk(m) = alpha * sim(query, m) - beta * sim(m, centroid[m.label])

    High risk = similar to current query AND far from own class centroid.
    Inter-class contrast: only retrieves from previous tasks, not current.
    """

    def __init__(self):
        self.memory   = None
        self.labels   = None
        self.task_ids = None
        self.centroids: Dict[Tuple[int, int], torch.Tensor] = {}

    def store(self, embeddings: torch.Tensor, labels: torch.Tensor, task_id: int):
        embeddings = embeddings.detach().cpu()
        labels     = labels.detach().cpu()
        tids       = torch.full((embeddings.size(0),), task_id, dtype=torch.long)

        for c in labels.unique():
            mask = labels == c
            self.centroids[(task_id, c.item())] = embeddings[mask].mean(0)

        if self.memory is None:
            self.memory   = embeddings
            self.labels   = labels
            self.task_ids = tids
        else:
            self.memory   = torch.cat([self.memory,   embeddings], dim=0)
            self.labels   = torch.cat([self.labels,   labels],     dim=0)
            self.task_ids = torch.cat([self.task_ids, tids],       dim=0)

        print(f"  LabelAwareHopfield: stored {embeddings.size(0)} embeddings for task {task_id}")

    def retrieve(
        self,
        query: torch.Tensor,
        current_task_id: int,
        top_k: int = 16,
        temperature: float = 0.3,
        alpha: float = 1.0,
        beta: float = 0.5,
    ):
        if self.memory is None:
            return None

        device   = query.device
        memory   = self.memory.to(device)
        labels   = self.labels.to(device)
        task_ids = self.task_ids.to(device)

        # Inter-class contrast — exclude current task
        mask = task_ids != current_task_id
        if mask.sum() == 0:
            return None

        mem_filtered = memory[mask]
        lab_filtered = labels[mask]
        tid_filtered = task_ids[mask]

        # Query similarity
        query_mean = F.normalize(query.mean(0, keepdim=True), dim=1)
        mem_norm   = F.normalize(mem_filtered, dim=1)
        query_sim  = torch.matmul(query_mean, mem_norm.T).squeeze(0)

        # Centroid similarity — how well anchored is each memory
        centroid_sim = torch.zeros(mem_filtered.size(0), device=device)
        for i in range(mem_filtered.size(0)):
            tid = tid_filtered[i].item()
            lab = lab_filtered[i].item()
            key = (tid, lab)
            if key in self.centroids:
                centroid = self.centroids[key].to(device)
                centroid_sim[i] = F.cosine_similarity(
                    mem_filtered[i].unsqueeze(0),
                    centroid.unsqueeze(0),
                    dim=1
                ).squeeze()

        # Risk score: relevant to query AND far from own centroid
        risk_score = alpha * query_sim - beta * centroid_sim

        # Top-k by risk, temperature softmax, weighted sum
        k = min(top_k, mem_filtered.size(0))
        topk_vals, topk_idx = torch.topk(risk_score, k=k, dim=0)
        selected  = mem_filtered[topk_idx]
        attn      = F.softmax(topk_vals / temperature, dim=0)
        retrieved = torch.sum(attn.unsqueeze(-1) * selected, dim=0, keepdim=True)
        retrieved = retrieved.expand(query.size(0), -1)

        return retrieved

    def centroid_drift(self, model, dataset, task_id: int, device) -> float:
        if not any(k[0] == task_id for k in self.centroids):
            return 0.0
        from data.cifar10_tasks import get_task_loader
        from train.hopfield_train import remap_labels
        model.eval()
        current_embs, current_labs = [], []
        with torch.no_grad():
            for x, y in get_task_loader(dataset, batch_size=128, shuffle=False):
                current_embs.append(model.get_features(x.to(device)).cpu())
                current_labs.append(remap_labels(y, task_id))
        current_embs = torch.cat(current_embs)
        current_labs = torch.cat(current_labs)
        drifts = []
        for c in current_labs.unique():
            mask    = current_labs == c
            cur_cen = current_embs[mask].mean(0)
            key     = (task_id, c.item())
            if key in self.centroids:
                drift = 1.0 - F.cosine_similarity(
                    cur_cen.unsqueeze(0),
                    self.centroids[key].unsqueeze(0),
                    dim=1
                ).item()
                drifts.append(drift)
        return sum(drifts) / len(drifts) if drifts else 0.0

    @property
    def n_tasks_stored(self):
        return self.task_ids.unique().size(0) if self.task_ids is not None else 0
