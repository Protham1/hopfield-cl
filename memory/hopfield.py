import torch
import torch.nn.functional as F


class HopfieldMemory:
    def __init__(self):
        self.memory = None

    def store(self, embeddings):
        if self.memory is None:
            self.memory = embeddings
        else:
            self.memory = torch.cat([self.memory, embeddings], dim=0)

    def retrieve(self, query, top_k=8, temperature=0.1):

        if self.memory is None:
            return None

        # 🔥 normalize (CRITICAL)
        query_norm = F.normalize(query, dim=1)
        memory_norm = F.normalize(self.memory, dim=1)

        # cosine similarity
        scores = torch.matmul(query_norm, memory_norm.T)

        # top-k selection
        k = min(top_k, self.memory.size(0))
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)

        selected = self.memory[topk_idx]  # (B, k, D)

        # 🔥 temperature softmax
        attn = F.softmax(topk_vals / temperature, dim=-1)

        retrieved = torch.sum(attn.unsqueeze(-1) * selected, dim=1)

        return retrieved