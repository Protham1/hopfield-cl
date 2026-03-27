import torch
import random


class EmbeddingBuffer:
    def __init__(self, capacity_per_task=200):
        self.capacity = capacity_per_task
        self.storage = {}

    def add_task(self, task_id, model, dataset, device):
        model.eval()
        reservoir = []
        n_seen = 0

        with torch.no_grad():
            for i in range(len(dataset)):
                x, y = dataset[i]

                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)

                x = x.unsqueeze(0).to(device)

                # 🔥 extract embedding
                h = model.extract_features(x).squeeze(0).cpu()

                n_seen += 1

                if len(reservoir) < self.capacity:
                    reservoir.append((h, y))
                else:
                    j = random.randint(0, n_seen - 1)
                    if j < self.capacity:
                        reservoir[j] = (h, y)

        self.storage[task_id] = reservoir

        print(f"  EmbeddingBuffer: stored {len(reservoir)} embeddings for task {task_id}")

    def get_all(self):
        all_samples = []
        for r in self.storage.values():
            all_samples.extend(r)
        return all_samples

    def collate(self, samples):
        h = torch.stack([s[0] for s in samples])
        y = torch.tensor([s[1] for s in samples], dtype=torch.long)
        return h, y

    @property
    def n_tasks_stored(self):
        return len(self.storage)