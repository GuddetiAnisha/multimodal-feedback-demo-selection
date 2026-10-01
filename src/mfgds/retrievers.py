"""Exact candidate search, learned utility regression, and external GRIP scores."""
import json
import numpy as np
import torch
from torch import nn
from .embeddings import normalize
from .devices import resolve_device, cpu_snapshot


class VectorIndex:
    def __init__(self, vectors, backend="auto", device="cpu"):
        self.vectors = normalize(vectors).astype(np.float32)
        if backend not in ("auto", "faiss", "sklearn", "torch"):
            raise ValueError(backend)
        self.backend = backend
        if backend == "torch":
            self.device = resolve_device(device)
            self.tensor = torch.as_tensor(self.vectors, device=self.device)
            return
        if backend != "sklearn":
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.vectors.shape[1])
                self.index.add(self.vectors)
                self.backend = "faiss"
                return
            except ImportError:
                if backend == "faiss":
                    raise
        from sklearn.neighbors import NearestNeighbors
        self.index = NearestNeighbors(metric="cosine", algorithm="brute").fit(self.vectors)
        self.backend = "sklearn"

    def search(self, query, k):
        k = min(k, len(self.vectors))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        q = normalize(query).reshape(1, -1).astype(np.float32)
        if self.backend == "torch":
            with torch.inference_mode():
                similarities = self.tensor @ torch.as_tensor(q[0], device=self.device)
                ids = torch.argsort(similarities, descending=True, stable=True)[:k]
                return ids.cpu().numpy(), similarities[ids].cpu().numpy()
        if self.backend == "faiss":
            scores, ids = self.index.search(q, k)
            return ids[0], scores[0]
        distances, ids = self.index.kneighbors(q, n_neighbors=k)
        return ids[0], 1 - distances[0]


def pair_features(q, d):
    q = np.broadcast_to(q, d.shape)
    return np.concatenate([q, d, q*d, np.abs(q-d)], axis=-1).astype(np.float32)


class UtilityRetriever:
    def __init__(self, dim, seed=0, hidden=64, device="cpu"):
        self.device = resolve_device(device)
        torch.manual_seed(seed)
        self.dim, self.seed, self.hidden = dim, seed, hidden
        self.net = nn.Sequential(nn.Linear(4*dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh()).to(self.device)

    def fit(self, features, rewards, epochs=60, lr=0.003, resume_state=None, checkpoint=None):
        x = torch.as_tensor(features, device=self.device)
        y = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        history = []
        if resume_state is not None:
            from .checkpoints import restore_rng
            self.net.load_state_dict(resume_state["model"])
            optimizer.load_state_dict(resume_state["optimizer"])
            history = list(resume_state["history"])
            restore_rng(resume_state["rng"])
        self.net.train()
        for _ in range(len(history), epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.net(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
            if checkpoint is not None:
                from .checkpoints import rng_state
                checkpoint({"model": cpu_snapshot(self.net.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                            "scheduler": None, "epoch": len(history), "history": list(history), "rng": rng_state()})
        self.net.eval()
        return history

    def score(self, query, demos):
        with torch.inference_mode():
            inputs = torch.as_tensor(pair_features(query, demos), device=self.device)
            return self.net(inputs).squeeze(-1).cpu().numpy()

    def save(self, path):
        import io
        from .checkpoints import atomic_bytes
        buffer = io.BytesIO()
        torch.save({"state": cpu_snapshot(self.net.state_dict()), "dim": self.dim, "seed": self.seed, "hidden": self.hidden}, buffer)
        atomic_bytes(path, buffer.getvalue())

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(payload["dim"], payload["seed"], payload["hidden"], device=device)
        obj.net.load_state_dict(payload["state"])
        obj.net.eval()
        return obj


class ExternalGrip:
    """JSON {query_id: {demo_id: score}} exported by an independent GRIP run."""
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.scores = json.load(f)

    def score(self, query_id, demo_ids):
        try:
            values = np.array([self.scores[query_id][i] for i in demo_ids], dtype=np.float32)
        except KeyError as e:
            raise ValueError(f"Missing external GRIP score: {e}") from e
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite external GRIP scores")
        return values
