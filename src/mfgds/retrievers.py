"""Exact candidate search, learned utility regression, and external GRIP scores."""
import json
import numpy as np
import torch
from torch import nn
from .embeddings import normalize


class VectorIndex:
    def __init__(self, vectors, backend="auto"):
        self.vectors = normalize(vectors).astype(np.float32)
        if backend not in ("auto", "faiss", "sklearn"):
            raise ValueError(backend)
        self.backend = backend
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
        if self.backend == "faiss":
            scores, ids = self.index.search(q, k)
            return ids[0], scores[0]
        distances, ids = self.index.kneighbors(q, n_neighbors=k)
        return ids[0], 1 - distances[0]


def pair_features(q, d):
    q = np.broadcast_to(q, d.shape)
    return np.concatenate([q, d, q*d, np.abs(q-d)], axis=-1).astype(np.float32)


class UtilityRetriever:
    def __init__(self, dim, seed=0, hidden=64):
        torch.manual_seed(seed)
        self.dim, self.seed, self.hidden = dim, seed, hidden
        self.net = nn.Sequential(nn.Linear(4*dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh())

    def fit(self, features, rewards, epochs=60, lr=0.003):
        x, y = torch.as_tensor(features), torch.as_tensor(rewards, dtype=torch.float32)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        history = []
        self.net.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.net(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        self.net.eval()
        return history

    def score(self, query, demos):
        with torch.inference_mode():
            return self.net(torch.from_numpy(pair_features(query, demos))).squeeze(-1).numpy()

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "dim": self.dim, "seed": self.seed, "hidden": self.hidden}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(payload["dim"], payload["seed"], payload["hidden"])
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
