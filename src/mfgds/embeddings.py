"""Separate image, question, and answer channels; queries never encode gold answers."""
import hashlib
import re
import numpy as np
from PIL import Image


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class HashEncoder:
    """Offline smoke-test encoder; pixels plus deterministic signed word hashing."""
    def __init__(self, dim=48):
        self.dim = dim

    def text(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
            v[h % self.dim] += 1 if h & 256 else -1
        return normalize(v)

    def image(self, path):
        if not path:
            return np.zeros(self.dim, dtype=np.float32)
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB").resize((4, 4)), dtype=np.float32).reshape(-1) / 255
        return normalize(np.resize(pixels, self.dim))

    def encode(self, rows, query=False, channels=("image", "question", "answer")):
        return np.asarray([np.concatenate([
            self.image(x.image) if "image" in channels else np.zeros(self.dim),
            self.text(x.prompt()) if "question" in channels else np.zeros(self.dim),
            self.text(x.answer) if "answer" in channels and not query else np.zeros(self.dim),
        ]) for x in rows], dtype=np.float32)


class ClipEncoder(HashEncoder):
    """Sentence-transformers CLIP encodes PIL images and strings in one space."""
    def __init__(self, model="clip-ViT-B-32", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def text(self, text):
        return normalize(self.model.encode(text, convert_to_numpy=True)) if text else np.zeros(self.dim, dtype=np.float32)

    def image(self, path):
        if not path:
            return np.zeros(self.dim, dtype=np.float32)
        with Image.open(path) as im:
            return normalize(self.model.encode(im.convert("RGB"), convert_to_numpy=True))


def retrieval_view(vectors, method):
    image, question, answer = np.split(vectors, 3, axis=-1)
    if method == "visual":
        return normalize(image)
    if method == "textual":
        return normalize(question + answer)
    if method == "multimodal":
        return normalize(image + question + answer)
    raise ValueError(method)
