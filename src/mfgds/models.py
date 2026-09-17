"""Answer-blind mock and local Hugging Face multimodal generation."""
from dataclasses import dataclass
import time
import numpy as np
from .embeddings import HashEncoder


@dataclass
class Prediction:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


def messages(query, demos):
    if query.answer or query.answers:
        raise ValueError("Pass query.query(): gold answers are forbidden in model inputs")
    def user(x):
        content = []
        if x.image:
            content.append({"type": "image", "path": x.image})
        content.append({"type": "text", "text": x.prompt()})
        return {"role": "user", "content": content}
    result = []
    for demo in demos:
        result.extend([user(demo), {"role": "assistant", "content": [{"type": "text", "text": demo.answer}]}])
    return result + [user(query)]


class MockLMM:
    """Deterministic example-copying simulator, not an LMM or benchmark proxy."""
    max_new_tokens = 8
    token_measurement = "estimated_words_plus_16_per_image"

    def __init__(self):
        self.encoder = HashEncoder()

    def count_tokens(self, query, demos):
        messages(query, demos)
        return sum(len(x.prompt().split()) + len(x.answer.split()) + 4 + (16 if x.image else 0)
                   for x in [*demos, query])

    def predict(self, query, demos):
        start = time.perf_counter()
        tokens = self.count_tokens(query, demos)
        answer = "unknown"
        if demos:
            q = self.encoder.encode([query], query=True)[0]
            d = self.encoder.encode(demos)
            # Artificial recency effect permits ordering smoke tests without targets.
            scores = d[:, :48] @ q[:48] + 0.8*(d[:, 48:96] @ q[48:96]) + np.arange(len(d))*0.02
            answer = demos[int(np.argmax(scores))].answer
        return Prediction(answer, tokens, len(answer.split()), time.perf_counter()-start)


class HuggingFaceLMM:
    token_measurement = "processor_input_ids_including_image_placeholders"

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="auto", max_new_tokens=32, revision="main"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, device_map=device,
            torch_dtype="auto", trust_remote_code=False).eval()
        self.max_new_tokens = max_new_tokens

    def inputs(self, query, demos):
        return self.processor.apply_chat_template(messages(query, demos), add_generation_prompt=True,
                                                  tokenize=True, return_dict=True, return_tensors="pt")

    def count_tokens(self, query, demos):
        return int(self.inputs(query, demos)["input_ids"].shape[-1])

    def predict(self, query, demos):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = self.inputs(query, demos).to(self.model.device)
        n = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        # Decoder-only multimodal chat models prepend their input IDs.
        generated = output[:, n:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return Prediction(text, int(n), int(generated.shape[-1]), time.perf_counter()-start)
