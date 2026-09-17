"""Local benchmark adapters and deterministic synthetic fixtures."""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str = ""
    image: str | None = None
    choices: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    group: str = ""

    def query(self):
        """Remove all gold targets before retrieval, prompting, or inference."""
        return replace(self, answer="", answers=())

    def prompt(self):
        options = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(self.choices))
        return self.question + ("\n" + options + "\nAnswer with the option letter only." if options else "\nAnswer briefly.")


def load_jsonl(path):
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        obj["choices"] = tuple(obj.get("choices", ()))
        obj["answers"] = tuple(obj.get("answers", ()))
        if obj.get("image"):
            obj["image"] = str((path.parent / obj["image"]).resolve())
            if not Path(obj["image"]).is_file():
                raise FileNotFoundError(obj["image"])
        rows.append(Example(**obj))
    return rows


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(x)) + "\n" for x in rows), encoding="utf-8")


def scienceqa(root, split):
    """Official problems.json, pid_splits.json, images/{split}/{pid}/image.png."""
    root = Path(root)
    problems = json.loads((root / "problems.json").read_text(encoding="utf-8"))
    ids = json.loads((root / "pid_splits.json").read_text(encoding="utf-8"))[split]
    rows = []
    for pid in ids:
        p = problems[str(pid)]
        image = root / "images" / split / str(pid) / p["image"] if p.get("image") else None
        if image and not image.is_file():
            raise FileNotFoundError(image)
        # Lecture/solution are deliberately excluded: these contain target explanations.
        rows.append(Example("scienceqa:" + str(pid), p["question"] + ("\n" + p["hint"] if p.get("hint") else ""),
                            chr(65 + int(p["answer"])), str(image.resolve()) if image else None,
                            tuple(p["choices"]), group="scienceqa:" + str(pid)))
    return rows


def vqav2(questions, annotations, images, split="train2014"):
    """Read official local VQAv2 files; image-level grouping avoids cross-split reuse."""
    qs = json.loads(Path(questions).read_text(encoding="utf-8"))["questions"]
    anns = {int(a["question_id"]): a for a in json.loads(Path(annotations).read_text(encoding="utf-8"))["annotations"]}
    rows = []
    for q in qs:
        a = anns[int(q["question_id"])]
        image = Path(images) / f"COCO_{split}_{q['image_id']:012d}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append(Example("vqav2:" + str(q["question_id"]), q["question"], a["multiple_choice_answer"],
                            str(image.resolve()), answers=tuple(x["answer"] for x in a["answers"]),
                            group="coco:" + str(q["image_id"])))
    return rows


def partition(rows, seed=0, fractions=(0.5, 0.25)):
    groups = sorted({x.group or x.id for x in rows})
    np.random.default_rng(seed).shuffle(groups)
    a, b = int(len(groups)*fractions[0]), int(len(groups)*sum(fractions))
    buckets = [set(groups[:a]), set(groups[a:b]), set(groups[b:])]
    return [[x for x in rows if (x.group or x.id) in g] for g in buckets]


def validate_splits(*splits):
    seen_ids, seen_groups, seen_images = set(), set(), set()
    for rows in splits:
        if not rows:
            raise ValueError("Every split must be nonempty")
        ids = [x.id for x in rows]
        if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
            raise ValueError("Duplicate IDs within/across splits")
        groups = {x.group for x in rows if x.group}
        if seen_groups & groups:
            raise ValueError("Shared image/task groups across splits")
        hashes = {hashlib.sha256(Path(x.image).read_bytes()).hexdigest() for x in rows if x.image}
        if seen_images & hashes:
            raise ValueError("Identical image content across splits")
        seen_ids.update(ids)
        seen_groups.update(groups)
        seen_images.update(hashes)
        if any(not x.answer for x in rows):
            raise ValueError("Labeled data required for feedback and evaluation")


def synthetic(root, seed=0, sizes=(24, 12, 12)):
    """Toy colors/shapes. No benchmark claim; labels are opaque learned symbols."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    colors = [(220, 35, 35), (35, 210, 35), (35, 35, 220)]
    labels = ["dax", "wug", "blick"]
    splits = []
    for s, n in enumerate(sizes):
        rows = []
        for i in range(n):
            category = i % 3
            task = i % 2
            image = Image.new("RGB", (40, 40), tuple(int(x) for x in rng.integers(0, 20, 3)))
            draw = ImageDraw.Draw(image)
            box = (5 + i % 4, 5, 34, 34 - i % 3)
            if task:
                draw.ellipse(box, fill=colors[category])
            else:
                draw.rectangle(box, fill=colors[category])
            image.putpixel((0, 0), (s*60, i, seed % 255))
            path = root / f"{s}_{i}.png"
            image.save(path)
            # Two question-dependent naming conventions.
            question = "Name the color using " + ("codebook beta." if task else "codebook alpha.")
            rows.append(Example(f"toy:{s}:{i}", question, labels[(category+task) % 3], str(path), group=f"toy:{s}:{i}"))
        splits.append(rows)
    return splits
