from dataclasses import replace
import pytest
from mfgds.data import Example, leakage_safe_training_pool, partition, validate_splits


def example(tmp_path, ident, content=None, group=None):
    image = None
    if content is not None:
        image = tmp_path / (ident + ".png")
        image.write_bytes(content)
    return Example(ident, "question", "A", str(image) if image else None, group=group or ident)


def test_repeated_training_images_stay_together(tmp_path):
    training = [example(tmp_path, "a", b"same"), example(tmp_path, "b", b"same")]
    training += [example(tmp_path, f"c{i}", str(i).encode()) for i in range(8)]
    evaluation = [example(tmp_path, "eval", b"held-out")]
    kept, held_out, audit = leakage_safe_training_pool(training, evaluation)
    assert len(kept) == len(training) and audit["excluded_training_rows"] == 0
    assert kept[0].group == kept[1].group
    for seed in range(10):
        parts = partition(kept, seed, (.8, .2))
        assert any({"a", "b"} <= {x.id for x in split} for split in parts[:2])
        validate_splits(parts[0], parts[1], held_out)


def test_evaluation_overlap_excludes_entire_connected_component(tmp_path):
    training = [example(tmp_path, "a", b"one", "shared"),
                example(tmp_path, "b", b"two", "shared"),
                example(tmp_path, "c", b"two", "different"),
                example(tmp_path, "retained", b"three"),
                example(tmp_path, "text")]
    evaluation = [example(tmp_path, "eval1", b"one"), example(tmp_path, "eval2", b"one")]
    kept, held_out, audit = leakage_safe_training_pool(training, evaluation)
    assert [x.id for x in kept] == ["retained", "text"]
    assert [x.id for x in held_out] == ["eval1", "eval2"]
    assert audit["excluded_training_ids"] == ["a", "b", "c"]
    assert audit["excluded_training_rows"] == 3
    assert held_out[0].group == held_out[1].group
    validate_splits(kept, held_out)
    # Grouping must not depend on gold labels.
    changed = [replace(x, answer="B") for x in training]
    assert leakage_safe_training_pool(changed, evaluation)[2] == audit


def test_text_only_rows_and_original_groups():
    training = [Example(str(i), "same question", "A", group=str(i // 2)) for i in range(10)]
    evaluation = [Example("eval", "same question", "A", group="evaluation")]
    first = leakage_safe_training_pool(training, evaluation)
    second = leakage_safe_training_pool(training, evaluation)
    assert first == second
    kept, held_out, audit = first
    assert len(kept) == 10 and audit["training_components"] == 5
    assert kept[0].group == kept[1].group
    validate_splits(kept, held_out)


def test_official_duplicate_ids_rejected():
    row = Example("duplicate", "Q", "A")
    with pytest.raises(ValueError, match="Duplicate IDs"):
        leakage_safe_training_pool([row], [row])
