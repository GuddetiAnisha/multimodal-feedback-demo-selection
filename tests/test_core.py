import json
from dataclasses import replace
import numpy as np
import pytest
from mfgds.data import Example, synthetic, validate_splits, partition, save_jsonl, load_jsonl, scienceqa, vqav2
from mfgds.embeddings import HashEncoder, retrieval_view
from mfgds.models import MockLMM, messages
from mfgds.retrievers import VectorIndex, UtilityRetriever, pair_features, ExternalGrip
from mfgds.selectors import select, order, fit_context, diversity
from mfgds.evaluation import score


def test_query_answer_cannot_change_embeddings_or_prompt():
    a = Example("x", "Which?", "SECRET", choices=("one", "two"), answers=("SECRET",))
    b = replace(a, answer="OTHER", answers=("OTHER",))
    encoder = HashEncoder()
    np.testing.assert_array_equal(encoder.encode([a], query=True), encoder.encode([b], query=True))
    assert messages(a.query(), []) == messages(b.query(), [])
    assert "SECRET" not in json.dumps(messages(a.query(), []))
    with pytest.raises(ValueError):
        MockLMM().predict(a, [])


def test_demo_answer_is_represented():
    e = HashEncoder()
    a, b = Example("a", "q", "yes"), Example("b", "q", "no")
    assert not np.array_equal(e.encode([a]), e.encode([b]))
    np.testing.assert_array_equal(e.encode([a], channels=("image", "question")), e.encode([b], channels=("image", "question")))


def test_splits_and_content_leakage(tmp_path):
    splits = synthetic(tmp_path)
    validate_splits(*splits)
    with pytest.raises(ValueError):
        validate_splits(splits[0], splits[0])
    with pytest.raises(ValueError, match="image content"):
        validate_splits([splits[0][0]], [replace(splits[0][0], id="other", group="other")])
    grouped = [Example(str(i), "q", "a", group=str(i//2)) for i in range(20)]
    validate_splits(*partition(grouped))
    save_jsonl(splits[0], tmp_path / "rows.jsonl")
    assert load_jsonl(tmp_path / "rows.jsonl") == splits[0]


def test_index_and_zero_vectors():
    index = VectorIndex(np.array([[1, 0], [0, 1], [0, 0]], np.float32), "sklearn")
    ids, scores = index.search(np.array([1, 0]), 99)
    assert ids[0] == 0 and scores[0] == pytest.approx(1)
    assert np.isfinite(index.search(np.zeros(2), 2)[1]).all()
    assert len(index.search(np.ones(2), 0)[0]) == 0


def test_faiss_parity_when_installed():
    pytest.importorskip("faiss")
    vectors = np.array([[1, 0], [.7, .7], [0, 1]], np.float32)
    a = VectorIndex(vectors, "faiss").search(np.array([1, .1]), 3)
    b = VectorIndex(vectors, "sklearn").search(np.array([1, .1]), 3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1], atol=1e-6)


def test_mmr_and_ordering():
    v = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
    assert select([1, .95, .9], v, 2, .5) == [0, 2]
    assert select([1, .95, .9], v, 2, 0) == [0, 1]
    assert select([1, 2, 3], v, 0) == []
    assert len(select([1, 2, 3], v, 9)) == 3
    assert order([0, 1], [2, 1], "best_last", np.random.default_rng(0)) == [1, 0]
    assert diversity(v[[0, 1]]) == (0, 1)
    assert diversity(v[[0, 2]]) == (1, 0)


def test_context_removes_whole_demos():
    model = MockLMM()
    query = Example("q", "question").query()
    demo = Example("d", "long example question", "answer")
    budget = model.count_tokens(query, [demo]) + model.max_new_tokens
    kept, tokens = fit_context(query, [demo, demo], model, budget)
    assert kept == [demo] and tokens + model.max_new_tokens == budget
    with pytest.raises(ValueError, match="Query"):
        fit_context(query, [], model, 1)


def test_learning_and_checkpoint(tmp_path):
    q = np.array([1, 0], np.float32)
    d = np.array([[1, 0], [0, 1]], np.float32)
    model = UtilityRetriever(2, seed=4)
    history = model.fit(pair_features(q, d), [1, -1], epochs=100)
    assert history[-1] < history[0]*.1
    assert model.score(q, d)[0] > .7 and model.score(q, d)[1] < -.7
    model.save(tmp_path / "model.pt")
    np.testing.assert_allclose(UtilityRetriever.load(tmp_path / "model.pt").score(q, d), model.score(q, d))


def test_metrics_and_external(tmp_path):
    assert score("B.", Example("x", "q", "B", choices=("red", "blue"))) == 1
    assert score("blue", Example("x", "q", "B", choices=("red", "blue"))) == 1
    assert score("blue because A", Example("x", "q", "B", choices=("red", "blue"))) == 0
    assert score("cat", Example("x", "q", "cat", answers=("cat",)*3 + ("dog",)*7)) == pytest.approx(.9)
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"q": {"d": .2}}))
    assert ExternalGrip(path).score("q", ["d"])[0] == pytest.approx(.2)
    with pytest.raises(ValueError):
        ExternalGrip(path).score("missing", ["d"])


def test_scienceqa_adapter_excludes_solutions(tmp_path):
    (tmp_path / "problems.json").write_text(json.dumps({"1": {"question": "Q", "choices": ["yes", "no"], "answer": 1, "image": None, "lecture": "SECRET", "solution": "SECRET"}}))
    (tmp_path / "pid_splits.json").write_text(json.dumps({"train": ["1"]}))
    row = scienceqa(tmp_path, "train")[0]
    assert row.answer == "B" and "SECRET" not in row.prompt()


def test_vqa_adapter(tmp_path):
    from PIL import Image
    Image.new("RGB", (4, 4)).save(tmp_path / "COCO_train2014_000000000009.jpg")
    (tmp_path / "q.json").write_text(json.dumps({"questions": [{"question_id": 1, "image_id": 9, "question": "Q"}]}))
    (tmp_path / "a.json").write_text(json.dumps({"annotations": [{"question_id": 1, "multiple_choice_answer": "yes", "answers": [{"answer": "yes"}]*10}]}))
    row = vqav2(tmp_path / "q.json", tmp_path / "a.json", tmp_path)[0]
    assert row.group == "coco:9" and len(row.answers) == 10
