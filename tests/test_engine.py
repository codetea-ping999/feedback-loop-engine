import pytest

from feedback_loop_engine.demo import seed
from feedback_loop_engine.engine import Event, Store, Video, evaluate_temporal, rank


def test_cold_start_diversifies_and_is_deterministic():
    videos = [
        Video("a1", "AI one", "ai", 100),
        Video("a2", "AI two", "ai", 100),
        Video("a3", "AI three", "ai", 100),
        Video("b1", "Car one", "cars", 100),
    ]
    first = rank(videos, [], "unknown", k=3, now=100, max_per_category=2)
    assert [item.video_id for item in first] == ["a1", "a2", "b1"]
    assert first == rank(videos, [], "unknown", k=3, now=100, max_per_category=2)


def test_feedback_changes_ranking_and_prevents_recommending_consumed():
    videos = [
        Video("ai-1", "AI first", "ai", 100),
        Video("ai-2", "AI second", "ai", 100),
        Video("car-1", "Car first", "cars", 100),
    ]
    events = [Event("alice", "ai-1", "click", 101)]
    ranked = rank(videos, events, "alice", k=3, now=102)
    assert ranked[0].video_id == "ai-2"
    assert "ai-1" not in [item.video_id for item in ranked]
    assert rank(videos, [Event("alice", "ai-1", "impression", 101)], "alice", now=102)[
        0
    ].video_id == "ai-1"


def test_exposure_required_identity_validation_and_idempotency(tmp_path):
    store = Store(tmp_path / "events.sqlite3")
    store.put_videos([Video("v1", "Example", "ai", 100)])
    batch = store.create_batch("alice", rank(store.videos(), [], "alice"))
    with pytest.raises(ValueError, match="matching recorded impression"):
        store.record_feedback(batch, "alice", "v1", "click")
    with pytest.raises(ValueError, match="for this user"):
        store.record_impressions(batch, "mallory", ["v1"])
    assert store.record_impressions(batch, "alice", ["v1"]) == 1
    assert store.record_impressions(batch, "alice", ["v1"]) == 0
    assert store.record_feedback(batch, "alice", "v1", "click") is True
    assert store.record_feedback(batch, "alice", "v1", "click") is False
    with pytest.raises(ValueError, match="Conflicting"):
        store.record_feedback(batch, "alice", "v1", "skip")
    assert [event.action for event in store.events()] == ["impression", "click"]


def test_invalid_batch_items_rejected_atomically(tmp_path):
    store = Store(tmp_path / "events.sqlite3")
    store.put_videos([Video("v1", "Example", "ai", 100)])
    batch = store.create_batch("alice", rank(store.videos(), [], "alice"))
    with pytest.raises(ValueError, match="recommended batch"):
        store.record_impressions(batch, "alice", ["v1", "nonexistent"])
    assert not store.events()


def test_synthetic_seed_and_temporal_evaluation(tmp_path):
    store = Store(tmp_path / "events.sqlite3")
    assert seed(store) == 36
    assert seed(store) == 0
    result = evaluate_temporal(store.videos(), store.events(), k=5)
    assert result["train_events"] > 0
    assert result["test_positive_events"] > 0
    assert 0 <= result["recall_at_k"] <= 1
    assert 0 <= result["ndcg_at_k"] <= 1
    assert 0 <= result["catalog_coverage_at_k"] <= 1
    assert result["evaluated_users"] > 0


def test_evaluation_rejects_insufficient_data():
    with pytest.raises(ValueError, match="At least five"):
        evaluate_temporal([], [])
