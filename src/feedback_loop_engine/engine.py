"""SQLite event store, baseline ranking, and offline temporal evaluation."""

from __future__ import annotations

import math
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    category: str
    published_at: float


@dataclass(frozen=True)
class Event:
    user_id: str
    video_id: str
    action: str
    occurred_at: float


@dataclass(frozen=True)
class Recommendation:
    video_id: str
    title: str
    category: str
    score: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class Store:
    """A fresh SQLite connection per operation keeps API requests thread-safe."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    published_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_items (
                    request_id TEXT NOT NULL REFERENCES batches(request_id),
                    video_id TEXT NOT NULL REFERENCES videos(id),
                    position INTEGER NOT NULL CHECK(position >= 0),
                    PRIMARY KEY(request_id, video_id),
                    UNIQUE(request_id, position)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('impression','click','like','skip')),
                    occurred_at REAL NOT NULL,
                    UNIQUE(request_id, video_id, action),
                    FOREIGN KEY(request_id, video_id)
                        REFERENCES batch_items(request_id, video_id)
                );
                CREATE INDEX IF NOT EXISTS ix_events_time ON events(occurred_at);
                CREATE INDEX IF NOT EXISTS ix_events_user ON events(user_id);
                """
            )

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=10)
        try:
            con.execute("PRAGMA foreign_keys = ON")
            con.row_factory = sqlite3.Row
            with con:
                yield con
        finally:
            con.close()

    def put_videos(self, videos: list[Video]) -> None:
        with self._connect() as con:
            con.executemany(
                "INSERT OR IGNORE INTO videos VALUES (?, ?, ?, ?)",
                [(v.id, v.title, v.category, v.published_at) for v in videos],
            )

    def videos(self) -> list[Video]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM videos ORDER BY id").fetchall()
        return [Video(**dict(row)) for row in rows]

    def events(self) -> list[Event]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT user_id, video_id, action, occurred_at FROM events ORDER BY occurred_at, id"
            ).fetchall()
        return [Event(**dict(row)) for row in rows]

    def create_batch(self, user_id: str, items: list[Recommendation]) -> str:
        request_id = str(uuid.uuid4())
        now = time.time()
        with self._connect() as con:
            con.execute("INSERT INTO batches VALUES (?, ?, ?)", (request_id, user_id, now))
            con.executemany(
                "INSERT INTO batch_items VALUES (?, ?, ?)",
                [(request_id, item.video_id, position) for position, item in enumerate(items)],
            )
        return request_id

    def record_impressions(self, request_id: str, user_id: str, video_ids: list[str]) -> int:
        if not video_ids or len(video_ids) != len(set(video_ids)):
            raise ValueError("video_ids must be a non-empty list of unique video IDs")
        with self._connect() as con:
            batch = con.execute(
                "SELECT user_id FROM batches WHERE request_id=?", (request_id,)
            ).fetchone()
            if batch is None or batch["user_id"] != user_id:
                raise ValueError("Unknown recommendation batch for this user")
            allowed = {
                row[0]
                for row in con.execute(
                    "SELECT video_id FROM batch_items WHERE request_id=?", (request_id,)
                )
            }
            if not set(video_ids).issubset(allowed):
                raise ValueError("An impression must belong to the recommended batch")
            inserted = 0
            for video_id in video_ids:
                result = con.execute(
                    """INSERT OR IGNORE INTO events
                    (request_id, user_id, video_id, action, occurred_at)
                    VALUES (?, ?, ?, 'impression', ?)""",
                    (request_id, user_id, video_id, time.time()),
                )
                inserted += result.rowcount
            return inserted

    def record_feedback(self, request_id: str, user_id: str, video_id: str, action: str) -> bool:
        if action not in {"click", "like", "skip"}:
            raise ValueError("Action must be click, like, or skip")
        with self._connect() as con:
            impression = con.execute(
                """SELECT 1 FROM events WHERE request_id=? AND user_id=?
                AND video_id=? AND action='impression'""",
                (request_id, user_id, video_id),
            ).fetchone()
            if impression is None:
                raise ValueError("Feedback requires a matching recorded impression")
            conflicting = "skip" if action in {"click", "like"} else "click"
            conflicts = {conflicting}
            if action == "skip":
                conflicts.add("like")
            if any(
                con.execute(
                    """SELECT 1 FROM events WHERE request_id=? AND video_id=? AND action=?""",
                    (request_id, video_id, other),
                ).fetchone()
                for other in conflicts
            ):
                raise ValueError("Conflicting feedback for this impression")
            result = con.execute(
                """INSERT OR IGNORE INTO events
                (request_id, user_id, video_id, action, occurred_at) VALUES (?, ?, ?, ?, ?)""",
                (request_id, user_id, video_id, action, time.time()),
            )
            return result.rowcount == 1


def rank(
    videos: list[Video],
    events: list[Event],
    user_id: str,
    k: int = 10,
    now: float | None = None,
    max_per_category: int = 2,
) -> list[Recommendation]:
    """Explainable baseline: preference + globally observed actions + freshness.

    Explicit actions affect interests; impressions alone do not imply interest.
    A deterministic category cap is used as a diversification first pass.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if now is None:
        now = time.time()
    prefs: Counter[str] = Counter()
    popularity: Counter[str] = Counter()
    consumed: set[str] = set()
    category_of = {video.id: video.category for video in videos}
    for event in events:
        if event.video_id not in category_of:
            continue
        weight = {"click": 2, "like": 3, "skip": -2}.get(event.action, 0)
        if weight > 0:
            popularity[event.video_id] += weight
        if event.user_id == user_id and weight:
            consumed.add(event.video_id)
            prefs[category_of[event.video_id]] += weight
    candidates: list[Recommendation] = []
    for video in videos:
        if video.id in consumed:
            continue
        age_days = max(0, (now - video.published_at) / 86400)
        freshness = 1 / (1 + age_days / 30)
        score = prefs[video.category] + 0.35 * math.log1p(popularity[video.id]) + freshness
        reason = "interest" if prefs[video.category] > 0 else "discovery"
        candidates.append(Recommendation(video.id, video.title, video.category, round(score, 6), reason))
    candidates.sort(key=lambda item: (-item.score, item.video_id))
    results: list[Recommendation] = []
    counts: Counter[str] = Counter()
    for candidate in candidates:
        if counts[candidate.category] < max_per_category and len(results) < k:
            results.append(candidate)
            counts[candidate.category] += 1
    if len(results) < k:
        chosen = {item.video_id for item in results}
        results.extend(item for item in candidates if item.video_id not in chosen)  # backfill
    return results[:k]


def evaluate_temporal(videos: list[Video], events: list[Event], k: int = 5) -> dict:
    """Single global chronological split; use training events only for scoring.

    Holdout positives are observed interactions, not unbiased relevance labels.
    """
    positives = sorted(
        (event for event in events if event.action in {"click", "like"}),
        key=lambda event: event.occurred_at,
    )
    if len(positives) < 5:
        raise ValueError("At least five positive events are required for evaluation")
    cutoff = positives[max(0, int(len(positives) * 0.8) - 1)].occurred_at
    train = [event for event in events if event.occurred_at <= cutoff]
    test = [event for event in positives if event.occurred_at > cutoff]
    seen = defaultdict(set)
    for event in train:
        if event.action in {"click", "like", "skip"}:
            seen[event.user_id].add(event.video_id)
    relevant: dict[str, set[str]] = defaultdict(set)
    for event in test:
        if event.video_id not in seen[event.user_id]:
            relevant[event.user_id].add(event.video_id)
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended: set[str] = set()
    for user_id, truth in sorted(relevant.items()):
        if not truth:
            continue
        predicted = rank(videos, train, user_id, k=k, now=cutoff)
        ids = [item.video_id for item in predicted]
        recommended.update(ids)
        recalls.append(len(set(ids) & truth) / len(truth))
        dcg = sum(1 / math.log2(position + 2) for position, video_id in enumerate(ids) if video_id in truth)
        idcg = sum(1 / math.log2(position + 2) for position in range(min(k, len(truth))))
        ndcgs.append(dcg / idcg if idcg else 0)
    if not recalls:
        raise ValueError("No unseen positive items remain in the holdout")
    return {
        "cutoff_timestamp": cutoff,
        "train_events": len(train),
        "test_positive_events": len(test),
        "evaluated_users": len(recalls),
        "recall_at_k": round(sum(recalls) / len(recalls), 6),
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 6),
        "catalog_coverage_at_k": round(len(recommended) / len(videos), 6) if videos else 0,
        "k": k,
    }
