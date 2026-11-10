"""Reproducible synthetic dataset, API seed command, and evaluation CLI."""

from __future__ import annotations

import argparse
import json
import time

from feedback_loop_engine.engine import Recommendation, Store, Video, evaluate_temporal, rank

CATEGORIES = ("ai", "cars", "music", "science", "games", "sports")


def seed(store: Store) -> int:
    """Seed 36 fictional videos and 48 exposures with corresponding actions once."""
    if store.videos():
        return 0
    now = time.time()
    videos = [
        Video(f"v-{group}-{i}", f"Sample {group} video {i}", group, now - i * 86400)
        for group in CATEGORIES
        for i in range(1, 7)
    ]
    store.put_videos(videos)
    for step in range(8):
        for user_index in range(6):
            user_id = f"user-{user_index}"
            category = CATEGORIES[user_index]
            chosen = videos[(user_index * 6 + step) % len(videos)]
            # Steps 6 and 7 visit another category to provide holdout diversity.
            if step < 6:
                chosen = videos[user_index * 6 + step]
            item = Recommendation(chosen.id, chosen.title, chosen.category, 0, "synthetic")
            request_id = store.create_batch(user_id, [item])
            store.record_impressions(request_id, user_id, [chosen.id])
            store.record_feedback(request_id, user_id, chosen.id, "click" if step < 7 else "skip")
    return len(videos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback Loop Engine demo")
    parser.add_argument("--db", default="feedback_loop.sqlite3")
    parser.add_argument("command", choices=["seed", "evaluate", "recommend"])
    parser.add_argument("--user", default="user-0")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    store = Store(args.db)
    if args.command == "seed":
        print(json.dumps({"inserted_videos": seed(store)}, indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_temporal(store.videos(), store.events(), args.k), indent=2))
    else:
        print(
            json.dumps(
                [r.to_dict() for r in rank(store.videos(), store.events(), args.user, args.k)],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
