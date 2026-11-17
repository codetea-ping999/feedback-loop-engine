# feedback-loop-engine

A research-oriented recommendation baseline exploring personalized ranking, **exposure-aware user feedback loops**, and honest offline evaluation.

## Phase 1 — runnable MVP

- **Candidate ranking:** reproducible content-category preferences, global popularity and recency, with a diversity cap and deterministic tie-breaking.
- **Exposure-aware logging:** recommendation requests are **not** counted as impressions. Record only items actually displayed, then clicks/likes/skips. SQLite persists requests, positions, impressions and feedback; duplicates are idempotent.
- **Evaluation:** chronological global split with Recall@K, NDCG@K and catalog coverage. The synthetic dataset is for smoke tests, **not** a claim of real-world recommendation quality.
- **CI:** pytest and Ruff on Python 3.11, 3.12 and 3.13.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
python -m feedback_loop_engine.demo seed
python -m feedback_loop_engine.demo evaluate
uvicorn feedback_loop_engine.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the API. The database is local `feedback_loop.sqlite3` by default; use `FEEDBACK_LOOP_DB` to override it. No external data, account or API key is needed.

## Recommendation → impression → feedback

```bash
# 1. Request a ranked batch. Copy its request_id and a video_id.
curl -sS -X POST http://127.0.0.1:8000/recommendations \\
  -H 'Content-Type: application/json' -d '{"user_id":"alice","k":5}'

# 2. When the UI has actually rendered the item, submit an impression.
curl -sS -X POST http://127.0.0.1:8000/impressions \\
  -H 'Content-Type: application/json' \\
  -d '{"user_id":"alice","request_id":"<request_id>","video_ids":["<video_id>"]}'

# 3. Record an observed action for that same impression.
curl -sS -X POST http://127.0.0.1:8000/feedback \\
  -H 'Content-Type: application/json' \\
  -d '{"user_id":"alice","request_id":"<request_id>","video_id":"<video_id>","action":"click"}'
```

Use `python -m feedback_loop_engine.demo recommend --user user-0` to inspect the baseline without serving HTTP.

## Quality checks

```bash
ruff check .
pytest -q
```

## Research limitations

- **Offline labels are observational.** Interactions are biased by what was previously shown; Recall/NDCG are diagnostic, not causal estimates of satisfaction. No IPS or online A/B test yet.
- **No learned neural model yet.** Two-Tower, FAISS and multi-task ranking are subsequent phases. Feedback updates rankings on the next request without retraining a model.
- **Local development only.** The API has no authentication, rate limiting, retention controls or privacy protections necessary for internet deployment. Use pseudonymous IDs and never upload real user activity without appropriate consent and protections.
- **Cold-start and sparse feedback** still limit accuracy. The synthetic data intentionally contains narrow interests; use an independently sourced dataset for meaningful research.

See [Phase 1 design and acceptance criteria](docs/phase1.md).
