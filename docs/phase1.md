# Phase 1: recommendation MVP and event logging

## Data semantics

1. A recommendation request creates a `batch` and ordered `batch_items`. **This is not exposure**.
2. A client reports only videos that actually appeared via `/impressions`. Duplicates do not inflate metrics.
3. A `click`, `like` or `skip` must reference a recorded impression from the same user and batch. `click`/`like` conflicts with `skip` for one impression.
4. Only explicit feedback adjusts interests. Impressions are retained for later position/exposure-bias analysis but do not imply preferences.

SQLite tables: `videos`, `batches`, `batch_items`, `events`. Foreign keys are on and uniqueness constraints support retry-safe writes. Database files are excluded from git.

## Baseline and evaluation

Ranking: positive/negative category affinities + weighted global interactions + publication freshness; a category cap encourages variety and underfilled rankings are backfilled. Scores are deliberately interpretable, not probabilities. Consumed content is excluded.

Evaluation: one global chronological 80/20 split on positive events. Only the training event log enters the ranker. Holdout already-seen items are removed; mean per-user Recall@K and NDCG@K plus catalog coverage are reported. There is no shuffle leakage, but logging-policy/exposure bias remains. Dataset is synthetic; numbers are not comparable to production recommender benchmarks.

## Acceptance criteria

- [x] `python -m feedback_loop_engine.demo seed` populates a repeatable local demo dataset.
- [x] API supports recommendation request, separately recorded impression and validated feedback.
- [x] Ranking responds to interests and excludes consumed videos.
- [x] Offline evaluation works on a strict historical split.
- [x] Tests exercise cold-start, ranking, idempotency, identity and input validation.
- [x] GitHub Actions runs lint, tests and evaluation across supported Python versions.

## Planned next steps

1. Make a MovieLens (or similarly licensed) import pipeline with schema validation, provenance and privacy-safe IDs.
2. Train a Two-Tower model and compare Recall@K and runtime against the baseline.
3. Add logged propensities, IPS/SNIPS offline evaluation and explicit exploration policies before interpreting feedback-loop effects causally.
4. Add dashboard and drift / category-concentration measures for iterative simulation.
