"""Local research API: requests, displays, and responses are separate events."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feedback_loop_engine.engine import Store, rank


class RecommendInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    k: int = Field(default=10, ge=1, le=50)


class ImpressionsInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1)
    video_ids: list[str] = Field(min_length=1, max_length=50)


class FeedbackInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    action: Literal["click", "like", "skip"]


def create_app(db_path: str | Path | None = None) -> FastAPI:
    store = Store(db_path or os.environ.get("FEEDBACK_LOOP_DB", "feedback_loop.sqlite3"))
    app = FastAPI(title="Feedback Loop Engine — Phase 1", version="0.1.0")
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/videos")
    def videos() -> list[dict]:
        return [vars(video) for video in store.videos()]

    @app.post("/recommendations")
    def recommendations(body: RecommendInput) -> dict:
        items = rank(store.videos(), store.events(), body.user_id, k=body.k)
        request_id = store.create_batch(body.user_id, items)
        return {"request_id": request_id, "items": [item.to_dict() for item in items]}

    @app.post("/impressions")
    def impressions(body: ImpressionsInput) -> dict:
        try:
            count = store.record_impressions(body.request_id, body.user_id, body.video_ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"recorded": count}

    @app.post("/feedback")
    def feedback(body: FeedbackInput) -> dict:
        try:
            recorded = store.record_feedback(
                body.request_id, body.user_id, body.video_id, body.action
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"recorded": recorded}

    return app


app = create_app()
