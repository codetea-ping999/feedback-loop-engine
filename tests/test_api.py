from fastapi.testclient import TestClient

from feedback_loop_engine.api import create_app
from feedback_loop_engine.demo import seed


def test_api_round_trip(tmp_path):
    app = create_app(tmp_path / "api.sqlite3")
    seed(app.state.store)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/recommendations", json={"user_id": "new-user", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 3
    first = body["items"][0]["video_id"]
    feedback = {"request_id": body["request_id"], "user_id": "new-user", "video_id": first}
    assert client.post("/feedback", json={**feedback, "action": "like"}).status_code == 400
    assert client.post(
        "/impressions", json={"request_id": body["request_id"], "user_id": "new-user", "video_ids": [first]}
    ).json() == {"recorded": 1}
    assert client.post("/feedback", json={**feedback, "action": "like"}).json() == {"recorded": True}
    assert client.post("/feedback", json={**feedback, "action": "like"}).json() == {"recorded": False}
    later = client.post("/recommendations", json={"user_id": "new-user", "k": 10}).json()
    assert first not in [item["video_id"] for item in later["items"]]


def test_input_validation(tmp_path):
    client = TestClient(create_app(tmp_path / "blank.sqlite3"))
    assert client.post("/recommendations", json={"user_id": "", "k": 5}).status_code == 422
    assert client.post("/recommendations", json={"user_id": "ok", "k": 0}).status_code == 422
    assert client.post(
        "/feedback", json={"user_id": "ok", "request_id": "bad", "video_id": "v", "action": "other"}
    ).status_code == 422
