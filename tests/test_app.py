import pytest

pytest.importorskip("flask")
from app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_index(client):
    assert client.get("/").status_code == 200


def test_api_check(client):
    r = client.post("/api/check", json={"text": "i love this amazing day"})
    assert r.status_code == 200
    body = r.get_json()
    assert "vibe" in body and "vibe_score" in body


def test_api_check_empty(client):
    assert client.post("/api/check", json={"text": ""}).status_code == 400
