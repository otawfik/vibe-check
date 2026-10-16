def test_index_renders_benchmark(client):
    html = client.get("/").get_data(as_text=True)
    assert "Vibe Check" in html and "Published, same test set" in html and "RoBERTa-base" in html


def test_api_check(client):
    r = client.post("/api/check", json={"text": "so happy we won!!"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["vibe"] in {"hype", "chill", "heated", "down", "neutral"}
    assert set(body["emotion"]) == {"anger", "joy", "optimism", "sadness"}


def test_api_check_requires_text(client):
    assert client.post("/api/check", json={"text": ""}).status_code == 400
    assert client.post("/api/check", data="x").status_code == 400


def test_api_metrics(client):
    m = client.get("/api/metrics").get_json()
    assert {"sentiment", "emotion"} <= set(m)
