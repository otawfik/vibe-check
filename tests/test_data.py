import pytest

from src import data


@pytest.mark.parametrize("task", ["sentiment", "emotion"])
def test_toy_is_balanced_and_labelled(task):
    x, y = data.toy(task, n_per_class=10)
    assert len(x) == len(y) == 10 * len(data.LABELS[task])
    assert set(y) == set(data.LABELS[task])


def test_toy_reproducible():
    assert data.toy("sentiment", seed=3) == data.toy("sentiment", seed=3)


def test_unknown_task():
    with pytest.raises(KeyError):
        data.load_tweeteval("irony", "train")


def test_load_tweeteval_reads_cache(tmp_path):
    for name, body in [("train_text.txt", "good day\nbad day\n"), ("train_labels.txt", "2\n0\n")]:
        (tmp_path / f"sentiment_{name}").write_text(body, encoding="utf-8")
    x, y = data.load_tweeteval("sentiment", "train", cache_dir=str(tmp_path))
    assert x == ["good day", "bad day"] and y == ["positive", "negative"]


def test_load_tweeteval_rejects_misaligned_files(tmp_path):
    (tmp_path / "emotion_val_text.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "emotion_val_labels.txt").write_text("1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_tweeteval("emotion", "val", cache_dir=str(tmp_path))
