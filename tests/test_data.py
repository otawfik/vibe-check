from src.data import VIBE_LABELS, generate_dataset, train_test_split


def test_generate_balanced():
    texts, labels = generate_dataset(n_per_class=20, seed=7)
    assert len(texts) == 20 * len(VIBE_LABELS)
    for lab in VIBE_LABELS:
        assert labels.count(lab) == 20


def test_generate_reproducible():
    a = generate_dataset(n_per_class=5, seed=7)
    b = generate_dataset(n_per_class=5, seed=7)
    assert a == b


def test_split_sizes():
    texts, labels = generate_dataset(n_per_class=20, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(texts, labels)
    assert len(Xtr) + len(Xte) == len(texts)
    assert len(ytr) == len(Xtr)
