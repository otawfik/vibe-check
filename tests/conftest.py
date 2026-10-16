import os
import sys

import joblib
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, models  # noqa: E402


@pytest.fixture(scope="session")
def fitted():
    """Full-pipeline models trained on the toy corpus (seconds, offline)."""
    out = {}
    for task in ("sentiment", "emotion"):
        x, y = data.toy(task)
        out[task] = models.build(models.FULL, C=4).fit(x, y)
    return out


@pytest.fixture(scope="session")
def client(tmp_path_factory, fitted):
    path = tmp_path_factory.mktemp("art") / "vibe.joblib"
    joblib.dump(fitted, path)
    os.environ["VIBE_ARTIFACT"] = str(path)
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
