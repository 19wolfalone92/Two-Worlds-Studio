"""Keep each Kaggle CLI subprocess independent of every other credential."""
from __future__ import annotations
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def isolated_environment(token: str, config_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("KAGGLE_") or
                   any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")))}
    env.update(KAGGLE_API_TOKEN=token, KAGGLE_CONFIG_DIR=str(config_dir))
    return env


@contextmanager
def token_environment(token: str):
    with tempfile.TemporaryDirectory(prefix="eb01-cli-auth-",
                                     dir=os.environ.get("RUNNER_TEMP")) as config:
        yield isolated_environment(token, Path(config))
