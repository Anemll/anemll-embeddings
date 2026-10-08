"""Where the runtime looks for its inputs when nothing is passed explicitly.

Standard library only, so the Core AI worker, the scripts, and the export
tools can import it in either virtualenv without pulling in torch.

Lookup order everywhere:

* artifacts: argument, ``$ANEMLL_EMBEDDINGS_ARTIFACTS``, then
  ``<home>/artifacts`` when it holds ``coreai/`` (what
  ``scripts/download_models.py`` writes)
* host model: argument, ``$ANEMLL_EMBEDDINGS_MODEL``, then
  ``<home>/embeddinggemma-2`` when it exists
* Core AI Python: argument, ``$ANEMLL_COREAI_PYTHON``, then the documented
  locations in :func:`coreai_python_candidates`

``<home>`` is ``$ANEMLL_EMBEDDINGS_HOME`` or ``~/.anemll-embeddings``. No
user-specific absolute paths are baked in.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HOME_ENV = "ANEMLL_EMBEDDINGS_HOME"
ARTIFACTS_ENV = "ANEMLL_EMBEDDINGS_ARTIFACTS"
MODEL_ENV = "ANEMLL_EMBEDDINGS_MODEL"
COREAI_PYTHON_ENV = "ANEMLL_COREAI_PYTHON"

SLIM_MODEL_DIRNAME = "embeddinggemma-2"
COREAI_VENV_DIRNAME = "coreai-venv"
_VENV_PYTHON = Path("bin") / "python"
_FORGE_VENV_PYTHON = Path("anemll-forge") / "coreai" / ".venv" / _VENV_PYTHON

COREAI_SETUP_HINT = (
    "Set ANEMLL_COREAI_PYTHON to a Python that can `import coreai.runtime`, "
    "or pass --coreai-python / Embedder(coreai_python=...). Quick setup:\n"
    "  python3.13 -m venv ~/.anemll-embeddings/coreai-venv\n"
    "  ~/.anemll-embeddings/coreai-venv/bin/python -m pip install "
    "'coreai-core==1.0.0b2' numpy\n"
    "  export ANEMLL_COREAI_PYTHON=~/.anemll-embeddings/coreai-venv/bin/python\n"
    "See README.md, section \"Core AI runtime\"."
)


class CoreAIPythonNotFound(RuntimeError):
    """No interpreter with ``coreai.runtime`` could be located."""


def embeddings_home() -> Path:
    raw = os.environ.get(HOME_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".anemll-embeddings"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


def default_artifacts() -> Path | None:
    """``$ANEMLL_EMBEDDINGS_ARTIFACTS``, else ``<home>/artifacts`` if downloaded."""
    env = _env_path(ARTIFACTS_ENV)
    if env is not None:
        return env
    candidate = embeddings_home() / "artifacts"
    return candidate if (candidate / "coreai").is_dir() else None


def default_model() -> Path | None:
    """``$ANEMLL_EMBEDDINGS_MODEL``, else ``<home>/embeddinggemma-2`` if downloaded."""
    env = _env_path(MODEL_ENV)
    if env is not None:
        return env
    candidate = embeddings_home() / SLIM_MODEL_DIRNAME
    return candidate if candidate.is_dir() else None


def coreai_python_candidates() -> tuple[Path, ...]:
    """Documented places to look for a Core AI interpreter, in order.

    1. ``<home>/coreai-venv`` (the README quick setup)
    2. an ``anemll-forge`` checkout next to this repo (``coreai/.venv``)
    3. ``~/anemll-forge/coreai/.venv``
    """
    return (
        embeddings_home() / COREAI_VENV_DIRNAME / _VENV_PYTHON,
        REPO_ROOT.parent / _FORGE_VENV_PYTHON,
        Path.home() / _FORGE_VENV_PYTHON,
    )


def resolve_coreai_python(
    explicit: Path | str | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Return a Core AI interpreter path, or raise :class:`CoreAIPythonNotFound`.

    An explicit path or ``$ANEMLL_COREAI_PYTHON`` must exist; neither falls
    through to the candidates, so a typo is reported instead of silently
    using some other interpreter.
    """
    if explicit is not None and str(explicit):
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        if required:
            raise CoreAIPythonNotFound(f"Core AI Python not found: {path}\n{COREAI_SETUP_HINT}")
        return None
    env = _env_path(COREAI_PYTHON_ENV)
    if env is not None:
        if env.is_file():
            return env
        if required:
            raise CoreAIPythonNotFound(
                f"{COREAI_PYTHON_ENV} is set but is not a file: {env}\n{COREAI_SETUP_HINT}"
            )
        return None
    checked = coreai_python_candidates()
    for path in checked:
        if path.is_file():
            return path
    if required:
        looked = ", ".join(str(path) for path in checked)
        raise CoreAIPythonNotFound(
            f"no Core AI Python found ({COREAI_PYTHON_ENV} is unset; looked in: {looked}).\n"
            f"{COREAI_SETUP_HINT}"
        )
    return None


def current_python_has_coreai() -> bool:
    """True when this interpreter can import ``coreai.runtime``."""
    try:
        return importlib.util.find_spec("coreai.runtime") is not None
    except (ImportError, ValueError):
        return False
