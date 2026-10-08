"""Core AI backend — thin re-export of the ``api`` runtime."""

from api.embedder import (
    CoreAIBackend,
    CoreAIWorkerClient,
    apply_compute_env,
    default_artifacts,
    default_coreai_python,
    default_model,
    json_line,
    json_loads,
    package_paths,
)

__all__ = [
    "CoreAIBackend",
    "CoreAIWorkerClient",
    "apply_compute_env",
    "default_artifacts",
    "default_coreai_python",
    "default_model",
    "json_line",
    "json_loads",
    "package_paths",
]
