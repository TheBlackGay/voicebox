"""
Model download helper: HuggingFace first, ModelScope fallback.

HuggingFace Hub is the primary download source: it is the canonical home for
these models and supports the ``HF_TOKEN`` / ``HF_ENDPOINT`` accelerator
environment variables (e.g. ``HF_ENDPOINT=https://hf-mirror.com``). When the
Hub is unreachable (network error, blocked endpoint, timeout) we transparently
fall back to ModelScope (modelscope.cn), which mirrors many of the same repos.

The accelerator variables can be placed in the per-user config file
``~/.voicebox/config.json`` — the app loads it on startup:
    {
      "env": {
        "HF_TOKEN": "hf_xxx",
        "HF_ENDPOINT": "https://hf-mirror.com"
      }
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_acceleration_hint() -> str:
    """Return the download-acceleration hint with the user config file path."""
    from ..config import get_config_file

    return (
        "If HuggingFace is slow or unreachable, put the accelerator variables "
        f"in the user config file {get_config_file()} (an \"env\" map with "
        "HF_TOKEN / HF_ENDPOINT) and restart the app. Downloads automatically "
        "fall back to ModelScope when HuggingFace cannot be reached."
    )


def _huggingface_snapshot_download(repo_id: str, **kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, **kwargs)


def _modelscope_snapshot_download(model_id: str, **kwargs) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError:
        try:
            # modelscope < 1.9 keeps the function under hub.snapshot_download
            from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
        except ImportError:
            raise RuntimeError(
                "ModelScope fallback is not available: install it with "
                "`pip install modelscope` to enable downloading from "
                "modelscope.cn when HuggingFace is unreachable."
            ) from None
    return snapshot_download(model_id, **kwargs)


def download_snapshot(
    repo_id: str,
    *,
    ms_repo_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Download a model snapshot, preferring HuggingFace and falling back to ModelScope.

    Args:
        repo_id: HuggingFace repo id (e.g. ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``).
        ms_repo_id: Optional ModelScope repo id; defaults to ``repo_id``.
        **kwargs: forwarded to :func:`huggingface_hub.snapshot_download`. The
            ``ignore_patterns`` value is mapped to ModelScope's
            ``ignore_file_pattern`` when falling back.

    Returns:
        Local directory containing the downloaded snapshot.

    Raises:
        RuntimeError: when both sources fail, with a hint about accelerator
            environment variables.
    """
    hf_error: Optional[Exception] = None
    try:
        logger.info("Downloading %s from HuggingFace Hub...", repo_id)
        return _huggingface_snapshot_download(repo_id, **kwargs)
    except Exception as e:  # noqa: BLE001 - fall back for any hub failure
        hf_error = e
        logger.warning("HuggingFace download failed (%s); trying ModelScope...", e)

    ms_repo = ms_repo_id or repo_id
    try:
        ms_kwargs = dict(kwargs)
        if "ignore_patterns" in ms_kwargs:
            ms_kwargs["ignore_file_pattern"] = ms_kwargs.pop("ignore_patterns")
        logger.info("Downloading %s from ModelScope...", ms_repo)
        return _modelscope_snapshot_download(ms_repo, **ms_kwargs)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download '{repo_id}' from HuggingFace ({hf_error}) "
            f"and the ModelScope fallback also failed ({e}). {get_acceleration_hint()}"
        ) from e


def is_modelscope_cached(
    model_id: str,
    *,
    required_files: Optional[list[str]] = None,
) -> bool:
    """Check whether a ModelScope snapshot exists in the default cache dir.

    ModelScope stores snapshots under
    ``~/.cache/modelscope/hub/models/<org>/<name>/<revision>``.
    """
    model_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_id
    if not model_dir.exists():
        return False
    if required_files:
        return all(any(model_dir.rglob(fname)) for fname in required_files)
    return any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.pt"))
