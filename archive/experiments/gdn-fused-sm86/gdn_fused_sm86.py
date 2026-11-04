"""Opt-in loader and call path for the private sm86 GDN MTP operator.

The wrapper does not import or patch vLLM.  A caller can use
``run_if_enabled`` at the same point where vLLM would choose its Triton GDN
decode path.  False means that the caller should keep using Triton.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


GATE_ENV = "VLLM_ENABLE_FUSED_GDN_SM86"
LIBRARY_ENV = "GDN_FUSED_SM86_LIBRARY"
_NAMESPACE = "gdn_fused_sm86"
_OP_NAME = "decode_post_conv_mtp"
_loaded_library: str | None = None


def enabled() -> bool:
    """Return whether this experiment was explicitly selected."""

    return os.environ.get(GATE_ENV, "").strip().lower() in {
        "1",
        "true",
        "on",
        "sm86",
        "cuda-sm86",
    }


def _library_path() -> Path:
    configured = os.environ.get(LIBRARY_ENV)
    if configured:
        return Path(configured).expanduser()
    candidates = sorted(Path(__file__).parent.glob("gdn_fused_sm86*.so"))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"set {LIBRARY_ENV} to the built gdn_fused_sm86 shared library"
    )


def load_library(path: str | os.PathLike[str] | None = None) -> None:
    """Load the standalone extension once into the PyTorch dispatcher."""

    global _loaded_library
    if path is None and _loaded_library is not None:
        return
    library = Path(path).expanduser() if path is not None else _library_path()
    library = library.resolve()
    if not library.is_file():
        raise FileNotFoundError(f"GDN fused extension not found: {library}")
    if _loaded_library == str(library):
        return
    torch.ops.load_library(str(library))
    _loaded_library = str(library)


def available() -> bool:
    """Check the opt-in gate, exact SM86 target, and dispatcher registration."""

    if not enabled() or not torch.cuda.is_available():
        return False
    if tuple(torch.cuda.get_device_capability()) != (8, 6):
        return False
    try:
        load_library()
    except (FileNotFoundError, RuntimeError, OSError):
        return False
    return hasattr(getattr(torch.ops, _NAMESPACE), _OP_NAME)


def fused_gdn_decode_post_conv_mtp(*args: Any, **kwargs: Any) -> None:
    """Run the fused op after the caller has enabled and loaded this variant."""

    if not available():
        raise RuntimeError(
            "gdn_fused_sm86 is unavailable; use the Triton GDN decode path"
        )
    getattr(getattr(torch.ops, _NAMESPACE), _OP_NAME)(*args, **kwargs)


def run_if_enabled(*args: Any, **kwargs: Any) -> bool:
    """Run the experiment when selected and return whether it handled the call."""

    if not available():
        return False
    fused_gdn_decode_post_conv_mtp(*args, **kwargs)
    return True
