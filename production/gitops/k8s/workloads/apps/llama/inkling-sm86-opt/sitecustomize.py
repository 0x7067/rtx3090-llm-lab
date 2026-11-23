"""Auto-install Inkling2 SM86 (RTX 3090) v3 concurrency optimizations lazily in vLLM worker processes."""
import importlib.abc
import os
import sys

_TARGETS = {
    "vllm.models.inkling2.nvidia.ops.attention_sm90",
    "vllm.models.inkling2.nvidia.ops.attention",
    "vllm.models.inkling2.nvidia.attention",
    "vllm.models.inkling2.nvidia.ops.moe_sm90",
    "vllm.models.inkling2.nvidia.ops.linear",
    "vllm.models.inkling2.nvidia.ops.dspark",
    "vllm.models.inkling2.nvidia.dspark",
    "vllm.models.inkling2.speculator",
    "vllm.models.inkling2.nvidia.model",
    "vllm.v1.core.kv_cache_planning",
}


class _InklingSM86PatchLoader(importlib.abc.Loader):
    def __init__(self, orig_loader, fullname: str):
        self._orig = orig_loader
        self._fullname = fullname

    def create_module(self, spec):
        if hasattr(self._orig, "create_module"):
            return self._orig.create_module(spec)
        return None

    def exec_module(self, module):
        self._orig.exec_module(module)
        import inkling_sm86_opt
        inkling_sm86_opt.patch_module(self._fullname, module)


class _InklingSM86Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        idx = sys.meta_path.index(self)
        for finder in sys.meta_path[idx + 1:]:
            if not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _InklingSM86PatchLoader(spec.loader, fullname)
                return spec
        return None


if os.environ.get("INKLING_SM86_OPT", "1") != "0":
    sys.meta_path.insert(0, _InklingSM86Finder())
