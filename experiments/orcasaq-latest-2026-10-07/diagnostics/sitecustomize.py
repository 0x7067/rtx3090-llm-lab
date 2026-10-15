"""Startup-only KV geometry diagnostic for isolated Orca trials.

This module changes no constructor arguments, return values or request paths.
It patches only after the target module has finished executing.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import sys

_TARGET = "vllm.v1.core.kv_cache_manager"
_SCHEDULER_TARGET = "vllm.v1.core.sched.scheduler"


def _log(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass


def _emit(manager):
    coordinator = manager.coordinator
    config = manager.kv_cache_config
    groups = []
    for index, group in enumerate(config.kv_cache_groups):
        spec = group.kv_cache_spec
        groups.append({
            "index": index,
            "spec_class": type(spec).__name__,
            "block_size": getattr(spec, "block_size", None),
            "is_eagle_group": getattr(group, "is_eagle_group", None),
            "prefix_cacheable": getattr(spec, "prefix_cacheable", None),
            "mamba_cache_mode": getattr(spec, "mamba_cache_mode", None),
            "num_prefill_checkpoint_blocks": getattr(
                spec, "num_prefill_checkpoint_blocks", None),
        })
    payload = {
        "groups": groups,
        "coordinator_class": type(coordinator).__name__,
        "scheduler_block_size": getattr(coordinator, "scheduler_block_size", None),
        "hash_block_size": getattr(coordinator, "hash_block_size", None),
        "eagle_group_ids": sorted(getattr(coordinator, "eagle_group_ids", set())),
        "enable_partial_hash_hits": getattr(coordinator, "enable_partial_hash_hits", None),
        "num_reprefillable_tokens": getattr(coordinator, "num_reprefillable_tokens", None),
        "mamba_shared_prefix_checkpoint": getattr(
            manager, "mamba_shared_prefix_checkpoint", None),
        "managers": [{
            "class": type(item).__name__,
            "block_size": getattr(item, "block_size", None),
            "use_eagle": getattr(item, "use_eagle", None),
            "shared_prefix_checkpoint": getattr(item, "shared_prefix_checkpoint", None),
            "cache_hit_alignment_tokens": getattr(item, "cache_hit_alignment_tokens", None),
        } for item in getattr(coordinator, "single_type_managers", ())],
    }
    _log("KV_GROUP_DIAGNOSTIC " + json.dumps(payload, sort_keys=True))


def _patch(module):
    is_scheduler = module.__name__ == _SCHEDULER_TARGET
    cls = getattr(module, "Scheduler" if is_scheduler else "KVCacheManager", None)
    if cls is None or getattr(cls, "_orca_startup_diagnostic", False):
        return
    original = cls.__init__

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            if is_scheduler:
                manager = getattr(self, "kv_cache_manager", None)
                payload = {
                    "block_size": getattr(self, "block_size", None),
                    "hash_block_size": getattr(self, "hash_block_size", None),
                    "mamba_partial_cache_hit": getattr(self, "mamba_partial_cache_hit", None),
                    "mamba_shared_prefix_checkpoint": getattr(
                        self, "mamba_shared_prefix_checkpoint", None),
                    "manager_mamba_shared_prefix_checkpoint": getattr(
                        manager, "mamba_shared_prefix_checkpoint", None),
                }
                _log("SCHEDULER_CHECKPOINT_DIAGNOSTIC " + json.dumps(payload, sort_keys=True))
            else:
                _emit(self)
        except Exception as error:
            _log("KV_GROUP_DIAGNOSTIC_ERROR " + type(error).__name__)
        return result

    cls.__init__ = wrapped
    cls._orca_startup_diagnostic = True


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        create = getattr(self.delegate, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        try:
            _patch(module)
        except Exception as error:
            _log("KV_GROUP_DIAGNOSTIC_PATCH_ERROR " + type(error).__name__)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (_TARGET, _SCHEDULER_TARGET):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


sys.meta_path.insert(0, _Finder())
