"""Chained-Flow.

Importing the package APPLIES THE CAPABILITY-GATED FLAG DEFAULTS (``chain_flow.defaults``)
before anything else.  That has to happen at import time, not at proposer construction: the
forked vLLM reads ``CF_TREE_FULLCG`` in ``GPUModelRunner.__init__`` a few hundred lines AFTER it
instantiates our ``custom_class`` proposer (which is what imports us), so import time is the
last moment at which those defaults can still be seen.  ``defaults.apply()`` is idempotent and
never overrides an explicitly set ``CF_*``.

THE PUBLIC NAMES ARE LAZY (PEP 562).  ``chain_flow.vllm_plugin.async_guard`` is a
``vllm.general_plugins`` entry point, so this module is imported inside
``EngineArgs.__post_init__`` of EVERY vLLM process -- including plain base-model runs that will
never touch a drafter.  Eagerly importing ``context`` / ``frozen_lm`` / ``generation`` there
would drag ``torch`` + ``transformers`` into that path for nothing, and -- worse -- vLLM's
``load_general_plugins`` SWALLOWS exceptions raised while loading an entry point, so any import
error in that chain would turn the plugin into a silent no-op.  ``defaults`` is stdlib-only, so
what is left at import time costs ~1 ms and cannot fail for an unrelated reason.
"""
from typing import TYPE_CHECKING

from chain_flow import defaults as _cf_defaults

_cf_defaults.apply()

__all__ = [
    "ChainedFlowContext",
    "FrozenLMWrapper",
    "GenerationResult",
    "LMState",
    "generate_with_drafter",
]

_LAZY = {
    "ChainedFlowContext": "chain_flow.context",
    "FrozenLMWrapper": "chain_flow.frozen_lm",
    "LMState": "chain_flow.frozen_lm",
    "GenerationResult": "chain_flow.generation",
    "generate_with_drafter": "chain_flow.generation",
}

if TYPE_CHECKING:                                   # pragma: no cover - type checkers only
    from chain_flow.context import ChainedFlowContext
    from chain_flow.frozen_lm import FrozenLMWrapper, LMState
    from chain_flow.generation import GenerationResult, generate_with_drafter


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value                          # subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
