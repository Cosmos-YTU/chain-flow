"""Make multi-token speculative verify numerically identical to sequential single-token decode.

Measured on this backbone: the fla `chunk_gated_delta_rule` kernel (used for any multi-token forward)
and the `fused_recurrent_gated_delta_rule` kernel (used for single-token decode) differ by ~0.125 in
logit space — a real algorithmic gap, not fp rounding — which flips ~0.4% of near-tie argmaxes. So a
multi-token tree verify is NOT bit-exact vs the model's own sequential greedy, breaking losslessness.

Fix: route the Gated-DeltaNet chunk kernel to the RECURRENT kernel for cached continuations (i.e. when
`initial_state is not None`, which is exactly the speculative-verify / decode path). fla's
`fused_recurrent` over T tokens is the same recurrence, in the same order, as T sequential single-token
calls, so verify becomes bit-identical to decode. Prefill (`initial_state=None`) keeps the fast chunk
kernel untouched. Idempotent; returns the number of layers patched.
"""
from __future__ import annotations

from typing import Any


def patch_model_for_lossless_verify(model: Any) -> int:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as rec

    patched = 0
    for layer in model.model.layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None or not hasattr(gdn, "chunk_gated_delta_rule"):
            continue
        if getattr(gdn, "_lossless_patched", False):
            patched += 1
            continue
        orig = gdn.chunk_gated_delta_rule

        def shim(q, k, v, *, g=None, beta=None, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=False, cu_seqlens=None, _orig=orig, **_):
            if initial_state is not None:
                # cached continuation (verify/decode): use recurrent kernel -> matches sequential decode
                return rec(q, k, v, g=g, beta=beta, initial_state=initial_state,
                           output_final_state=output_final_state,
                           use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
            # prefill: keep the fast chunk kernel
            return _orig(q, k, v, g=g, beta=beta, initial_state=initial_state,
                         output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)

        gdn.chunk_gated_delta_rule = shim
        gdn._lossless_patched = True
        patched += 1
    return patched
