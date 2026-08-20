"""End-to-end check: run the REAL TreeFlowTrainingModule.forward twice -- reference vs fused head --
and compare every loss component and every trainable parameter gradient.

scripts/verify_fused_head.py proves the reduction ALGEBRA in fp64. This proves the WIRING: that the
fused branch feeds the same tensors into the same losses and produces gradients for the same set of
parameters. Deviations here are float32 rounding (~1e-7), which is why it runs at that tolerance and
the algebraic claim is made in the other script.
"""
import os, sys, torch

os.environ["CF_COMPILE"] = "0"          # compile is orthogonal; test the math alone
torch.manual_seed(0)

from chained_flow.training.train_tree_flow import TreeFlowTrainingModule
from chained_flow.drafters.tree_vae_flow import TreeVAEFlowConfig
from chained_flow.training.train_chunked_flow import FlowLossArguments

V, H = 131, 32
class _Cfg:
    hidden_size = H
    vocab_size = V
class _LM(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.lm_head = torch.nn.Linear(H, V, bias=False); s.config = _Cfg()
        s.embed = torch.nn.Embedding(V, H)
    def get_input_embeddings(s): return s.embed
class _Frozen(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.model = _LM()
        for q in s.model.parameters():          # FrozenLMWrapper freezes every param; mirror that
            q.requires_grad_(False)
    def lm_head(s, h): return s.model.lm_head(h)
    @property
    def vocab_size(s): return V

dcfg = TreeVAEFlowConfig(context_size=4, draft_length=4, chunk_size=2, expert_dim=16, num_heads=2,
                         ffn_multiplier=2, num_drafter_layers=1, num_flow_steps=2, markov_rank=8,
                         path_order=4, path_ffn_multiplier=1, cov_b=3, cov_margin=1.0,
                         latent_size=16, vae_intermediate_size=16, vae_num_layers=1,
                         vae_num_heads=2, vae_max_sequence_length=8, train_vae=True,
                         vae_type="transformer_hidden", vae_dir=None, prev_token_cond=True)
loss_cfg = FlowLossArguments()
B, K, C = 3, dcfg.draft_length, dcfg.context_size
ctx  = torch.randn(B, C, H, dtype=torch.float32)
tgt  = torch.randn(B, K, H, dtype=torch.float32)
tok  = torch.randint(0, V, (B, K))
prev = torch.randint(0, V, (B,))

res = {}
for tag, fused in (("reference", "0"), ("fused", "1")):
    torch.manual_seed(1234)
    os.environ["CF_FUSED_HEAD"] = fused; os.environ["CF_FUSED_CHUNK"] = "5"
    m = TreeFlowTrainingModule(_Frozen(), dcfg, loss_cfg).float()
    torch.manual_seed(99)                       # tau is random inside forward
    out = m(ctx, tgt, tok, prev_token=prev)
    out["loss"].backward()
    res[tag] = ({k: v.item() for k, v in out.items() if k.startswith("loss_component")},
                {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
                out["loss"].item())

(rc, rg, rl), (fc, fg, fl) = res["reference"], res["fused"]
ok = abs(rl - fl) < 2e-5
print(f"  total loss   ref={rl:+.12f} fused={fl:+.12f} |d|={abs(rl-fl):.3e}")
for k in sorted(rc):
    d = abs(rc[k] - fc[k]); ok &= d < 2e-5
    print(f"  {k.split('/')[-1]:<26} |d|={d:.3e}")
worst = ("", 0.0)
for n in sorted(rg):
    if n not in fg: print(f"  !! grad missing in fused: {n}"); ok = False; continue
    d = (rg[n] - fg[n]).abs().max().item()
    if d > worst[1]: worst = (n, d)
miss = set(fg) - set(rg)
print(f"  worst param-grad deviation: {worst[1]:.3e}  ({worst[0]})   params compared={len(rg)}")
if miss: print(f"  !! extra grads in fused only: {sorted(miss)}"); ok = False
ok &= worst[1] < 2e-5
print("  ALL COMPONENTS + GRADS MATCH" if ok else "  !! MISMATCH")
sys.exit(0 if ok else 1)
