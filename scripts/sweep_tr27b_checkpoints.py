"""Accept-vs-training-duration curve across the saved Turkish checkpoints.

ESTIMATOR: offline, per-token-POSITION (diff_plugin_vs_harness.py plugin arm). NOT comparable to
any served/in-engine number, which averages run length over draft STEPS -- a ~-0.40 difference on
4B Turkish, from the denominator alone. Use this curve only to compare checkpoints WITH EACH OTHER,
which is what it is for; never lift a value out of it and set it beside a tok/s figure.

Why: the 4B Turkish run reached Turkish mean 3.40 (from 1.58) in 2 EPOCHS on 7.12M tokens, ending
above its own English mean, with English essentially flat (-0.05). This 27B run is budgeted at 6
epochs on 13.2M tokens -- ~5.6x that adaptation -- so 6 epochs is an upper bound, not a target. The
stopping point is an empirical question and this produces the evidence to answer it.

`save_steps: 400` with 2401 total steps means one checkpoint per epoch, so the sweep is per-epoch.

Two things this handles that a naive loop would not:

  * HF Trainer writes `model.safetensors` into `checkpoint-N/` but NOT
    `chained_flow_tree_config.json` -- that is only written by the custom save at the end. Every
    eval would fail on a missing config, so it is copied in (architecture is identical across
    checkpoints of one run).
  * English regression from Turkish-only fine-tuning is NOT uniform. The 4B run lost accept exactly
    where replay was missing -- writing -0.11, summarization -0.07 -- while technical domains barely
    moved (-0.01/-0.04). A mean would have hidden that, so free-form English is broken out.

  python scripts/sweep_tr27b_checkpoints.py --out out/flow/tr27b_sweep.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/shadeform/chained-flow")
CKPT_DIR = ROOT / "out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8"
V2 = ROOT / "out/flow/ckpts/tree-vae-joint-q3527bx-1024-k8-l8"
SHORTLIST = ROOT / "out/flow/shortlist_q3527b_tr_bare.pt"
ROW = re.compile(r"^(?P<dom>\S+)\s+(?P<n>\d+)\s+[\d.]+\s+[\d.]+\s+(?P<pchain>[\d.]+)\s+[+-][\d.]+")
FREEFORM = {"writing", "summarization", "qa"}   # vs technical: HumanEval, math_reasoning, gsm8k-test


def run_eval(ckd: Path, states: str, per_domain: int, gpu: str) -> dict[str, float]:
    cmd = [".venv/bin/python", "scripts/diff_plugin_vs_harness.py",
           "--ckd", str(ckd), "--model", "Qwen/Qwen3.5-27B", "--states", states,
           "--shortlist", str(SHORTLIST), "--per_domain", str(per_domain)]
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "src", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    import os
    e = dict(os.environ); e.update(env)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=e)
    out: dict[str, float] = {}
    for line in p.stdout.splitlines():
        m = ROW.match(line)
        if m and m.group("dom") not in {"MEAN", "domain"}:
            out[m.group("dom")] = float(m.group("pchain"))
    if not out:
        print(f"    !! no rows parsed for {ckd.name} / {states}")
        print("    " + (p.stderr.strip().splitlines() or ["<no stderr>"])[-1][:200])
    return out


def mean(d: dict[str, float], keys=None) -> float | None:
    v = [x for k, x in d.items() if keys is None or k in keys]
    return sum(v) / len(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/flow/tr27b_sweep.json")
    ap.add_argument("--per-domain", type=int, default=200, help="windows/domain; lower = faster sweep")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--tr-states", default="teacher_states/bench-tr27b-*")
    ap.add_argument("--en-states", default="teacher_states/[bh]*-q3527b-*")
    args = ap.parse_args()

    cks = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    targets: list[tuple[str, Path]] = [("v2 (before)", V2)]
    for c in cks:
        cfg = c / "chained_flow_tree_config.json"
        if not cfg.exists():
            src = (CKPT_DIR / "chained_flow_tree_config.json")
            src = src if src.exists() else (V2 / "chained_flow_tree_config.json")
            shutil.copy(src, cfg)
        targets.append((c.name, c))
    if (CKPT_DIR / "model.safetensors").exists():
        targets.append(("final", CKPT_DIR))

    print(f"sweeping {len(targets)} checkpoints, per_domain={args.per_domain}, GPU {args.gpu}\n")
    rows = []
    for label, ckd in targets:
        tr = run_eval(ckd, args.tr_states, args.per_domain, args.gpu)
        en = run_eval(ckd, args.en_states, args.per_domain, args.gpu)
        rows.append({"label": label, "ckpt": str(ckd), "tr": tr, "en": en,
                     "tr_mean": mean(tr), "en_mean": mean(en),
                     "en_freeform": mean(en, FREEFORM),
                     "en_technical": mean(en, set(en) - FREEFORM)})
        r = rows[-1]
        f = lambda x: "  n/a" if x is None else f"{x:5.2f}"
        print(f"{label:<16} TR {f(r['tr_mean'])}   EN {f(r['en_mean'])}  "
              f"(free-form {f(r['en_freeform'])}, technical {f(r['en_technical'])})", flush=True)

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)

    base = rows[0]
    print("\n" + "=" * 78)
    print(f"{'checkpoint':<16} {'TR mean':>8} {'ΔTR':>7} {'EN mean':>8} {'ΔEN':>7} "
          f"{'EN free':>8} {'Δfree':>7}")
    print("-" * 78)
    for r in rows:
        d = lambda a, b: "     —" if a is None or b is None else f"{a - b:+6.2f}"
        f = lambda x: "     —" if x is None else f"{x:6.2f}"
        print(f"{r['label']:<16} {f(r['tr_mean']):>8} {d(r['tr_mean'], base['tr_mean']):>7} "
              f"{f(r['en_mean']):>8} {d(r['en_mean'], base['en_mean']):>7} "
              f"{f(r['en_freeform']):>8} {d(r['en_freeform'], base['en_freeform']):>7}")
    print("\nShip the checkpoint where TR has saturated and EN free-form has not yet eroded.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
