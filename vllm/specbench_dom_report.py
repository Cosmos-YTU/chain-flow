#!/usr/bin/env python3
"""Aggregate the PER-DOMAIN specbench runs produced by specbench_dom_run.sh.

METRIC DEFINITIONS (stated so they can be quoted):

  acceptance   = mean accepted length = 1 + num_accepted_tokens / num_drafts, taken from
                 the delta of vLLM's OWN cumulative Prometheus counters
                 (vllm:spec_decode_num_accepted_tokens_total / _num_drafts_total) across
                 the domain's benchmark window.  THE BONUS TOKEN IS INCLUDED.  This is the
                 same formula as speculators/tests/e2e/run_vllm.py and as vLLM's
                 "Mean acceptance length" log line.  It stays comparable across draft
                 geometries, which a per-position rate does not: for a TREE
                 num_draft_tokens would be the NODE COUNT, so any rate of the form
                 accepted/draft_tokens -- including vLLM's "Avg Draft acceptance rate"
                 and its per-position rates -- would be meaningless there; mean accepted
                 length has no such problem.  Base has no drafts and is 1.000 by
                 definition.  The chain arm drafts K=5 tokens per step.

  pooled_tps   = sum(output tokens) / benchmark wall duration.  PREFILL IS IN THE
                 DENOMINATOR.  This is the end-to-end number a user sees.
  decode_tps   = 1000 / mean(time_per_output_token_ms).  TPOT starts at the first token,
                 so PREFILL IS EXCLUDED.  Unweighted mean over requests.
  speedup      = arm / base of the same metric, per domain, on the mean over repeats.
  spread       = (max - min) / mean over repeats, in percent.

Every request emits exactly 256 tokens (ignore_eos), so all arms do identical work and
throughput stays comparable even where a fp16 tie-flip sends the text down another path.
"""
import argparse
import collections
import json
import pathlib
import re
import statistics

# Tree is deliberately EXCLUDED. The tree arm is greedy-only in practice (a sampled
# request is rejected unless a default-off prototype is enabled, and then it serves at
# 0.58x), so it is not the configuration real serving uses and not the number to publish.
# Any tree files on disk are left alone; they simply never enter this report.
ARMS = ("base", "chain")
PROM_RE = re.compile(r"^(vllm:\S+?)(?:\{[^}]*\})?\s+([\d.eE+-]+)$")


def read_prom(path):
    vals = {}
    if not path.exists():
        return vals
    for line in path.read_text().splitlines():
        mo = PROM_RE.match(line.strip())
        if mo:
            vals[mo.group(1)] = vals.get(mo.group(1), 0.0) + float(mo.group(2))
    return vals


def accept_counters(before, after):
    b, a = read_prom(before), read_prom(after)
    if not a:
        return None

    def d(k):
        return a.get(k, 0.0) - b.get(k, 0.0)

    drafts = d("vllm:spec_decode_num_drafts_total") or d("vllm:spec_decode_num_drafts")
    dtok = (d("vllm:spec_decode_num_draft_tokens_total")
            or d("vllm:spec_decode_num_draft_tokens"))
    acc = (d("vllm:spec_decode_num_accepted_tokens_total")
           or d("vllm:spec_decode_num_accepted_tokens"))
    if drafts <= 0:
        return None
    return {"accept": 1.0 + acc / drafts, "accepted": acc, "draft_tokens": dtok,
            "drafts": drafts}


def load_run(path):
    d = json.loads(path.read_text())
    b = d["benchmarks"][0]
    m = b["metrics"]
    succ = b["requests"]["successful"]
    out_tokens = m["output_token_count"]["successful"]["total_sum"]
    dur = b["duration"]
    tpot = m["time_per_output_token_ms"]["successful"]["mean"]
    outputs = {}
    for r in succ:
        body = json.loads(r["request_args"])["body"]
        outputs[body["messages"][0]["content"][0]["text"]] = r.get("output") or ""
    conc = m.get("request_concurrency", {}).get("successful", {})
    return {"n_req": len(succ), "n_err": len(b["requests"].get("errored") or []),
            "conc_mean": conc.get("mean"), "conc_max": conc.get("max"),
            "out_tokens": out_tokens, "duration": dur,
            "pooled_tps": out_tokens / dur if dur else None,
            "decode_tps": 1000.0 / tpot if tpot else None,
            "ttft_ms": m["time_to_first_token_ms"]["successful"]["mean"],
            "outputs": outputs}


def collect(root, size):
    """-> {arm: {domain: {rep: rec}}}"""
    data = {}
    for arm in ARMS:
        d = pathlib.Path(root) / f"{size}_{arm}"
        if not d.exists():
            continue
        per = collections.defaultdict(dict)
        for f in sorted(d.glob("gl_*.json")):
            tag = f.stem[3:]
            if "_r" not in tag:
                continue
            dom, rep = tag.rsplit("_r", 1)
            if rep == "0" or dom in ("WARMUP", "gate_pooled"):
                continue
            try:
                rec = load_run(f)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! {f.name}: {exc}")
                continue
            rec["accept"] = accept_counters(d / f"metrics_{tag}.before",
                                            d / f"metrics_{tag}.after")
            per[dom][rep] = rec
        if per:
            data[arm] = dict(per)
    return data


def agg(reps, key):
    v = [r[key] for r in reps.values() if r.get(key)]
    if not v:
        return None, 0.0
    m = statistics.mean(v)
    sp = (max(v) - min(v)) / m * 100 if len(v) > 1 else 0.0
    return m, sp


def agg_accept(reps):
    accs = [r["accept"] for r in reps.values() if r.get("accept")]
    if not accs:
        return None, 0.0
    vals = [a["accept"] for a in accs]
    return statistics.mean(vals), (max(vals) - min(vals)) / statistics.mean(vals) * 100 \
        if len(vals) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/shadeform/chained-flow/logs/specbench_dom")
    ap.add_argument("--size", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data = collect(args.root, args.size)
    if not data:
        print(f"no data for {args.size}")
        return
    doms = sorted(set().union(*[set(v) for v in data.values()]))

    result = {"size": args.size, "domains": {}, "pooled": {}}

    hdr = f"{'domain':16s}"
    for arm in ARMS:
        if arm in data:
            hdr += f" | {arm+' acc':>9s} {arm+' tps':>9s} {'spr%':>5s}"
    print(f"\n########## {args.size}  (concurrency 1, greedy, 256 fixed output tokens, "
          f"async sched ON all arms) ##########")
    print("\n--- pooled_tps (prefill INCLUDED) + acceptance, per domain ---")
    print(hdr + " | " + " ".join(f"{a+' spdup':>11s}" for a in ARMS if a in data and a != "base"))

    for dom in doms:
        line = f"{dom:16s}"
        vals = {}
        for arm in ARMS:
            if arm not in data:
                continue
            reps = data[arm].get(dom, {})
            if not reps:
                line += f" | {'-':>9s} {'-':>9s} {'-':>5s}"
                continue
            tps, spr = agg(reps, "pooled_tps")
            acc, aspr = agg_accept(reps)
            dtps, dspr = agg(reps, "decode_tps")
            vals[arm] = {"pooled_tps": tps, "pooled_spread_pct": spr,
                         "decode_tps": dtps, "decode_spread_pct": dspr,
                         "accept": acc if arm != "base" else 1.0,
                         "accept_spread_pct": aspr,
                         "n_req": max(r["n_req"] for r in reps.values()),
                         "reps": len(reps)}
            astr = f"{acc:.3f}" if acc else ("1.000" if arm == "base" else "-")
            line += f" | {astr:>9s} {tps:9.1f} {spr:5.1f}"
        for arm in ARMS:
            if arm in vals and arm != "base" and "base" in vals:
                line += f" {vals[arm]['pooled_tps']/vals['base']['pooled_tps']:10.3f}x"
        print(line)
        result["domains"][dom] = vals

    print("\n--- decode_tps (prefill EXCLUDED), per domain ---")
    print(f"{'domain':16s}" + "".join(f" | {a+' dtps':>10s} {'spr%':>5s}"
                                      for a in ARMS if a in data)
          + " | " + " ".join(f"{a+' spdup':>11s}" for a in ARMS if a in data and a != "base"))
    for dom in doms:
        v = result["domains"][dom]
        line = f"{dom:16s}"
        for arm in ARMS:
            if arm in v:
                line += f" | {v[arm]['decode_tps']:10.1f} {v[arm]['decode_spread_pct']:5.1f}"
            elif arm in data:
                line += f" | {'-':>10s} {'-':>5s}"
        for arm in ARMS:
            if arm in v and arm != "base" and "base" in v:
                line += f" {v[arm]['decode_tps']/v['base']['decode_tps']:10.3f}x"
        print(line)

    # ---- pooled over all domains ------------------------------------------
    # Pool ONLY over domains every arm actually ran. Pooling a domain the baseline is
    # missing would compare different domain mixes and silently flatter whichever arm has
    # the easier set -- the exact error the "always quote pooled" rule exists to prevent.
    common_doms = set.intersection(*[set(v) for v in data.values()]) if data else set()
    skipped = sorted(set(doms) - common_doms)
    print("\n--- POOLED over domains run by every arm (sum tokens / sum duration; "
          "acceptance = 1 + sum accepted / sum drafts) ---")
    print(f"  pooling {len(common_doms)} domains" +
          (f"; EXCLUDED (not in all arms): {skipped}" if skipped else ""))
    for arm in ARMS:
        if arm not in data:
            continue
        # Pool only COMPLETE repeats. A repeat still in flight covers a subset of the
        # domains, so including it pools a different domain mix and produces a fake
        # run-to-run "spread" that is really a mix difference.
        all_reps = {rep for d in common_doms for rep in data[arm].get(d, {})}
        complete = {rep for rep in all_reps
                    if all(rep in data[arm].get(d, {}) for d in common_doms)}
        tok = dur = acc = drf = 0.0
        per_rep = collections.defaultdict(lambda: [0.0, 0.0])
        for dom, reps in data[arm].items():
            if dom not in common_doms:
                continue  # pool only what every arm has, or arms aren't comparable
            for rep, r in reps.items():
                if rep not in complete:
                    continue
                tok += r["out_tokens"]
                dur += r["duration"]
                per_rep[rep][0] += r["out_tokens"]
                per_rep[rep][1] += r["duration"]
                if r.get("accept"):
                    acc += r["accept"]["accepted"]
                    drf += r["accept"]["drafts"]
        rep_tps = [v[0] / v[1] for v in per_rep.values() if v[1]]
        spr = (max(rep_tps) - min(rep_tps)) / statistics.mean(rep_tps) * 100 \
            if len(rep_tps) > 1 else 0.0
        # decode_tps pooled = mean over domains of per-domain decode_tps
        dt = statistics.mean([result["domains"][d][arm]["decode_tps"]
                              for d in sorted(common_doms)
                              if arm in result["domains"][d]])
        accept = 1.0 + acc / drf if drf else 1.0
        result["pooled"][arm] = {"pooled_tps": tok / dur, "pooled_spread_pct": spr,
                                 "complete_repeats": len(complete),
                                 "decode_tps": dt, "accept": accept}
        print(f"  {arm:6s} accept {accept:6.3f}  pooled_tps {tok/dur:7.2f} "
              f"(spread {spr:4.1f}% over {len(complete)} complete repeat(s))  "
              f"decode_tps {dt:7.2f}", end="")
        if arm != "base" and "base" in result["pooled"]:
            b = result["pooled"]["base"]
            print(f"  | speedup pooled {tok/dur/b['pooled_tps']:.3f}x "
                  f"decode {dt/b['decode_tps']:.3f}x")
        else:
            print()

    # ---- equal-work check --------------------------------------------------
    # HumanEval hits EOS after ~45-67 tokens when left to itself, and identical repeats
    # have then read 233 vs 277 tok/s. output_tokens_count=256 + ignore_eos is supposed to
    # remove that entirely. VERIFY it rather than assume: every request must emit exactly
    # 256 tokens, or the arms are not doing identical work and tok/s is not comparable.
    print("\n--- equal-work check: every request must emit exactly 256 tokens ---")
    bad = []
    for arm in ARMS:
        for dom, reps in data.get(arm, {}).items():
            for rep, r in reps.items():
                exp = 256 * r["n_req"]
                if r["out_tokens"] != exp or r["n_err"]:
                    bad.append(f"{arm}/{dom}/r{rep}: {r['out_tokens']:.0f} tokens over "
                               f"{r['n_req']} reqs (expected {exp}), errors={r['n_err']}")
    if bad:
        print("  !! UNEQUAL WORK -- throughput NOT comparable:")
        for b in bad:
            print(f"     {b}")
    else:
        nreq = sum(r["n_req"] for arm in ARMS for reps in data.get(arm, {}).values()
                   for r in reps.values())
        print(f"  OK: all {nreq} requests emitted exactly 256 tokens, 0 errors")
    result["equal_work_violations"] = bad

    cm = [r["conc_mean"] for arm in ARMS for reps in data.get(arm, {}).values()
          for r in reps.values() if r.get("conc_mean") is not None]
    cx = [r["conc_max"] for arm in ARMS for reps in data.get(arm, {}).values()
          for r in reps.values() if r.get("conc_max") is not None]
    if cm:
        result["concurrency"] = {"mean": statistics.mean(cm), "max": max(cx)}
        print(f"  concurrency: mean {statistics.mean(cm):.4f}, max {max(cx):.2f} "
              f"(synchronous profile -> one request in flight)")

    # ---- divergence vs base ------------------------------------------------
    print("\n--- greedy output divergence vs base (repeat 1; identical / common) ---")
    div = {}
    for arm in ARMS:
        if arm == "base" or arm not in data or "base" not in data:
            continue
        tot_c = tot_d = 0
        per = {}
        for dom in doms:
            b = data["base"].get(dom, {}).get("1")
            o = data[arm].get(dom, {}).get("1")
            if not b or not o:
                continue
            common = set(b["outputs"]) & set(o["outputs"])
            nd = sum(1 for p in common if b["outputs"][p] != o["outputs"][p])
            per[dom] = (len(common) - nd, len(common))
            tot_c += len(common)
            tot_d += nd
        div[arm] = {"per_domain": per, "total_common": tot_c, "total_diff": tot_d}
        print(f"  {arm:6s} " + "  ".join(f"{d}:{a}/{b}" for d, (a, b) in per.items())
              + f"   TOTAL divergent {tot_d}/{tot_c}")
    result["divergence"] = div

    # base repeat-vs-repeat determinism check
    if "base" in data:
        nd = nc = 0
        for dom in doms:
            r1 = data["base"].get(dom, {}).get("1")
            r2 = data["base"].get(dom, {}).get("2")
            if not r1 or not r2:
                continue
            common = set(r1["outputs"]) & set(r2["outputs"])
            nc += len(common)
            nd += sum(1 for p in common if r1["outputs"][p] != r2["outputs"][p])
        if nc:
            print(f"  base   repeat1 vs repeat2: {nc-nd}/{nc} identical "
                  f"(server-side determinism check)")

    # Provenance stamped into the results file so a card can prove which run it quotes.
    import subprocess, datetime as _dt
    snap = pathlib.Path(args.root) / "src_snapshot_PROVENANCE.txt"
    result["provenance"] = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "results_root": str(args.root),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, cwd="/home/shadeform/chained-flow"
                                   ).stdout.strip(),
        "src_snapshot": snap.read_text().splitlines()[0] if snap.exists() else None,
        "benchmark": "RedHatAI/speculator_benchmarks",
        "harness": "guidellm 0.6.0 -> vllm serve (vLLM 0.25.1)",
        "arms": list(ARMS),
        "profile": "synchronous (concurrency 1)",
        "output_tokens": 256,
        "greedy": True,
        "async_scheduling_all_arms": True,
    }

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
