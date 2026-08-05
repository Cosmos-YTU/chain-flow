#!/usr/bin/env python3
"""Aggregate the RedHat AI `speculator_benchmarks` runs produced by specbench_run.sh.

Reports guidellm's OWN metrics, with the prefill question made explicit:

  gl_output_tps   guidellm's headline "Output Tokens Per Sec" =
                  metrics.output_tokens_per_second.successful.mean.  guidellm builds
                  this from a per-token time series, so it is a TOKEN-weighted mean of
                  instantaneous rates, not a pooled ratio.
  pooled_tps      sum(output tokens) / benchmark wall duration.  PREFILL IS IN THE
                  DENOMINATOR.  This is the ratio our own 7-domain sweep reports.
  decode_tps      1000 / mean(time_per_output_token_ms).  TPOT starts at the first
                  token, so PREFILL IS EXCLUDED.  Unweighted mean over REQUESTS.
  ttft_ms         mean time to first token (the prefill cost that pooled_tps carries).

Acceptance comes from vLLM's own counters, not ours: the server's periodic
"SpecDecoding metrics:" line, whose "Mean acceptance length" is
1 + num_accepted_tokens/num_drafts -- i.e. the BONUS TOKEN IS INCLUDED, the same
convention as our CF_ACCEPT counter.  Lines are attributed to a run by timestamp
using the CFMARK_START/CFMARK_END marks specbench_run.sh writes.
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import statistics

SPEC_RE = re.compile(
    r"(?P<mon>\d\d)-(?P<day>\d\d) (?P<h>\d\d):(?P<m>\d\d):(?P<s>\d\d)"
    r".*SpecDecoding metrics: Mean acceptance length: (?P<mal>[\d.]+), "
    r".*Accepted: (?P<acc>\d+) tokens, Drafted: (?P<drf>\d+) tokens, "
    r"Per-position acceptance rate: (?P<pos>[\d., ]+?), Avg Draft acceptance rate: (?P<rate>[\d.]+)%"
)


def parse_marks(path):
    marks = {}
    if not path.exists():
        return marks
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        kind, tag, ts = parts
        marks.setdefault(tag, {})[kind] = float(ts)
    return marks


def parse_spec_log(path, year):
    """Return [(unix_ts, mean_accept_len, accepted, drafted, per_pos)] from a server log."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        mo = SPEC_RE.search(line)
        if not mo:
            continue
        stamp = dt.datetime(year, int(mo["mon"]), int(mo["day"]),
                            int(mo["h"]), int(mo["m"]), int(mo["s"]))
        out.append((stamp.timestamp(), float(mo["mal"]), int(mo["acc"]),
                    int(mo["drf"]), [float(x) for x in mo["pos"].split(",")]))
    return out


def accept_for_window(spec_rows, t0, t1):
    """Pool vLLM's own counters over the run window: 1 + accepted/drafts.

    num_drafts is not in the log line, but each logged interval reports Accepted and
    Drafted totals; drafts are recovered as Drafted/num_spec_tokens only when the draft
    length is fixed, which it is not for the tree arm.  So pool the interval-level
    mean acceptance lengths weighted by Drafted tokens, which is what the RedHat
    parse_logs.py does for the per-position rates.
    """
    rows = [r for r in spec_rows if t0 - 1 <= r[0] <= t1 + 1]
    if not rows:
        return None
    wsum = sum(r[3] for r in rows)
    if wsum == 0:
        return None
    mal = sum(r[1] * r[3] for r in rows) / wsum
    acc = sum(r[2] for r in rows)
    drf = sum(r[3] for r in rows)
    return {"mean_accept_len": mal, "accepted": acc, "drafted": drf,
            "draft_accept_rate_pct": 100.0 * acc / drf, "intervals": len(rows)}


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


def accept_from_counters(before, after):
    """Exact acceptance from vLLM's own cumulative counters.

    acceptance_length = 1 + num_accepted_tokens / num_drafts -- the BONUS TOKEN IS
    INCLUDED, identical to speculators/tests/e2e/run_vllm.py:extract_metrics and to
    vLLM's own "Mean acceptance length" log line.
    """
    b, a = read_prom(before), read_prom(after)
    if not a:
        return None
    def d(k):
        return a.get(k, 0.0) - b.get(k, 0.0)
    drafts = d("vllm:spec_decode_num_drafts_total") or d("vllm:spec_decode_num_drafts")
    dtok = d("vllm:spec_decode_num_draft_tokens_total") or d("vllm:spec_decode_num_draft_tokens")
    acc = (d("vllm:spec_decode_num_accepted_tokens_total")
           or d("vllm:spec_decode_num_accepted_tokens"))
    if drafts <= 0:
        return None
    return {"mean_accept_len": 1.0 + acc / drafts, "accepted": acc, "drafted": dtok,
            "drafts": drafts,
            "draft_accept_rate_pct": 100.0 * acc / dtok if dtok else float("nan"),
            "source": "prometheus"}


def load_run(path):
    d = json.loads(path.read_text())
    b = d["benchmarks"][0]
    m = b["metrics"]
    succ = b["requests"]["successful"]
    out_tokens = m["output_token_count"]["successful"]["total_sum"]
    dur = b["duration"]
    tpot = m["time_per_output_token_ms"]["successful"]["mean"]
    return {
        "n_req": len(succ),
        "n_err": len(b["requests"].get("errored") or []),
        "out_tokens": out_tokens,
        "duration": dur,
        "gl_output_tps": m["output_tokens_per_second"]["successful"]["mean"],
        "pooled_tps": out_tokens / dur if dur else None,
        "decode_tps": 1000.0 / tpot if tpot else None,
        "tpot_ms": tpot,
        "ttft_ms": m["time_to_first_token_ms"]["successful"]["mean"],
        "outputs": [(r["request_args"], r.get("output") or "") for r in succ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/shadeform/chained-flow/logs/specbench")
    ap.add_argument("--size", required=True)
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    arms = {}
    for arm in ("base", "chain", "tree"):
        d = root / f"{args.size}_{arm}"
        if not d.exists():
            continue
        marks = parse_marks(d / "marks.txt")
        spec = parse_spec_log(d / "vllm_server.log", args.year)
        runs = {}
        for f in sorted(d.glob("gl_*.json")):
            tag = f.stem[3:]
            if tag.endswith("_r0"):
                continue  # warmup
            try:
                rec = load_run(f)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! {f.name}: {exc}")
                continue
            # Prefer the exact Prometheus counter delta; fall back to log-window pooling.
            rec["accept"] = accept_from_counters(d / f"metrics_{tag}.before",
                                                 d / f"metrics_{tag}.after")
            if rec["accept"] is None:
                mk = marks.get(tag, {})
                if "CFMARK_START" in mk and "CFMARK_END" in mk:
                    rec["accept"] = accept_for_window(spec, mk["CFMARK_START"],
                                                      mk["CFMARK_END"])
            runs[tag] = rec
        arms[arm] = runs

    # ---- table -------------------------------------------------------------
    cfgs = sorted({t.rsplit("_r", 1)[0] for runs in arms.values() for t in runs})
    for cfg in cfgs:
        print(f"\n=== {args.size}  config={cfg} ===")
        print(f"{'arm':6s} {'rep':4s} {'reqs':5s} {'outTok':8s} {'secs':7s} "
              f"{'pooled_tps':11s} {'decode_tps':11s} {'gl_out_tps':11s} "
              f"{'ttft_ms':8s} {'accept':7s}")
        base_vals = {}
        for arm in ("base", "chain", "tree"):
            for tag, rec in sorted(arms.get(arm, {}).items()):
                if not tag.startswith(cfg + "_r"):
                    continue
                rep = tag.rsplit("_r", 1)[1]
                acc = rec.get("accept")
                accs = f"{acc['mean_accept_len']:.3f}" if acc else "-"
                def f(v, w=11, p=2):
                    return f"{v:<{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':<{w}}"

                print(f"{arm:6s} {rep:4s} {rec['n_req']:<5d} {rec['out_tokens']:<8.0f} "
                      f"{rec['duration']:<7.1f} {f(rec['pooled_tps'])} "
                      f"{f(rec['decode_tps'])} {f(rec['gl_output_tps'])} "
                      f"{f(rec['ttft_ms'], 8, 1)} {accs:7s}")
                if rec["n_req"] and rec["pooled_tps"] and rec["decode_tps"]:
                    base_vals.setdefault(arm, []).append(rec)
        # speedups + spread
        if "base" in base_vals:
            def mean(a, k):
                return statistics.mean(r[k] for r in base_vals[a])

            def spread(a, k):
                v = [r[k] for r in base_vals[a]]
                return (max(v) - min(v)) / statistics.mean(v) * 100 if len(v) > 1 else 0.0

            print(f"  {'-'*70}")
            for a in ("base", "chain", "tree"):
                if a not in base_vals:
                    continue
                print(f"  {a:6s} pooled {mean(a,'pooled_tps'):7.2f} "
                      f"(spread {spread(a,'pooled_tps'):4.1f}%)  "
                      f"decode {mean(a,'decode_tps'):7.2f} "
                      f"(spread {spread(a,'decode_tps'):4.1f}%)"
                      + (f"  | speedup pooled {mean(a,'pooled_tps')/mean('base','pooled_tps'):.3f}x"
                         f" decode {mean(a,'decode_tps')/mean('base','decode_tps'):.3f}x"
                         if a != "base" else ""))

    # ---- output diff (greedy configs only) ---------------------------------
    print(f"\n=== {args.size} output-token diff vs base (greedy fixed256 only) ===")
    for cfg in cfgs:
        if "fixed256" not in cfg:
            continue
        btag = next((t for t in arms.get("base", {}) if t.startswith(cfg + "_r")), None)
        if btag is None:
            continue
        bout = {a: o for a, o in arms["base"][btag]["outputs"]}
        for arm in ("chain", "tree"):
            tag = next((t for t in arms.get(arm, {}) if t.startswith(cfg + "_r")), None)
            if tag is None:
                continue
            aout = {a: o for a, o in arms[arm][tag]["outputs"]}
            common = set(bout) & set(aout)
            diff = [a for a in common if bout[a] != aout[a]]
            print(f"  {cfg:24s} {arm:5s}: {len(common)-len(diff)}/{len(common)} identical"
                  + (f"  DIFFERING: {len(diff)}" if diff else ""))

    per_domain(args.root, args.size, "fixed256_synchronous_r1")


def per_domain(root, size, cfg):
    """Pooled tok/s per dataset FILE, so the aggregation choice is visible.

    The seven files are separate benchmarks in the RedHat harness and are never pooled by
    it; our own sweep pools them. The two give different speedups because the domains have
    very different accept rates, so this prints both.
    """
    import collections
    dom = {}
    for line in open(f"{root}/data/subset_fixed256.jsonl"):
        r = json.loads(line)
        dom[r["prompt"]] = r["domain"]

    print(f"\n=== {size} per-domain pooled tok/s [{cfg}] "
          f"(sum output tokens / sum request latency, prefill INCLUDED) ===")
    rows = {}
    for arm in ("base", "chain", "tree"):
        f = pathlib.Path(root) / f"{size}_{arm}" / f"gl_{cfg}.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())["benchmarks"][0]
        agg = collections.defaultdict(lambda: [0.0, 0.0])
        for r in b["requests"]["successful"]:
            p = json.loads(r["request_args"])["body"]["messages"][0]["content"][0]["text"]
            d = dom.get(p, "?")
            agg[d][0] += r["output_tokens"]
            agg[d][1] += r["request_latency"]
        rows[arm] = {k: v[0] / v[1] for k, v in agg.items() if v[1]}
    if not rows:
        return
    doms = sorted(set().union(*[set(v) for v in rows.values()]))
    print(f"{'domain':16s} " + " ".join(f"{a:>9s}" for a in rows)
          + "   " + " ".join(f"{a+'/base':>11s}" for a in rows if a != "base"))
    for d in doms:
        line = f"{d:16s} " + " ".join(f"{rows[a].get(d, float('nan')):9.1f}" for a in rows)
        if "base" in rows and rows["base"].get(d):
            line += "   " + " ".join(
                f"{rows[a].get(d, float('nan'))/rows['base'][d]:11.3f}"
                for a in rows if a != "base")
        print(line)


if __name__ == "__main__":
    main()
