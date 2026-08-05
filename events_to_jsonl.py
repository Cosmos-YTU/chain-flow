#!/usr/bin/env python3
"""
Convert a recovered Claude Code session events dump (from
/v1/code/sessions/<id>/events) into a local ~/.claude/projects/<slug>/<uuid>.jsonl
history file that `claude --resume` can load.

Usage:
    python3 events_to_jsonl.py megakernel_session_events.json \
        --cwd /root/megakernel \
        --session-id 3a027529-5e70-4641-8250-1137e63d1044

Then:
    cd /root/megakernel && claude --resume
"""

import argparse
import json
import uuid as uuidlib
from datetime import datetime, timezone
from pathlib import Path

# Event types carrying conversation turns. 'system' (11k of them), 'result',
# and the control_* events are runtime bookkeeping -- dropped.
KEEP = {"user", "assistant"}


def slug_for(cwd: str) -> str:
    """Claude Code encodes cwd as the project dir: /root/megakernel -> -root-megakernel"""
    return cwd.replace("/", "-")


def iso(ts: str) -> str:
    if not ts:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return ts.replace("+00:00", "Z")


def blocks(message):
    c = message.get("content")
    return c if isinstance(c, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--cwd", required=True, help="e.g. /root/megakernel")
    ap.add_argument("--session-id", default=None,
                    help="reuse the original session UUID; omit to generate a new one")
    ap.add_argument("--version", default="2.1.207")
    ap.add_argument("--branch", default="")
    ap.add_argument("--projects-root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--keep-dangling", action="store_true",
                    help="do NOT trim trailing turns with unmatched tool_use blocks")
    args = ap.parse_args()

    events = json.load(open(args.events, encoding="utf-8"))
    print(f"{len(events)} events loaded")

    session_id = args.session_id or str(uuidlib.uuid4())
    outdir = Path(args.projects_root) / slug_for(args.cwd)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{session_id}.jsonl"

    if outpath.exists():
        backup = outpath.with_suffix(".jsonl.bak")
        outpath.rename(backup)
        print(f"existing file moved to {backup}")

    # --- collect conversation turns in order ---
    turns = []
    for ev in events:
        if ev.get("event_type") not in KEEP:
            continue
        payload = ev.get("payload") or {}
        msg = payload.get("message")
        if not msg or not msg.get("content"):
            continue
        turns.append((ev, payload, msg))

    print(f"{len(turns)} conversation turns")

    # --- validate tool_use / tool_result pairing ---
    # A tool_use block in an assistant turn MUST be answered by a tool_result
    # block (same id) in a later user turn, or the API rejects the history.
    results_seen = set()
    for _, _, msg in turns:
        for b in blocks(msg):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                results_seen.add(b.get("tool_use_id"))

    dangling_at = None
    for i, (_, _, msg) in enumerate(turns):
        ids = [b.get("id") for b in blocks(msg)
               if isinstance(b, dict) and b.get("type") == "tool_use"]
        if ids and not all(x in results_seen for x in ids):
            dangling_at = i
            break

    if dangling_at is not None:
        print(f"WARNING: turn {dangling_at} has tool_use with no matching tool_result")
        if not args.keep_dangling:
            turns = turns[:dangling_at]
            print(f"trimmed to {len(turns)} turns (use --keep-dangling to override)")

    # --- emit jsonl ---
    lines = [
        {"type": "mode", "mode": "normal", "sessionId": session_id},
        {"type": "permission-mode", "permissionMode": "default", "sessionId": session_id},
    ]

    parent = None
    for ev, payload, msg in turns:
        u = ev.get("event_id") or str(uuidlib.uuid4())
        rec = {
            "parentUuid": parent,
            "isSidechain": False,
            "type": ev["event_type"],
            "message": msg,
            "uuid": u,
            "timestamp": iso(ev.get("created_at", "")),
            "userType": "external",
            "entrypoint": "cli",
            "cwd": args.cwd,
            "sessionId": session_id,
            "version": args.version,
        }
        if args.branch:
            rec["gitBranch"] = args.branch
        if ev["event_type"] == "assistant" and payload.get("requestId"):
            rec["requestId"] = payload["requestId"]
        lines.append(rec)
        parent = u

    with outpath.open("w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nwrote {outpath}  ({outpath.stat().st_size / 1e6:.1f} MB)")
    print(f"session id: {session_id}")
    print(f"\nNow run:\n    cd {args.cwd} && claude --resume")


if __name__ == "__main__":
    main()