"""``chain-flow tree-patch`` -- apply / revert / verify the OPTIONAL forked-vLLM tree path.

WHY THIS IS A COMMAND AND NOT A README PARAGRAPH
------------------------------------------------
The tree arm is worth 1.35x / 1.34x / 1.78x against the chain's ~1.13x / 1.5x, and the only
thing standing between a pip user and those numbers is "patch your site-packages".  Told to do
that by hand with ``patch -p1``, the realistic outcomes are: it half-applies with ``.rej`` files
and the user runs anyway; it applies with fuzz against a near-miss vLLM; or it applies against a
version whose call sites moved.  None of those raise.  A tree run on a half-applied fork produces
MALFORMED DRAFTS -- wrong ancestor masks, wrong GDN state -- which is not a crash, it is quietly
wrong text at full speed.  So every unsafe outcome has to be turned into a refusal here.

THE FIVE SAFETY PROPERTIES, AND WHAT ENFORCES EACH
--------------------------------------------------
1. **The target is the ACTIVE vLLM.**  Located through the import system
   (``importlib.util.find_spec``), never by guessing ``site-packages`` from ``sys.prefix``.
   ``find_spec`` is deliberately used in place of ``import vllm``: a real import costs ~20 s of
   torch, and on a half-applied install it RAISES -- exactly the state ``--status`` exists to
   diagnose.  It resolves through the same finders/``sys.path`` an import would, so the answer is
   the same one the engine will get.

2. **The version gate is hard and doubled.**  ``vllm/_version.py`` (which is where
   ``vllm.__version__`` comes from) and the dist-info metadata must BOTH read 0.25.1 and must
   agree with each other.  A disagreement means the tree was swapped underneath its own metadata,
   which is worse than a plain mismatch, so it is its own refusal.

3. **Before and after are verified against the WHEEL'S OWN RECORD**, not against a checked-in
   backup copy.  ``vllm-<ver>.dist-info/RECORD`` carries a b64 sha256 for all 4510 installed
   files; every one is hashed (0.6 s).  This is the only trustworthy oracle: this repo's
   ``vllm/_vllm_backup/`` is STALE -- it omits ``config/vllm.py`` and holds five zero-diff
   leftovers -- and a previous investigation drew conclusions from it.

4. **All-or-nothing, with no fuzz.**  The unified diff is applied by ``_apply_hunks`` below at
   the exact line numbers in the hunk headers, with every context and removed line compared
   byte-for-byte.  There is no fuzz factor, no offset search, and no ``.rej``: one bad hunk
   aborts the whole command before a single byte is written.  Every file's new content is built
   in memory and only then written (atomically, via a temp file + ``os.replace``).

5. **Revert is proved, not hoped.**  ``--revert`` REVERSE-applies the same hunks and then
   re-hashes against RECORD.  That means it needs no saved backup and therefore cannot be
   defeated by anything the user did to their environment in between -- a pip upgrade that
   rewrote an untouched file is simply reported by the RECORD scan.  Deleting the added files is
   the easy half; restoring the modified ones byte-identically is the half that needs proof.

WHAT "APPLIED" MEANS HERE
-------------------------
Not "the marker files exist".  A file counts as patched only if reverse-applying the hunks
reproduces content whose sha256 equals the RECORD entry -- i.e. it is provably the pristine file
plus exactly this patch.  Added files must hash equal to the content the patch carries.  Anything
else is ``UNKNOWN`` and blocks both apply and revert with the file named.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PIN = "0.25.1"
PATCH_NAME = f"vllm-{PIN}-chain-flow-tree.patch"


class PatchError(RuntimeError):
    """Anything that must stop the command before it writes.  Always names the file."""


# --------------------------------------------------------------------------- unified diff


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    body: list[tuple[str, str]] = field(default_factory=list)   # (' '|'-'|'+', text)


@dataclass
class FilePatch:
    rel: str                       # path relative to the vllm PACKAGE dir, e.g. v1/spec_decode/x.py
    hunks: list[Hunk]

    @property
    def is_new(self) -> bool:
        """A new file is one whose only hunk replaces the empty file (``@@ -0,0 +1,N @@``)."""
        return len(self.hunks) == 1 and self.hunks[0].old_count == 0 and self.hunks[0].old_start == 0

    def new_content(self) -> str:
        if not self.is_new:
            raise PatchError(f"{self.rel} is not a new file")
        return "\n".join([t for tag, t in self.hunks[0].body if tag == "+"] + [""])


def _hunk_header(line: str) -> Hunk:
    # @@ -<start>[,<count>] +<start>[,<count>] @@ [section]
    core = line.split("@@")[1].strip()
    old, new = core.split(" ")[0], core.split(" ")[1]

    def _pair(s: str) -> tuple[int, int]:
        s = s[1:]
        a, _, b = s.partition(",")
        return int(a), (int(b) if b else 1)

    (os_, oc), (ns, nc) = _pair(old), _pair(new)
    return Hunk(os_, oc, ns, nc)


def parse_patch(text: str) -> list[FilePatch]:
    """Parse the shipped unified diff.

    Hunk bodies are consumed by COUNT, never by scanning for the next ``---``.  That distinction
    is load-bearing: the patch adds three ``.cu`` files whose own text contains lines starting
    with ``---`` and ``+++``, and a scanning parser truncates them silently -- which is the
    "malformed, not broken" failure this whole module exists to prevent.
    """
    lines = text.split("\n")
    out: list[FilePatch] = []
    i, cur = 0, None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            new = lines[i + 1][4:].strip()
            for pfx in ("b/", "a/"):
                if new.startswith(pfx):
                    new = new[2:]
            if not new.startswith("vllm/"):
                raise PatchError(f"patch touches {new!r}, which is outside the vllm package")
            cur = FilePatch(new[len("vllm/"):], [])
            out.append(cur)
            i += 2
            continue
        if ln.startswith("@@"):
            if cur is None:
                raise PatchError(f"hunk header with no file header at patch line {i + 1}")
            h = _hunk_header(ln)
            cur.hunks.append(h)
            i += 1
            seen_old = seen_new = 0
            while seen_old < h.old_count or seen_new < h.new_count:
                if i >= len(lines):
                    raise PatchError(f"patch truncated inside a hunk for {cur.rel}")
                b = lines[i]
                i += 1
                if b.startswith("\\"):          # "\ No newline at end of file"
                    continue
                tag, txt = (b[0], b[1:]) if b else (" ", "")
                if tag not in " -+":
                    raise PatchError(f"bad hunk line {b!r} in {cur.rel}")
                if tag in " -":
                    seen_old += 1
                if tag in " +":
                    seen_new += 1
                h.body.append((tag, txt))
            continue
        i += 1
    if not out:
        raise PatchError("the patch file contains no file sections")
    return out


_FLIP = {"+": "-", "-": "+", " ": " "}


def _apply_hunks(lines: list[str], hunks: list[Hunk], reverse: bool) -> list[str]:
    """Exact-offset, zero-fuzz application.  Raises rather than guessing.

    ``lines`` is ``text.split("\\n")``, so the trailing empty element of a newline-terminated file
    is preserved and ``"\\n".join`` round-trips the file byte-for-byte.
    """
    out: list[str] = []
    pos = 0
    for n, h in enumerate(hunks, 1):
        start = h.new_start if reverse else h.old_start
        idx = max(start - 1, 0)
        if idx < pos:
            raise PatchError(f"hunk {n} overlaps the previous one (line {start})")
        if idx > len(lines):
            raise PatchError(f"hunk {n} starts at line {start}, past end of file ({len(lines)})")
        out.extend(lines[pos:idx])
        pos = idx
        for tag, txt in h.body:
            t = _FLIP[tag] if reverse else tag
            if t in " -":
                if pos >= len(lines) or lines[pos] != txt:
                    got = lines[pos] if pos < len(lines) else "<end of file>"
                    raise PatchError(
                        f"hunk {n} does not match at line {pos + 1}:\n"
                        f"      patch expects: {txt!r}\n"
                        f"      file contains: {got!r}")
                pos += 1
                if t == " ":
                    out.append(txt)
            else:
                out.append(txt)
    out.extend(lines[pos:])
    return out


def transform(text: str, fp: FilePatch, reverse: bool) -> str:
    return "\n".join(_apply_hunks(text.split("\n"), fp.hunks, reverse))


# --------------------------------------------------------------------------- files / hashes


def read(p: Path) -> str:
    """Read WITHOUT universal-newline translation, so a round trip is byte-exact."""
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def write_atomic(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".cf-tree-tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, p)


def b64sha(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def sha_of(p: Path) -> str:
    return b64sha(p.read_bytes())


def drop_pyc(p: Path) -> None:
    """Remove the stale bytecode for a file we just rewrote.

    CPython invalidates on mtime+size and would recompile anyway; this is belt-and-braces for the
    case where a restore lands the same size within the same mtime granularity."""
    c = p.parent / "__pycache__"
    if c.is_dir():
        for f in c.glob(p.stem + ".*.pyc"):
            try:
                f.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- the target


@dataclass
class Target:
    pkg: Path                 # .../site-packages/vllm
    site: Path                # .../site-packages
    dist_info: Path | None
    meta_version: str | None  # from the dist-info directory name / METADATA
    file_version: str | None  # from vllm/_version.py -- i.e. what `vllm.__version__` reads


def locate() -> Target:
    """Find the vLLM the CURRENT interpreter would import.

    ``find_spec`` runs the real import machinery (finders, ``sys.path``, editable-install hooks)
    without executing the package, so it is neither a path guess nor a 20-second torch import.
    """
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception as e:                                  # a broken vllm is a refusal, not a crash
        raise PatchError(f"could not resolve the `vllm` package: {e!r}") from e
    locs = list(spec.submodule_search_locations) if spec is not None else []
    if not locs:
        raise PatchError(
            "no `vllm` package is importable from this interpreter "
            f"({sys.executable}).\n"
            "      The tree patch modifies the vLLM you are going to RUN, so there is nothing to\n"
            f"      patch. Install it first:  pip install 'vllm=={PIN}'")
    pkg = Path(locs[0]).resolve()
    site = pkg.parent
    if spec.origin is None or not (pkg / "__init__.py").is_file():
        # A NAMESPACE package: some directory called `vllm` that is not the installed one.  This
        # happens for real -- this repo has a `vllm/` folder of bench scripts, so running the
        # command from the checkout root resolves to it -- and it must be named, because
        # "vllm.__version__ is None" reads like a broken install rather than a shadowed one.
        raise PatchError(
            f"`import vllm` resolves to {pkg}, which is a DIRECTORY named vllm, not an installed\n"
            "      vLLM (no __init__.py, no dist-info). Something on sys.path -- most likely the\n"
            f"      current directory ({Path.cwd()}) -- is shadowing the real package.\n"
            "      cd somewhere else, or run the command with the interpreter of the venv you\n"
            "      mean to patch:  <venv>/bin/python -m chain_flow.cli tree-patch --status")

    file_version = None
    vf = pkg / "_version.py"
    if vf.is_file():
        ns: dict = {}
        try:
            exec(compile(read(vf), str(vf), "exec"), ns)      # noqa: S102 - a generated 3-line file
            file_version = str(ns.get("__version__"))
        except Exception:                                     # noqa: BLE001 - reporting only
            file_version = None

    dist_info, meta_version = None, None
    cands = sorted(site.glob("vllm-*.dist-info"))
    if cands:
        dist_info = cands[0]
        meta_version = dist_info.name[len("vllm-"):-len(".dist-info")]
    return Target(pkg, site, dist_info, meta_version, file_version)


def check_version(t: Target) -> list[str]:
    """Every reason this install must NOT be patched.  Empty list means the gate passes."""
    bad = []
    if t.file_version != PIN:
        bad.append(f"vllm.__version__ is {t.file_version!r}, and this patch is generated against "
                   f"{PIN} EXACTLY")
    if t.dist_info is None:
        bad.append("no vllm-*.dist-info next to the package -- without its RECORD manifest there "
                   "is no way to verify the files before or after, so this refuses to touch them")
    elif t.meta_version != PIN:
        bad.append(f"the installed distribution is vllm {t.meta_version}, and this patch is "
                   f"generated against {PIN} EXACTLY")
    if (t.file_version and t.meta_version and t.file_version != t.meta_version):
        bad.append(f"vllm.__version__ ({t.file_version}) disagrees with the dist-info "
                   f"({t.meta_version}): the package tree has been swapped underneath its own "
                   f"metadata, and neither number can be trusted")
    return bad


def check_writable(t: Target, rels: list[str]) -> list[str]:
    """Can we actually write every file the plan touches?  Checked UP FRONT, so a system or
    conda-owned install fails before the first write rather than halfway through.

    Permission bits alone are not the question, so this also PROBES each directory by creating
    and removing a real file: ``os.access`` answers about the uid, and says yes to root on a
    read-only mount, inside a container with a read-only layer, or on an immutable image -- all
    of which are exactly how a "just patch site-packages" instruction fails in practice.  Writes
    go through a temp file in the same directory (see ``write_atomic``), so the directory has to
    be writable even for a file that already exists.
    """
    bad, probed = [], set()
    for rel in rels:
        p = t.pkg / rel
        d = p.parent
        if not d.is_dir():
            bad.append(f"{rel}: {d} does not exist")
            continue
        if p.exists() and not os.access(p, os.W_OK):
            bad.append(f"{rel} is not writable")
        if d in probed:
            continue
        probed.add(d)
        try:
            probe = d / ".cf-tree-writetest"
            probe.touch()
            probe.unlink()
        except OSError as e:
            bad.append(f"{rel}: cannot create files in {d} ({e.strerror})")
    return bad


# --------------------------------------------------------------------------- RECORD


def record_entries(t: Target) -> dict[str, str]:
    """``{path relative to site-packages: "sha256=<b64>"}`` for everything under ``vllm/``."""
    if t.dist_info is None:
        raise PatchError("no dist-info: cannot read the RECORD manifest")
    rec = t.dist_info / "RECORD"
    if not rec.is_file():
        raise PatchError(f"{rec} is missing -- the install cannot be verified")
    ents: dict[str, str] = {}
    with open(rec, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[1].startswith("sha256="):
                continue                                   # RECORD's own line has no hash
            if row[0].startswith("vllm/"):
                ents[row[0]] = row[1]
    if not ents:
        raise PatchError(f"{rec} lists no vllm/ files -- refusing to verify against it")
    return ents


@dataclass
class Scan:
    modified: list[str]        # in RECORD, hash differs
    missing: list[str]         # in RECORD, not on disk
    added: list[str]           # on disk under vllm/, not in RECORD
    total: int


def scan(t: Target) -> Scan:
    """Hash every RECORD entry and walk the tree for extras.  ~0.6 s for vLLM's 4510 files.

    ``__pycache__`` is excluded from the extras walk because bytecode is generated, is not in
    RECORD by design, and would otherwise bury the three files that matter under hundreds.
    """
    ents = record_entries(t)
    mod, miss = [], []
    for rel, want in ents.items():
        p = t.site / rel
        if not p.is_file():
            miss.append(rel)
        elif sha_of(p) != want:
            mod.append(rel)
    added = []
    for root, dirs, files in os.walk(t.pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            rel = str((Path(root) / fn).relative_to(t.site))
            if rel not in ents:
                added.append(rel)
    return Scan(sorted(mod), sorted(miss), sorted(added), len(ents))


# --------------------------------------------------------------------------- state


def patch_path() -> Path:
    return Path(__file__).resolve().parent / "patches" / PATCH_NAME


def load_patch() -> list[FilePatch]:
    p = patch_path()
    if not p.is_file():
        raise PatchError(
            f"the tree patch is missing from this install (expected {p}).\n"
            f"      It ships as package data; a checkout has it at "
            f"src/chain_flow/patches/{PATCH_NAME}")
    return parse_patch(read(p))


# per-file verdicts
PRISTINE, PATCHED, UNKNOWN, ABSENT = "pristine", "patched", "UNKNOWN", "absent"


def classify(t: Target, fps: list[FilePatch], ents: dict[str, str]) -> dict[str, str]:
    """What state is each file the patch touches actually in?

    The test for "patched" is not a marker string but a PROOF: reverse-apply the hunks and check
    the result against RECORD.  A file that merely contains our markers -- because someone
    hand-edited it, or applied an older patch with fuzz -- comes back UNKNOWN, which is the whole
    point.  Added files are compared against the exact bytes the patch carries.
    """
    out: dict[str, str] = {}
    for fp in fps:
        p = t.pkg / fp.rel
        if fp.is_new:
            if not p.is_file():
                out[fp.rel] = ABSENT
            else:
                out[fp.rel] = PATCHED if sha_of(p) == b64sha(fp.new_content().encode()) else UNKNOWN
            continue
        want = ents.get("vllm/" + fp.rel)
        if not p.is_file():
            out[fp.rel] = ABSENT
            continue
        if want is None:
            out[fp.rel] = UNKNOWN
            continue
        if sha_of(p) == want:
            out[fp.rel] = PRISTINE
            continue
        try:
            back = transform(read(p), fp, reverse=True)
        except PatchError:
            out[fp.rel] = UNKNOWN
            continue
        out[fp.rel] = PATCHED if b64sha(back.encode()) == want else UNKNOWN
    return out


APPLIED, CLEAN, MIXED = "APPLIED", "NOT APPLIED", "INCONSISTENT"


def overall(verdicts: dict[str, str], fps: list[FilePatch]) -> str:
    new = {fp.rel for fp in fps if fp.is_new}
    ok_applied = all(v == PATCHED for v in verdicts.values())
    ok_clean = all(verdicts[fp.rel] == (ABSENT if fp.rel in new else PRISTINE) for fp in fps)
    if ok_applied:
        return APPLIED
    if ok_clean:
        return CLEAN
    return MIXED


# --------------------------------------------------------------------------- reporting


def _plan_lines(fps: list[FilePatch], verdicts: dict[str, str] | None = None) -> list[str]:
    mod = [fp for fp in fps if not fp.is_new]
    new = [fp for fp in fps if fp.is_new]
    out = [f"  MODIFIES {len(mod)} file(s):"]
    for fp in mod:
        v = f"   [{verdicts[fp.rel]}]" if verdicts else ""
        out.append(f"    vllm/{fp.rel}{v}")
    out.append(f"  ADDS {len(new)} file(s):")
    for fp in new:
        v = f"   [{verdicts[fp.rel]}]" if verdicts else ""
        out.append(f"    vllm/{fp.rel}{v}")
    return out


def _scan_line(s: Scan, expect_mod: int | None, expect_add: int | None) -> str:
    """``None`` for an expectation means "we do not know what this install should look like" --
    which is the honest answer in the INCONSISTENT state, and better than printing a target the
    user should not be trying to hit."""
    if expect_mod is None:
        verdict = "no clean expectation -- see the per-file verdicts below"
    elif (len(s.modified), len(s.added)) == (expect_mod, expect_add):
        verdict = "as expected"
    else:
        verdict = f"EXPECTED {expect_mod} modified / {expect_add} added"
    return (f"RECORD     : {s.total} files hashed -- {len(s.modified)} modified, "
            f"{len(s.missing)} missing, {len(s.added)} added  ({verdict})")


def _fork_report() -> str:
    """Would ``[cf-defaults]`` now offer the tree flags?

    ``defaults.py`` is loaded as a bare FILE on purpose, and for a reason beyond import cost:
    ``defaults.fork()`` MEMOISES its answer in a module global, and ``chain_flow/__init__``
    already called it (via ``apply()``) when this CLI started -- i.e. BEFORE the patch was
    written.  Reusing that module would report ``fork missing`` immediately after a successful
    apply, which is precisely the wrong answer at the one moment the user is looking for
    reassurance.  A fresh module object re-probes the files on disk.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "_cf_defaults_probe", Path(__file__).resolve().parent / "defaults.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        fk = m.fork()
    except Exception as e:                                  # noqa: BLE001 - reporting only
        return f"fork probe : unavailable ({e!r})"
    if fk["present"]:
        return ("fork probe : PRESENT -- the tree flags (gdn_defer, gdn_bv, tree_fused_attn, "
                "tree_fullcg) are\n             now OFFERED; the `[cf-defaults]` line will list "
                "them under ON instead of\n             `fork_missing`.")
    return ("fork probe : NOT PRESENT -- the tree flags will still report `fork_missing`. "
            f"Missing: {', '.join(fk['missing'][:4])}")


# --------------------------------------------------------------------------- commands


def cmd_status(t: Target, fps: list[FilePatch], verbose: bool = True) -> tuple[str, Scan | None]:
    print(f"interpreter: {sys.executable}")
    print(f"vLLM       : {t.pkg}")
    print(f"version    : vllm.__version__={t.file_version}  dist-info={t.meta_version}  "
          f"(this patch requires {PIN} exactly)")
    print(f"patch      : {patch_path()}")
    bad = check_version(t)
    if bad:
        for b in bad:
            print(f"  REFUSED: {b}")
        return MIXED, None
    ents = record_entries(t)
    s = scan(t)
    verdicts = classify(t, fps, ents)
    st = overall(verdicts, fps)
    # What SHOULD the RECORD diff look like in the state we believe we are in?  Printing the
    # expectation next to the count is the difference between a number and a verdict.
    if st == APPLIED:
        e_mod, e_add = sum(1 for f in fps if not f.is_new), sum(1 for f in fps if f.is_new)
    elif st == CLEAN:
        e_mod, e_add = 0, 0
    else:
        e_mod = e_add = None
    print(_scan_line(s, e_mod, e_add))
    if s.modified and verbose:
        for r in s.modified:
            print(f"             modified: {r}")
    if s.missing and verbose:
        for r in s.missing:
            print(f"             MISSING : {r}")
    if s.added and verbose:
        for r in s.added:
            print(f"             added   : {r}")
    print(f"tree patch : {st}")
    if verbose:
        print("\n".join(_plan_lines(fps, verdicts)))
    if st == MIXED:
        print("\n  This install is NEITHER pristine nor cleanly patched. A tree run here can\n"
              "  produce MALFORMED DRAFTS rather than an error, so both --apply and --revert\n"
              "  refuse. Files marked UNKNOWN are neither the stock 0.25.1 file nor that file\n"
              "  plus this patch. Restore them with:\n"
              f"      pip install --force-reinstall --no-deps 'vllm=={PIN}'\n"
              "  and run --apply again.")
    elif st == APPLIED:
        print("\n" + _fork_report())
    return st, s


def cmd_apply(t: Target, fps: list[FilePatch], dry: bool) -> int:
    bad = check_version(t)
    if bad:
        print("[cf] REFUSING to patch this vLLM:", file=sys.stderr)
        for b in bad:
            print(f"     - {b}", file=sys.stderr)
        print(f"     A patch that half-applies to a near-miss version is the worst outcome:\n"
              f"     it does not raise, it drafts wrongly. Get the pinned build with\n"
              f"         pip install --force-reinstall --no-deps 'vllm=={PIN}'", file=sys.stderr)
        return 2

    ents = record_entries(t)
    verdicts = classify(t, fps, ents)
    st = overall(verdicts, fps)
    if st == APPLIED:
        print("[cf] the tree patch is ALREADY APPLIED and verifies against RECORD "
              f"({len(fps)}/{len(fps)} files). Nothing to do.")
        print(_fork_report())
        return 0
    if st == MIXED:
        # Name ONLY the files that are unaccounted for.  Listing the correctly-patched ones as
        # well (an earlier version did) buries the two lines the user has to act on under twelve
        # they do not.
        print("[cf] REFUSING: this vLLM is neither pristine nor cleanly patched.", file=sys.stderr)
        for rel, v in verdicts.items():
            if v == UNKNOWN:
                print(f"     - vllm/{rel}: neither the stock {PIN} file nor that file + this "
                      f"patch", file=sys.stderr)
        part = [r for r, v in verdicts.items() if v == PATCHED]
        if part:
            print(f"     ({len(part)} of {len(fps)} files ARE correctly patched, so this install "
                  f"is half-forked.)", file=sys.stderr)
        print(f"     Restore it first:  pip install --force-reinstall --no-deps 'vllm=={PIN}'",
              file=sys.stderr)
        return 2

    print(f"[cf] target: {t.pkg}  (vllm {t.file_version})")
    print("[cf] this will change your installed vLLM:")
    print("\n".join(_plan_lines(fps)))

    w = check_writable(t, [fp.rel for fp in fps])
    if w:
        print("[cf] REFUSING: the vLLM install is not writable by this user "
              f"(uid {os.getuid()}):", file=sys.stderr)
        for b in w[:8]:
            print(f"     - {b}", file=sys.stderr)
        print("     This is a system or conda-owned install. Either re-run in a virtualenv you\n"
              "     own (python -m venv .venv && .venv/bin/pip install 'vllm==%s'), or\n"
              "     re-run this command as the owner of that tree." % PIN, file=sys.stderr)
        return 2

    # --- build every new file in memory BEFORE writing anything ------------------------
    # `originals` is the rollback image: an I/O failure PART WAY THROUGH the write loop is the
    # one remaining way to end up with a half-fork, and unlike everything above it cannot be
    # refused in advance (a disk fills, a mount goes read-only mid-command).
    staged: dict[str, str] = {}
    originals: dict[str, str | None] = {}
    for fp in fps:
        p = t.pkg / fp.rel
        try:
            originals[fp.rel] = None if fp.is_new else read(p)
            staged[fp.rel] = fp.new_content() if fp.is_new else transform(read(p), fp, reverse=False)
        except PatchError as e:
            print(f"[cf] REFUSING: the patch does not apply cleanly to vllm/{fp.rel}:\n"
                  f"      {e}\n"
                  "      Nothing has been written. This means the file is not the stock 0.25.1\n"
                  "      one even though its hash said otherwise -- report it rather than forcing.",
                  file=sys.stderr)
            return 2

    if dry:
        print(f"[cf] DRY RUN: all {len(fps)} files apply cleanly; nothing was written.")
        print("[cf] re-run without --dry-run to apply.")
        return 0

    done: list[str] = []
    try:
        for rel, text in staged.items():
            p = t.pkg / rel
            write_atomic(p, text)
            drop_pyc(p)
            done.append(rel)
    except OSError as e:
        print(f"[cf] WRITE FAILED on vllm/{rel}: {e}\n"
              f"     Rolling back the {len(done)} file(s) already written...", file=sys.stderr)
        for r in reversed(done):
            p = t.pkg / r
            try:
                if originals[r] is None:
                    p.unlink(missing_ok=True)
                else:
                    write_atomic(p, originals[r])
                drop_pyc(p)
            except OSError as e2:
                print(f"     ROLLBACK FAILED for vllm/{r}: {e2}", file=sys.stderr)
        print(f"     Check with --status; restore with "
              f"pip install --force-reinstall --no-deps 'vllm=={PIN}'", file=sys.stderr)
        return 1
    print(f"[cf] wrote {len(staged)} files.")

    # --- verify AFTER, against RECORD ---------------------------------------------------
    s = scan(t)
    verdicts = classify(t, fps, ents)
    st = overall(verdicts, fps)
    n_mod = sum(1 for f in fps if not f.is_new)
    n_add = sum(1 for f in fps if f.is_new)
    print(_scan_line(s, n_mod, n_add))
    ok = (st == APPLIED and sorted(s.modified) == sorted("vllm/" + f.rel for f in fps if not f.is_new)
          and sorted(s.added) == sorted("vllm/" + f.rel for f in fps if f.is_new)
          and not s.missing)
    if not ok:
        print("[cf] POST-APPLY VERIFICATION FAILED -- the install does not look the way this\n"
              "     patch says it should. Run `chain-flow tree-patch --revert`, then\n"
              f"     pip install --force-reinstall --no-deps 'vllm=={PIN}'.", file=sys.stderr)
        for rel, v in verdicts.items():
            if v != PATCHED:
                print(f"     - vllm/{rel}: {v}", file=sys.stderr)
        return 1

    print(f"[cf] APPLIED and verified: exactly {n_mod} modified + {n_add} added, "
          "every other file byte-identical to the wheel.")
    print(_fork_report())
    print(dedent_block(f"""
        Verify it works:
          chain-flow info                 # expect `vLLM build : FORKED`
          VLLM_SPEC_TREE=1 python -c "from vllm import LLM"   # engine-side import check
        Then run a tree arm -- the `[cf-defaults]` line must show gdn_defer / gdn_bv /
        tree_fused_attn / tree_fullcg under ON (never `fork_missing`):
          ./vllm/bench_cf.sh 4b tree
        num_speculative_tokens must equal CF_TREE_KEEP*CF_TREE_DEPTH + 1, and the tree arm is
        greedy-only. Undo at any time with:
          chain-flow tree-patch --revert"""))
    return 0


def cmd_revert(t: Target, fps: list[FilePatch], dry: bool) -> int:
    if t.dist_info is None:
        print("[cf] no vllm dist-info: cannot verify a revert, refusing.", file=sys.stderr)
        return 2
    ents = record_entries(t)
    verdicts = classify(t, fps, ents)
    st = overall(verdicts, fps)
    if st == CLEAN:
        print("[cf] the tree patch is NOT applied (every file already matches the wheel's "
              "RECORD). Nothing to do.")
        return 0
    if st == MIXED:
        # Revert what is provably ours; refuse if anything is unrecognisable, because a partial
        # revert leaves exactly the half-fork this module exists to prevent.
        print("[cf] REFUSING: some files are neither stock nor cleanly patched, so reverting\n"
              "     them cannot be proved byte-identical:", file=sys.stderr)
        for rel, v in verdicts.items():
            if v == UNKNOWN:
                print(f"     - vllm/{rel}: {v}", file=sys.stderr)
        print(f"     Restore the whole install instead:\n"
              f"         pip install --force-reinstall --no-deps 'vllm=={PIN}'", file=sys.stderr)
        return 2

    mod = [fp for fp in fps if not fp.is_new]
    new = [fp for fp in fps if fp.is_new]
    print(f"[cf] target: {t.pkg}  (vllm {t.file_version})")
    print(f"[cf] this will RESTORE {len(mod)} file(s) and DELETE {len(new)} file(s):")
    print("\n".join(_plan_lines(fps)))

    w = check_writable(t, [fp.rel for fp in fps])
    if w:
        print("[cf] REFUSING: the vLLM install is not writable by this user "
              f"(uid {os.getuid()}):", file=sys.stderr)
        for b in w[:8]:
            print(f"     - {b}", file=sys.stderr)
        return 2

    staged: dict[str, str] = {}
    for fp in mod:
        p = t.pkg / fp.rel
        try:
            back = transform(read(p), fp, reverse=True)
        except PatchError as e:
            print(f"[cf] REFUSING: cannot reverse the patch on vllm/{fp.rel}:\n      {e}",
                  file=sys.stderr)
            return 2
        want = ents.get("vllm/" + fp.rel)
        if b64sha(back.encode()) != want:
            print(f"[cf] REFUSING: reversing vllm/{fp.rel} would NOT restore the wheel's bytes\n"
                  "      (RECORD sha256 mismatch). Nothing written.", file=sys.stderr)
            return 2
        staged[fp.rel] = back

    if dry:
        print(f"[cf] DRY RUN: all {len(mod)} restores verify against RECORD and {len(new)} added "
              "files would be deleted; nothing was written.")
        return 0

    for rel, text in staged.items():
        p = t.pkg / rel
        write_atomic(p, text)
        drop_pyc(p)
    for fp in new:
        p = t.pkg / fp.rel
        if p.is_file():
            drop_pyc(p)
            p.unlink()

    s = scan(t)
    print(_scan_line(s, 0, 0))
    if s.modified or s.added or s.missing:
        print("[cf] REVERT INCOMPLETE -- files still differ from the wheel:", file=sys.stderr)
        for r in (s.modified + s.added + s.missing)[:12]:
            print(f"     - {r}", file=sys.stderr)
        print(f"     Restore with: pip install --force-reinstall --no-deps 'vllm=={PIN}'",
              file=sys.stderr)
        return 1
    print("[cf] REVERTED: this vLLM is byte-identical to the published wheel "
          f"({s.total}/{s.total} files match RECORD).")
    print(_fork_report())
    return 0


def cmd_show() -> int:
    """The manual route.  Kept because a user without this package on the target interpreter
    (a different venv, a container build step) still needs the file and the commands."""
    print(f"patch: {patch_path()}\n")
    print("The tree path is OPTIONAL. It needs a patched vLLM 0.25.1 (tree-aware verify, "
          "tree-shaped GDN\nrecurrence and attention). The default chain build needs none of it "
          "and does not modify vLLM.\n")
    print("The safe way -- verifies against the wheel's RECORD manifest before and after, and is "
          "reversible:\n")
    print("  chain-flow tree-patch --status")
    print("  chain-flow tree-patch --apply")
    print("  chain-flow tree-patch --revert\n")
    print("By hand, into the vLLM install of a DIFFERENT interpreter (no verification):\n")
    print("  cd \"$(python -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')\"")
    print(f"  patch -p1 --dry-run < {patch_path()}")
    print(f"  patch -p1 < {patch_path()}\n")
    print("Then run with VLLM_SPEC_TREE=1 and "
          "num_speculative_tokens = CF_TREE_KEEP*CF_TREE_DEPTH + 1.")
    print("`chain-flow info` reports `vLLM build : FORKED` once it is in.")
    return 0


def dedent_block(s: str) -> str:
    return "\n".join(ln[8:] if ln.startswith("        ") else ln for ln in s.split("\n"))


def main(args) -> int:
    """Exit codes.  ``--status`` is the scriptable one, so it answers the QUESTION in its code:
    0 applied, 1 not applied, 3 inconsistent.  The bare command is informational and always 0.
    ``--apply`` / ``--revert``: 0 done (or already in that state), 1 it went wrong AFTER writing,
    2 refused before writing anything."""
    try:
        fps = load_patch()
    except PatchError as e:
        print(f"[cf] {e}", file=sys.stderr)
        return 2
    if getattr(args, "show", False):
        return cmd_show()
    try:
        t = locate()
    except PatchError as e:
        print(f"[cf] {e}", file=sys.stderr)
        return 2
    dry = bool(getattr(args, "dry_run", False))
    if getattr(args, "apply", False):
        return cmd_apply(t, fps, dry)
    if getattr(args, "revert", False):
        return cmd_revert(t, fps, dry)
    st, s = cmd_status(t, fps)
    if getattr(args, "status", False):
        return {APPLIED: 0, CLEAN: 1, MIXED: 3}[st]
    if s is not None:                       # None means the version gate already refused: the
        print("\nApply it with:  chain-flow tree-patch --apply "   # last thing that install
              "   (--dry-run to see it without writing)")            # needs is an invitation
        print("Undo it with :  chain-flow tree-patch --revert")
    print("The patch file itself: chain-flow tree-patch --show")
    return 0
