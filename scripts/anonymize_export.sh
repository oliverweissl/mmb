#!/usr/bin/env bash
# Clean, double-blind copy of this repository for anonymous.4open.science (see ARTIFACT.md).
#
#   scripts/anonymize_export.sh [OUT_DIR]          (default: build-artifact/mmb-anon)
#
# 1. scripts/make_artifact.sh builds the bundle: files git tracks or would add, no history; excluded are
#    runs/ (29 GB of run stores), .seedbank, .mycelium, .claude, AGENTS.md/CLAUDE.md, PLAN/PROGRESS.md, the
#    machine-specific launch scripts (scripts/_pilot*, scripts/replay_maintenance.sh), images, weights and
#    files above 8 MB. It rewrites names, e-mails, git-host user names and home paths and fails on leftovers.
# 2. This wrapper adds a second pass: institution names in the licence files become "Anonymous institution",
#    the forbidden list is extended (institution, workspace path names), and lines that refer to earlier
#    work of the authors ("thesis") are listed for manual review (not changed automatically).
# The working tree is never modified. Exit 1 if anything identifying remains.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build-artifact/mmb-anon}"
"$ROOT/scripts/make_artifact.sh" --out "$OUT" --results "$ROOT/results" --max-file-mb "${MAX_FILE_MB:-8}"

python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
SUBS = [(r"(?i)Technical University of Munich|Technische Universit(?:ä|ae)t M(?:ü|ue)nchen", "Anonymous institution"),
        (r"\bTUM\b", "Anonymous institution")]
FORBIDDEN = [r"(?i)munich|m(?:ü|ue)nchen", r"(?i)technical university", r"\bTUM\b", r"(?i)gardens/",
             r"(?i)github\.com/(?!explosion|huggingface|pytorch)[\w-]+/(?:mmb|smoo)"]
REVIEW = re.compile(r"(?i)\bthesis\b")
bad, review = [], []
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.suffix in (".zip",):
        continue
    try:
        t = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    new = t
    if p.name.startswith("LICENSE") or p.name == "THIRD-PARTY-NOTICES.md":
        for pat, rep in SUBS:
            new = re.sub(pat, rep, new)
    if new != t:
        p.write_text(new)
    rel = str(p.relative_to(root))
    for pat in FORBIDDEN:
        m = re.search(pat, new)
        if m:
            bad.append((rel, new[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")))
    for i, line in enumerate(new.split("\n"), 1):
        if REVIEW.search(line):
            review.append(f"{rel}:{i}")
if review:
    print(f"REVIEW ({len(review)} lines mention a thesis; check they do not identify the authors):")
    for r in review[:40]:
        print("  ", r)
if bad:
    for b in bad:
        print("LEFTOVER", *b)
    print(f"FAIL: {len(bad)} identifying string(s) remain"); sys.exit(1)
print("second pass OK")
PY
(cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")")
echo "upload: $OUT (archive $OUT.zip) to anonymous.4open.science"
