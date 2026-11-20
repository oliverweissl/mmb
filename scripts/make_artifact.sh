#!/usr/bin/env bash
# Build the anonymous artifact bundle (see ARTIFACT.md). History-free, path-free, name-free.
#
#   scripts/make_artifact.sh [--out DIR] [--runs RUNS_DIR] [--results RESULTS_DIR] [--max-file-mb N]
#
#   --out         bundle directory (default: build-artifact/mmb-artifact; a .zip is written next to it)
#   --runs        include run stores from this directory (trace/convergence parquet, stats and first_cross
#                 json only; images are never copied)
#   --results     include results CSVs, tables and figures from this directory (default: ./results if present)
#   --max-file-mb files above this size are NOT copied and are listed in MANIFEST.txt (default 8; an
#                 assumption about the hosting service, confirm the current limit)
#
# What it does: copies the files git would track or would add (so ignored files, .git, venvs, caches and
# model weights stay out), excludes agent/tooling directories and every image, rewrites author names,
# e-mail addresses, absolute paths and git-host user names in text files, then scans the bundle for
# forbidden strings and exits 1 if any remain. The working tree is never modified.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build-artifact/mmb-artifact"; RUNS=""; RESULTS=""; MAXMB=8
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --runs) RUNS="$2"; shift 2;;
    --results) RESULTS="$2"; shift 2;;
    --max-file-mb) MAXMB="$2"; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -z "$RESULTS" ] && [ -d "$ROOT/results" ] && RESULTS="$ROOT/results"

rm -rf "$OUT"; mkdir -p "$OUT"
cd "$ROOT"

# 1. files: tracked + untracked-but-not-ignored, minus exclusions (no history is copied)
git ls-files --cached --others --exclude-standard -z \
  | grep -zv -E '(^|/)(\.git|\.venv|__pycache__|\.seedbank|\.mycelium|\.claude|build-artifact|runs)(/|$)|^(AGENTS|CLAUDE)\.md$|^scripts/make_artifact\.sh$|\.egg-info/' \
  | grep -zv -E '\.(png|jpg|jpeg|gif|webp|bmp|tif|tiff|pt|pth|ckpt|safetensors|bin|onnx|pkl|npy|npz)$' \
  | while IFS= read -r -d '' f; do
      [ -f "$f" ] || continue
      mkdir -p "$OUT/$(dirname "$f")"; cp -p "$f" "$OUT/$f"
    done

# 2. optional results and run stores (data files only; images are derived from licensed photographs)
if [ -n "$RESULTS" ] && [ -d "$RESULTS" ]; then
  (cd "$RESULTS" && find . -type f \( -name '*.csv' -o -name '*.tex' -o -name '*.pdf' -o -name '*.json' -o -name '*.txt' \) -print0) \
    | while IFS= read -r -d '' f; do mkdir -p "$OUT/results/$(dirname "$f")"; cp -p "$RESULTS/$f" "$OUT/results/$f"; done
fi
if [ -n "$RUNS" ] && [ -d "$RUNS" ]; then
  (cd "$RUNS" && find . -type f \( -name '*.parquet' -o -name '*.json' -o -name '*.csv' \) -print0) \
    | while IFS= read -r -d '' f; do mkdir -p "$OUT/runs/$(dirname "$f")"; cp -p "$RUNS/$f" "$OUT/runs/$f"; done
fi

# 3. drop oversize files, record them
: > "$OUT/.oversize"
find "$OUT" -type f -size +"${MAXMB}"M -print0 | while IFS= read -r -d '' f; do
  echo "${f#$OUT/} ($(du -m "$f" | cut -f1) MB)" >> "$OUT/.oversize"; rm -f "$f"
done

# 4. anonymize text files and scan for leftovers
python3 - "$OUT" "$USER" "$(hostname)" <<'PY'
import re, sys
from pathlib import Path

root, user, host = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SUBS = [  # applied in order
    (r"Ruben\.Kaiser@[\w.-]+", "anonymous@example.org"),
    (r"[\w.+-]+@(?:[\w-]+\.)*(?:tum\.de|fortiss\.org|outlook\.com|gmail\.com)", "anonymous@example.org"),
    (r"(?i)KaiserRuben", "anonymous"),
    (r"(?i)oliverweissl", "anonymous"),
    (r"Ruben Kaiser", "Anonymous"),
    (r"/home/[\w.-]+/[^\s'\"):,;]*/\.venv/bin/python", "python"),
    (r"/(?:home|Users)/[\w.-]+/[^\s'\"):,;]*", "<path>"),
    (r"(?i)\bRuben\b|\bKaiser\b", "Anonymous"),
    (r"(?i)wei(?:ss|\xdf)l", "Anonymous"),
]
FORBIDDEN = [r"(?i)ruben", r"(?i)kaiser", r"(?i)oliver", r"(?i)wei(?:ss|\xdf)l", r"(?i)fortiss", r"\btum\.de\b",
             r"/home/", r"/Users/", r"(?i)sharelatex", r"(?i)stocco", re.escape(host) if host else r"$^"]
if user and user not in ("root",):
    FORBIDDEN.append(r"\b" + re.escape(user) + r"\b")
changed, bad = 0, []
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.name in (".oversize",):
        continue
    try:
        t = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        raw = p.read_bytes()                      # binary (parquet, pdf): scan only
        for pat in FORBIDDEN:
            if re.search(pat.encode("utf-8", "ignore"), raw):
                bad.append((str(p.relative_to(root)), pat, "binary"))
        continue
    new = t
    for pat, rep in SUBS:
        new = re.sub(pat, rep, new)
    if new != t:
        p.write_text(new); changed += 1
    for pat in FORBIDDEN:
        m = re.search(pat, new)
        if m:
            bad.append((str(p.relative_to(root)), pat, new[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")))
print(f"rewrote {changed} text files")
if bad:
    for b in bad[:60]:
        print("LEFTOVER", *b)
    print(f"FAIL: {len(bad)} forbidden string(s) remain")
    sys.exit(1)
print("scan OK: no forbidden strings")
PY

# 5. manifest and archive
{
  echo "# MANIFEST (sha256, path)"
  (cd "$OUT" && find . -type f ! -name MANIFEST.txt ! -name .oversize -print0 | sort -z | xargs -0 sha256sum)
  if [ -s "$OUT/.oversize" ]; then echo; echo "# NOT INCLUDED (above ${MAXMB} MB):"; cat "$OUT/.oversize"; fi
} > "$OUT/MANIFEST.txt"
rm -f "$OUT/.oversize"
(cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")")
echo "bundle: $OUT ($(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1)); archive: $OUT.zip"
