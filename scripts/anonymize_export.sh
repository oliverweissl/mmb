#!/usr/bin/env bash
# Clean, double-blind copy of this repository for anonymous.4open.science (see ARTIFACT.md).
#
#   scripts/anonymize_export.sh [OUT_DIR]          (default: build-artifact/mmb-anon)
#
# 1. scripts/make_artifact.sh builds the bundle: files git tracks or would add, no history; excluded are
#    runs/ (29 GB of run stores), .seedbank, .mycelium, .claude, AGENTS.md/CLAUDE.md, PLAN/PROGRESS.md, the
#    machine-specific launch scripts (scripts/_pilot*, scripts/replay_maintenance.sh), images, weights and
#    files above 8 MB. It rewrites names, e-mails, git-host user names and home paths and fails on leftovers.
# 2. Second pass on the copy: generic anonymous licence preambles (terms unchanged), neutral wording for
#    mentions of the earlier version of the work, SMOO cited as "anonymized for review", institution names.
# 3. Final gate: grep over every file for thesis (not 'synthesis'/'hypothesis')|oliver|weissl|tum.de|Technical University|Munich|/home/.
# The working tree is never modified. Exit 1 if anything identifying remains.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build-artifact/mmb-anon}"
"$ROOT/scripts/make_artifact.sh" --out "$OUT" --results "$ROOT/results" --max-file-mb "${MAX_FILE_MB:-8}"

python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])

# (a) licence preambles: generic and anonymous (no title, no institution); the licence terms are unchanged
CODE_PRE = """Copyright (c) 2026 Anonymous (identity withheld for double-blind review)

Required Notice: Copyright (c) 2026 Anonymous

The source code and configuration files of this artifact are licensed under the PolyForm
Noncommercial License 1.0.0, reproduced in full below: anyone may use, modify and redistribute
them for any noncommercial purpose, including academic research and teaching.

Documentation, derived data and figures are licensed under CC BY-NC 4.0 (see LICENSE-DATA).
Third-party code, datasets and model weights keep their own terms (see THIRD-PARTY-NOTICES.md).

If you use this artifact, please cite the accompanying paper.
"""
DATA_TXT = """Copyright (c) 2026 Anonymous (identity withheld for double-blind review)

# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

The documentation, notebooks, result tables (CSV/TeX) and figures of this artifact are licensed
under CC BY-NC 4.0.

  SPDX-License-Identifier: CC-BY-NC-4.0
  Human-readable summary: https://creativecommons.org/licenses/by-nc/4.0/
  Full legal code:        https://creativecommons.org/licenses/by-nc/4.0/legalcode

You may share and adapt this material for any noncommercial purpose, provided you give appropriate
credit (cite the accompanying paper), link to the license, and indicate whether changes were made.
The source code is licensed separately (see LICENSE); third-party material keeps its own terms
(see THIRD-PARTY-NOTICES.md).
"""
lic = root / "LICENSE"
if lic.exists():
    body = lic.read_text(encoding="utf-8")
    i = body.find("# PolyForm Noncommercial License 1.0.0")
    if i < 0:
        sys.exit("LICENSE: PolyForm text not found")
    lic.write_text(CODE_PRE + "\n---\n\n" + body[i:])
(root / "LICENSE-DATA").write_text(DATA_TXT)

# (b) neutral wording for mentions of the earlier version of the work; (c) SMOO cited without user name
SUBS = [
    (r"Thesis hypothesis \(07_discussion\)", "Hypothesis of the earlier prototype"),
    (r"\bthe thesis'(?:s)?", "the earlier prototype's"),
    (r"\bthesis'(?:s)?", "earlier prototype's"),
    (r"\bthe thesis\b", "the earlier prototype"),
    (r"(?i)\bthesis\b", "earlier prototype"),
    (r"(?i)_thesis_", "_prototype_"),
    (r"\((?:\x60)?[\w-]+/SMOO(?:\x60)?\)", "(anonymized for review)"),
    (r"(?i)Technical University of Munich|Technische Universit(?:ä|ae)t M(?:ü|ue)nchen", "Anonymous institution"),
    (r"\bTUM\b", "Anonymous institution"),
]
changed = 0
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.suffix == ".zip" or p.name in ("LICENSE", "LICENSE-DATA"):
        continue
    try:
        t = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    new = t
    for pat, rep in SUBS:
        new = re.sub(pat, rep, new)
    if new != t:
        p.write_text(new); changed += 1
print(f"second pass: licences replaced, {changed} files reworded")
PY

# final gate: fails on any identifying string, in text and binary files alike
GATE='(^|[^a-z])thesis|oliver|weissl|tum\.de|Technical University|Munich|/home/'
if hits=$(grep -rIaliE "$GATE" "$OUT"); then
  echo "GATE FAIL:"; grep -rIaniE "$GATE" "$OUT" | cut -c1-200 | head -40; exit 1
fi
echo "gate OK: no match for /$GATE/ (case-insensitive) in $(find "$OUT" -type f | wc -l) files"
(cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")")
echo "upload: $OUT (archive $OUT.zip) to anonymous.4open.science"
