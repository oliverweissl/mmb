# Artifact plan (anonymous review bundle)

Goal: a double-blind-safe copy of this repository that a reviewer can read, run on stubs, and use to regenerate every table and figure from the released results. It is built by `scripts/make_artifact.sh` and published on an anonymising host (anonymous.4open.science or equivalent) while the paper is under review. A DOI-bearing archive (Zenodo or similar) is created only after acceptance, because it would reveal the authors.

## Contents

| included | notes |
|---|---|
| `mmb/`, `scripts/`, `tests/`, `pyproject.toml` | source code and its tests |
| `configs/` | one configuration per arm and experiment; `scripts/make_matrix.py` regenerates the matrix |
| `cells/` | fixed cell lists as identifiers (dataset, item id, class pair, anchor), never images |
| `METHODOLOGY.md`, `EXP_SETUP.md`, `README.md`, `ARTIFACT.md`, `docs/` | decisions, recipe, entry points |
| `LICENSE`, `LICENSE-DATA`, `THIRD-PARTY-NOTICES.md` | with the copyright holder and contact rewritten to "Anonymous" |
| `results/` (with `--results`) | per-run CSVs of every experiment, the LaTeX tables and the figure PDFs; every number in the paper is traceable to one of these files |
| `runs/` (with `--runs`) | run stores: `trace.parquet`, `convergence.parquet`, `stats.json`, `first_cross.json`. Included only if the total fits the host's limits; otherwise the per-run CSVs stand in, and the stores follow with the archival copy |
| `MANIFEST.txt` | sha256 of every file, plus the files left out for size |

## Excluded

- **Images of any kind**: ImageNet and COCO photographs and everything derived from them (`origin.png`, `first_cross.png`, `pareto_*.png`) are not redistributable; genotypes in the traces regenerate the edited images from the seeds. The scripts that fetch the data need the item identifiers only.
- **Model weights and caches** (`*.pt`, `*.pth`, `*.safetensors`, `*.ckpt`, Hugging Face and Redis caches). Weights are listed in EXP_SETUP.md with their identifiers; versions and licences go into the paper's artifact statement.
- **Git history**, `.git`, virtual environments, egg-info, `__pycache__`, agent and editor tooling directories, build output.
- **Files above the size limit** (default 8 MB, an assumption about the host; confirm the current limit and set `--max-file-mb`). They are named in `MANIFEST.txt`.
- **Human-study material**: the HS-01 ratings and stimuli are not part of this bundle. They are released, if at all, as aggregated ordinal data after the ethics question is settled (see the paper repository's checklist).

## Anonymisation

`scripts/make_artifact.sh` copies files (no history), then rewrites in text files: author names, e-mail addresses, git-host user names, absolute home paths (including virtual-environment interpreters), and scans the whole bundle, binaries included, for a forbidden list (names, affiliations, user and host name of the building machine, home-directory path prefixes, hosting domains). It exits 1 with the offending files if anything remains. Limits: strings inside compressed parquet columns cannot be scanned, and the script cannot recognise identifying free text such as a sentence that names a group. Read `MANIFEST.txt` and the rewritten licence files once before uploading.

## Build

```bash
git status                      # commit or stash: untracked but non-ignored files ARE included
scripts/make_artifact.sh --results results --runs runs/E1 --max-file-mb 8
# -> build-artifact/mmb-artifact/ and build-artifact/mmb-artifact.zip
```

Upload the zip contents to the anonymising host and put its URL into the paper's data-availability statement (the placeholder is in the appendix of the paper).

## Review checklist

- [ ] `scripts/make_artifact.sh` ends with "scan OK".
- [ ] `pytest -q` passes inside the unpacked bundle (the model-backed validity tests skip themselves without cached models).
- [ ] `scripts/analysis/run_analysis.sh` regenerates the paper's tables from `results/` or `runs/`.
- [ ] `THIRD-PARTY-NOTICES.md` matches what the bundle contains (the inherited text still describes material of an earlier layout and needs a refresh before submission).
- [ ] The paper's artifact statement names exact checkpoints, hashes and licences, and the data-availability statement carries the anonymous URL.
