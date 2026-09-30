# Runner contract (9 tools)

Every runner in `src/runners/` follows these rules so the harness
(`src/backend/local.py`), the analysis (`src/analysis/analyze_run.py`)
and the tests (`tests/test_runners_contract.py`) can treat them uniformly.
Deliberately a document, not a base class: each runner wraps a different
external tool and stays a plain script.

## 1. CLI flags

Required: `--pos POS.fasta --neg NEG.fasta -o OUT`
(`-o` accepts a `.tsv`/`.csv` file or a directory).
`--pos`/`--neg` are optional-but-at-least-one in `mldspp`, `prokbert_mini`,
`ipromp_sp12`; required in the other six.
Optional where applicable: `--batch-size N`, `--device` (prokbert only;
lcnn/ipromp use env/`gpu_id` instead), model paths (`--model` lcnn,
`--model-path` prompt, `--db` fimo, `--split` mldspp_75, `--species`
ipromp, `--timeout` promotech).
`--help` must exit 0 using the runner's own pixi interpreter.

## 2. Output files

Written under OUT, exact names (loaders in `analyze_run.py` address
them literally — renaming breaks scoring):

| Loader | Files (relative to OUT) |
|---|---|
| `posneg_dir` (mldspp, mldspp_75, promotech, lcnn) | `{stem}_pos.csv` + `{stem}_neg.csv`, single `PRED` column, TSV, row i = sequence i of the input FASTA |
| `single_tsv` (prompt, prokbert) | file mode: one TSV with `ID LABEL PRED`; dir mode: `{tool}_pos.csv`/`{tool}_neg.csv` + `{tool}.tsv` |
| `ipromp_dir` | subdir `ipromp/` + single `PRED` CSV |
| `fimo_dir` | `fimo_prok_pos.csv` + `fimo_prok_neg.csv` |
| `meme_dir` | `meme_pos.csv` + `meme_neg.csv` |

Stems: `mldspp`, `mldspp_75spn`, `lcnn/lcnn_*`, PromoTech via
`promotech/workdir/*/sequences_predictions.csv`.

## 3. Timing line (stdout, last line)

```
<TOOL>: <N> seqs (train <X.XXX>s / infer <Y.YYY>s)
```

`train` = model fitting / motif discovery (STREME, XGBoost fit);
`infer` = scoring the test sequences only. Only runners with a train
phase (meme, mldspp, mldspp_75) print the split; pure-inference runners
print `<TOOL>: N seqs in Xs` (decorations like `(Pos/Neg)` or `[batch=..]`
allowed — the harness matches by regex, see `local._parse_train_infer`).
Runners without a split record `train_s=0.0`, `infer_s=wall_seconds`.

Start-point convention: `elapsed` runs from the first compute op
(post-load, post-parse) to the last score; loading, parsing and saving
are excluded (see RUNNING.md methodology). MLDSPP test featurization
counts as infer; FIMO per-chunk DB reloads count as infer (natural).

## 4. Seeds (all 42, all isolated)

| Runner | Mechanism |
|---|---|
| meme | `random.Random(42)` local instance + `streme -seed 42` |
| mldspp / mldspp_75 | `np.random.RandomState(42)` (local / module-level object) + `random_state=42` in `MLDSPP_XGB_PARAMS` (`_shared.py`) |
| lcnn, ipromp_sp12, prokbert_mini, prompt, promotech_hot, fimo | no RNG (pretrained inference / deterministic scan) |

Rule: never mutate global random state (`random.seed`, `np.random.seed`
are banned in `src/runners/`).

## 5. Failures

Exit code != 0, no half-written TSVs. Constant-score outputs are
reported downstream as `BROKEN` by `analyze_run.py` (no silent 0.500).

## 6. Provenance

The harness appends one line per tool to `<run>/RUN.log`
(command context, env, input sha, timings, exit). Runners keep
their TSVs metadata-free (loaders read them strictly).
