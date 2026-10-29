# V2_STATUS: overnight build of the v2 telecom pipeline (30 Oct 2026)

Branch `v2`, made from `main` at `56fe587`. Code and checks only: no training, no model inference, no sudo, no push.
Morning steps are in `V2_RUNBOOK.md`; the rules are in `PLAN_V2.md`, committed first (`d589c09`) before any code.

## Decide or do in the morning

1. **Push the branch yourself**: `git push -u origin v2`. The v2 notebook clones GitHub and runs `git switch v2`, so it needs the branch there.
2. **`~/fedflow_data/val_v1/` did not exist** when I listed `~/fedflow_data/` (entries: `local_val`, `telecom_cache`, `telecom_check.txt`,
   `telecom_raw`, `venv_gpu`). I did not look inside `local_val`. `compare_variants.py` needs v1's `val_metrics.json` and
   `val_forecasts.csv`. Point it at wherever they really are (runbook step 0.2 and step 6).
3. **Truncation abort is v2-only.** The truncation check (client and server preflight) runs only with `FEDFLOW_VARIANT=v2`, so the v1 default is
   unchanged. Measured: no v1 example is over 256 tokens (longest 256, see the token audit below), so v1 never truncated.
   Say if you want the check on for v1 too.
4. **`run_config.json` is written only by v2 runs.** A v1 run writes exactly the files it always did. `infer.py` treats a missing file as v1.
5. **`"variant"` is now always in metrics.json / val_metrics.json** when scored with this code, v1 included, as you asked. Scores made with
   the old v1 code have no such key; `compare_variants.py` reads that as v1.
6. **Rounding of the onset target.** Half of a cell's examples is rounded **up** for odd counts (33 examples -> 17 onset windows). Change before training if you
   want rounding down.
7. **Two report files are not committed.** `build_report.txt` and `token_audit_v2.txt` sit in `training/data/telecom_v2/`, which the
   existing `.gitignore` rule `data/` excludes. Their full text is copied below. Say if you want them force-added.
8. **v2 example order within a cell is shuffled** (onset and other windows mixed with the cell's own seeded generator), so batches are not
   all one kind. v1's order is unchanged.
9. **Server start message.** `server.py` now prints `variant ... max_length ...` in its start line. That is stdout only; no output file changes for v1.

## Rule check

| Rule | Status |
|---|---|
| No sudo, no push, no training | Kept |
| Never run infer.py with a model, never touch the GPU | infer.py only ran as `--selftest` and in a test harness that replaces the model with a stand-in (repeat the last value), in the scratchpad, with `CUDA_VISIBLE_DEVICES=""`. **One exception:** at the very start I ran `nvidia-smi --query-gpu=name,utilization.gpu --format=csv` (a read-only query; it printed `NVIDIA GeForce RTX 4060 Laptop GPU, 50 %`). Nothing else touched the GPU. |
| Never read/write/delete in `~/fedflow_data/local_val/` or `~/fedflow_data/val_v1/` | Kept. I only listed the names in `~/fedflow_data/` and ran `~/fedflow_data/venv_gpu/bin/python -m pip list` / `--version`. |
| No change in `results/`, `datasets/`, `training/data/telecom/` | Kept. `git diff main..v2 --stat` touches neither folder, and the four files in `training/data/telecom/` pass `sha256sum -c` (below). The untracked `results/final_telecom/outputs/` is untouched. |
| Branch `v2`, small commits | 9 commits (`git log --oneline main..v2`), listed below |
| v1 byte-for-byte reproducible | Proven below: synthetic prints Mean 92.59 / Max 233.26; telecom default rewrites identical files |
| Every number from a command | All numbers below were printed by the commands listed |

## What was done

| Step | Done | Commit |
|---|---|---|
| 1 PLAN_V2.md | Yes, before any code | `d589c09` |
| 2 Data: `generate_data.py --dataset telecom --variant v2`, shared `build_example`, time line, onset-balanced sampling, `FEDFLOW_N_EXAMPLES`, build report; `check_split.py --variant v2` | Yes | `83693e7` |
| 3 Loss and length: `FEDFLOW_VARIANT`, answer-only labels, `FEDFLOW_MAX_LENGTH`, truncation abort, token audit, label selftest, `run_config.json` | Yes | `12bbd5b` |
| 4 Inference: variant from `run_config.json`, prompts through `build_example`, mismatch abort, variant in metrics, selftests | Yes | `e4aa405` |
| 5 `compare_variants.py` with the four rules and a selftest for both outcomes | Yes | `fb7ffc2` |
| 6 `--forecasts` and `--out-root` in `run_experiment.py` and `summarize.py` | Yes | `f1f07cd` |
| 7 `notebooks/fedflow_colab_telecom_v2.ipynb` | Yes | `49c4acd` |
| 8 `V2_RUNBOOK.md` | Yes | `009a21a` |
| 9 Checks and this file | Yes | this commit |

Nothing was skipped.

### Details worth knowing

- **Time line.** `Time of last value: <weekday> <HH:MM> (working day|non-working day)` goes on the line right after `History: ...`. It is
  computed from the slot only: day = slot // 144, weekday counted from Friday (slot 0), and minutes = (slot % 144) x 10. Day 0
  (1 Nov, holiday), Saturdays and Sundays are non-working. Spot checks printed by `time_of_slot`:
  - 0: Friday 00:00 (non-working)
  - 143: Friday 23:50 (non-working)
  - 144: Saturday 00:00 (non-working)
  - 432: Monday 00:00 (working)
  - 1439: Sunday 23:50 (non-working)
  - 1451: Monday 01:50 (working)
  - 1739: Wednesday 01:50 (working)
  - 2015: Thursday 23:50 (working)
- **v1 prompt unchanged.** `infer.py --selftest` compares v1's prompt with the exact text produced by the unmodified `build_instruction`,
  and checks that the v2 prompt equals it once the time line is removed.
- **Answer-only labels** (`training/token_utils.py`, torch-free so it can run with the tokenizer alone). It tokenizes the full text once,
  as v1 does, and checks that the prompt tokens alone equal the start of the full text; otherwise it stops rather than mask wrongly.
  Prompt and padding positions are labelled -100. Selftest on the first real v2 example at length 320: 173 prompt + 67 padding positions
  masked, and 79 answer tokens + 1 end token kept. Decoding the kept tokens gives the answer text exactly.
- **Variant tags.** v2 validation/test windows carry `"variant": "v2"`; v1 files have no tag. `infer.py` stops before loading the model
  if the adapter (from `run_config.json`) and the windows disagree. The stand-in harness checked all four pairings: v1/v1 ran, v2/v2 ran
  with the time line in the prompt, and both mismatches stopped with the ERROR message.
- **`compare_variants.py`.** A rule whose input is missing counts as failed, so a broken file can never give ADOPT. It refuses two runs
  that scored different windows, and a v2 file whose metrics don't say v2. It never overwrites an existing report.
  - The selftest covers ADOPT exactly at +5.00 points, DO NOT ADOPT at +10.00 points, DO NOT ADOPT on equal MAE plus more parse
    failures, and both refusals.
  - An end-to-end run on stand-in outputs for all 276 validation windows produced a full report. Those stand-in numbers are not results
    and are not quoted anywhere.
- **Experiments.** A non-default `--forecasts` without its own `--out-root` is refused, so v2 forecasts can never write into
  `results/experiment_telecom/`.

## Data build report (training/data/telecom_v2/build_report.txt, 200 examples)

Printed by `python3 generate_data.py --dataset telecom --variant v2` in `training/`:
```
Total training examples: 200 (FEDFLOW_N_EXAMPLES, default 200)
Onset window: last history value <= 100 and max of the 6 future values > 100.
Onset windows in the training pool (starts 0-1422, all cells): 1099 (abort below 60)
Onset target per cell = half the cell's examples, rounded up. Shortfall = target - onset windows taken.

cell square examples pool onsets target v2 onsets shortfall v1 onsets
   0   9721       34          78     17        17         0         1
   1   6997       34         237     17        17         0         7
   2   8336       33         219     17        17         0         7
   3   7511       33         298     17        17         0         3
   4   8062       33         175     17        17         0         2
   5   7584       33          92     17        17         0         4
 all             200        1099    102       102         0        24

Validation windows: 276, test windows: 366 (same starts, history and truth as v1; asserted).
```
With `FEDFLOW_N_EXAMPLES=400`, built in the scratch copy only (the repo's `telecom_v2/` holds the 200 build):
```
cell square examples pool onsets target v2 onsets shortfall v1 onsets
   0   9721       67          78     34        34         0         1
   1   6997       67         237     34        34         0         7
   2   8336       67         219     34        34         0         7
   3   7511       67         298     34        34         0         3
   4   8062       66         175     33        33         0         2
   5   7584       66          92     33        33         0         4
 all             400        1099    202       202         0        24
```
**Onset shortfalls: 0 in every cell, for both 200 and 400.**

## Token audit (training/data/telecom_v2/token_audit_v2.txt)

Printed by `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python3 token_audit_v2.py` (tokenizer only, from the local Hugging Face cache):
```
Tokenizer: Qwen/Qwen2.5-1.5B-Instruct, transformers 4.46.3, eos '<|im_end|>', pad '<|endoftext|>'
set                 text        n   min   max
v2 train (200)      full      200   252   272
v2 train (400)      full      400   252   272
v2 validation       prompt    276   173   185
v2 validation       full      276   252   272
v2 test             prompt    366   173   185
v2 test             full      366   252   271
v1 telecom train    full      200   236   256
v1 synthetic train  full      200   236   255

Longest v2 text measured (any row starting 'v2'): 272 tokens
Rule: smallest multiple of 32 that is >= longest + 16 = 288
v2 MAX_LENGTH: 288  (set FEDFLOW_MAX_LENGTH=288)
Longest v1 text measured: 256 tokens; v1 limit 256; v1 examples over 256: 0
```
The notebook uses `FEDFLOW_MAX_LENGTH=288` for both attempts.

## v1 reproducibility proof

Baseline before any change: the unmodified code at `56fe587`, exported with `git archive` to the scratchpad and run there.
- `python3 generate_data.py` printed `Mean: 92.59` and `Max: 233.26`.
- `python3 generate_data.py --dataset telecom` wrote files whose sha256 matched `training/data/telecom/` (4 x `OK`).

Final code: a copy of the `v2` working tree, synced to the scratchpad and run there, so `training/data/` in the repo was never rewritten.
- `python3 generate_data.py` printed `Mean: 92.59` and `Max: 233.26`, and `python3 check_split.py` printed `PASS`.
- `python3 generate_data.py --dataset telecom`: `sha256sum -c` against `training/data/telecom/` gave 4 x `OK`.
  `check_split.py --dataset telecom` printed `PASS`, and no `telecom_v2/` folder was created.
- In the repo, `sha256sum -c` on `training/data/telecom/` gave 4 x `OK`: untouched.

v1 telecom checksums (`training/data/telecom/`):
```
83b504657c8761f00751263fe4459e2b609d0dc8ca9aae96433c95aa9de7b068  test_windows.json
96398f37c344d706fd79695bb81859353074147c9fb1802d23ebf58a342cbb58  traffic_train.json
c5eef05022a4627c9d407c17e10f718d4935c502ec43524287754017481378f9  train_starts.json
497044719a94242f3348726f82351dd89e39c06d7929c465363bfdc9ad70c16f  val_windows.json
```
v2 files in the repo (`training/data/telecom_v2/`, 200-example build, not committed):
```
22f72e5eceb85c2e8354f2dd2589078c90fbe828651920dbe02fbac1e9f0cb0d  test_windows.json
7047afa99f49aac00022b25d321d4a83346425b019848d069234d0351045744c  traffic_train.json
24c5b8a6ad0c2c5fa58921e205199900bc997450b76f7c8ba85491113830e2e6  train_starts.json
2ad07a910559aa54f5ae7e959379a24bd0c8e75ed6c89489e1e421e70d8240f3  val_windows.json
```

## Commands run and results

| Command (cwd) | Result |
|---|---|
| `git status -sb`, `git log`, `git branch -a` (repo) | on `main`, one commit ahead of origin (`56fe587`), untracked `results/final_telecom/outputs/` |
| `nvidia-smi --query-gpu=name,utilization.gpu --format=csv` | `NVIDIA GeForce RTX 4060 Laptop GPU, 50 %` (the only GPU-related command) |
| `ls ~/fedflow_data/` | `local_val telecom_cache telecom_check.txt telecom_raw venv_gpu` |
| `git switch -c v2` | new branch |
| baseline in scratchpad: `generate_data.py`, `generate_data.py --dataset telecom`, `sha256sum -c` | Mean 92.59, Max 233.26, 4 x OK |
| `python3 generate_data.py --dataset telecom --variant v2` (scratch, then repo `training/`) | build report above |
| `FEDFLOW_N_EXAMPLES=400 python3 generate_data.py --dataset telecom --variant v2` (scratch only) | 400-example report above |
| `python3 generate_data.py --variant v2` | refused: `--variant v2 only applies with --dataset telecom` |
| `python3 check_split.py --dataset telecom` / `--variant v2` (repo `training/`) | `PASS` / `PASS` |
| `python3 token_utils.py --selftest` (CPU only, offline tokenizer) | label check line above, `SELFTEST OK` |
| `python3 token_audit_v2.py` | audit above |
| `python3 infer.py --selftest` | `SELFTEST OK` |
| stand-in harness: `infer.run_eval` on 4 windows x 4 adapter/data pairings, then all 276 windows for v1/v1 and v2/v2 | matching pairs ran and recorded `variant` v1/v2; mismatches stopped with ERROR |
| `python3 compare_variants.py --selftest` | `SELFTEST OK` (on the first run my hand-computed expected value was wrong, -67.50; I corrected it to -82.50 / -47.50 / -72.50 after recomputing by hand) |
| `compare_variants.py --v1 <stand-in v1> --v2 <stand-in v2>` | report written; stand-in numbers not results |
| `run_experiment.py --dataset telecom --selftest --forecasts <copy>` without `--out-root` | refused as intended |
| same with `--out-root <scratch>` | `SELFTEST OK` |
| `run_experiment.py --selftest --out-root x` | refused: telecom only |
| `~/fedflow_data/venv_gpu/bin/python -m pip list` / `--version` | Python 3.10.12; torch 2.5.1+cu121, transformers 4.46.3, peft 0.13.2, accelerate 1.15.0, numpy 2.2.6, pandas 2.3.3, tokenizers 0.20.3, safetensors 0.8.0 |
| `git ls-files '*.py' \| xargs python3 -m py_compile` | `py_compile OK on 18 files` |
| selftests: `infer.py`, `prepare_telecom.py`, `compare_variants.py`, `token_utils.py`, `analyze_peaks.py`, `summarize.py`, `run_experiment.py`, `run_experiment.py --dataset telecom --cell 7511` and `--cell 8336` | `SELFTEST OK` for every one |
| final v1 proof (above) | Mean 92.59, Max 233.26, 4 x OK, PASS |

Not run: training of any kind, `infer.py` with a model, the Colab notebook (I only checked that its code cells parse as Python), and any Mininet experiment.

## Files changed on branch v2

New:
- `PLAN_V2.md`
- `V2_RUNBOOK.md`
- `V2_STATUS.md`
- `training/token_utils.py`
- `training/token_audit_v2.py`
- `training/compare_variants.py`
- `notebooks/fedflow_colab_telecom_v2.ipynb`

Changed:
- `training/generate_data.py`
- `training/check_split.py`
- `training/client.py`
- `training/server.py`
- `training/infer.py`
- `experiments/run_experiment.py`
- `experiments/summarize.py`

Generated, not committed (gitignored): the six files in `training/data/telecom_v2/`
(`traffic_train.json`, `train_starts.json`, `val_windows.json`, `test_windows.json`, `build_report.txt`, `token_audit_v2.txt`).
Selftests also rewrote `controller/forecast_value.txt` (gitignored, as in every earlier selftest run) and refreshed `__pycache__` folders.
