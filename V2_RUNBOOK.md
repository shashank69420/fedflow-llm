# V2 runbook: the morning steps

This is the exact order of things to do to train v2 on Colab, score it on the laptop's RTX 4060, and decide whether
to adopt it. The rules come from `PLAN_V2.md` and are not changed here. Read `V2_STATUS.md` first.

## How to use this page

- **Ubuntu terminal**: press the Windows key, type `Ubuntu`, press Enter. A black window opens with a line ending in `$`.
  That is where every grey command block below goes, unless it says "browser".
- **Paste** into it with right-click or `Ctrl + Shift + V`. Paste one block at a time and press Enter.
- Wait until the `$` line comes back before pasting the next block.
- If anything prints `ERROR`, `ABORT`, `FAIL`, `Traceback` or `No such file`, **stop** and give the exact text to Claude.
- Lines that start with `#` are comments for you; the terminal ignores them.
- **Never run anything with `sudo` unless this page says so.** Only the experiment step needs it.

## 0. Before you start

**0.1** Make sure the v1 scoring job has finished. Nothing that says `python` should be listed under "Processes":
```bash
nvidia-smi
```

**0.2** Make sure the v1 validation results exist where `PLAN_V2.md` says they are. When Claude checked overnight, the folder
`~/fedflow_data/val_v1/` did **not** exist yet; only `~/fedflow_data/local_val/` did, and Claude did not look inside it.
```bash
find ~/fedflow_data/val_v1 -name 'val_metrics.json' -o -name 'val_forecasts.csv' 2>&1
```
You need one `val_metrics.json` and one `val_forecasts.csv`, both from scoring v1 with `--split val`. Note their folder.
If the command says "No such file or directory", ask Claude before going on: the comparison needs these two files.

**0.3** Go to the project and the v2 branch:
```bash
cd ~/fedflow
git switch v2
git status -sb
git log --oneline -12
```
You should see `## v2` and, at the top of the log, the commit that adds `V2_STATUS.md`.

## 1. Push the v2 branch

Colab downloads the code from GitHub, so the branch must be there. (Claude never pushes; you do.)
```bash
cd ~/fedflow
git push -u origin v2
git rev-parse v2
```
Write down the long code the last line prints (the commit hash). You will compare it with Colab's later.

## 2. Colab smoke test (browser)

**2.1** Open this link in the browser:
https://colab.research.google.com/github/shashank69420/fedflow-llm/blob/v2/notebooks/fedflow_colab_telecom_v2.ipynb

**2.2** Runtime → Change runtime type → **T4 GPU** → Save.

**2.3** Optional: follow the keep-alive note in the first cell so Colab doesn't disconnect.

**2.4** Run the cells **one at a time, from the top**, with the ▶ button. Never run a cell twice. In order:
1. `nvidia-smi`: must show a Tesla T4.
2. Install. It takes a few minutes, and red pip warnings are normal.
3. Clone and `git switch v2`. The printed commit must start with the hash you wrote down in step 1.
4. Generate the v2 data and run the split check. It must end with `PASS`.
5. Copy the four files. It prints `training examples: 200`.
6. Smoke test, 2 rounds.
7. Smoke-test check. It must print **`SMOKE TEST PASSED`**.

**Pass** = 2 rounds finish, `outputs/adapter.npz` and `outputs/run_config.json` exist, and there is no `ABORT ... longer than
MAX_LENGTH` message. If it fails, stop and give Claude the last 40 lines of the cell output.

## 3. Colab full run (browser)

**3.1** Run the cell under "DO NOT RUN BELOW until the smoke test has been checked": 25 rounds, about the length of the v1 full run.

**3.2** As soon as it finishes, run the next cell. It zips and downloads **`v2_attempt1_outputs.zip`**, which lands in your Windows
Downloads folder. Don't close Colab until the download has finished.

**3.3** Do **not** run the "ATTEMPT 2 ONLY IF PLAN_V2 SAYS SO" section now.

## 4. Make the frozen v2 scoring folder

This makes `~/fedflow_data/local_val_v2/`, a copy of exactly the v2 code you pushed, plus the v2 adapter and the v2
validation windows. It is never edited afterwards.

**4.1** Check that the folder doesn't exist yet. The command must print `No such file or directory`:
```bash
ls ~/fedflow_data/local_val_v2
```

**4.2** Copy the code at the pushed commit, then add the Colab outputs:
```bash
mkdir ~/fedflow_data/local_val_v2
cd ~/fedflow
git archive --format=tar v2 | tar -x -C ~/fedflow_data/local_val_v2
git rev-parse v2 > ~/fedflow_data/local_val_v2/CODE_COMMIT.txt
cd ~/fedflow_data/local_val_v2/training
unzip /mnt/c/Users/shash/Downloads/v2_attempt1_outputs.zip
cp data/telecom_v2/val_windows.json data/val_windows.json
```
If Windows renamed the download (for example `v2_attempt1_outputs (1).zip`), put that name in quotes in the `unzip` line.

**4.3** Check what you got:
```bash
cd ~/fedflow_data/local_val_v2/training
cat outputs/run_config.json
cat ../CODE_COMMIT.txt
sha256sum data/val_windows.json ~/fedflow/training/data/telecom_v2/val_windows.json
ls -l outputs
```
- `run_config.json` must say `"variant": "v2"`, `"max_length": 288`, `"n_examples": 200`, `"rounds": 25`.
- Its `"git_commit"` must equal the hash in `CODE_COMMIT.txt`.
- The two `sha256sum` lines must show the same long code. The validation windows don't depend on the number of examples,
  so they match the ones Claude built overnight.
- `outputs/` must contain `adapter.npz`.

## 5. Score v2 on the validation block (RTX 4060)

```bash
cd ~/fedflow_data/local_val_v2/training
~/fedflow_data/venv_gpu/bin/python infer.py --eval --split val 2>&1 | tee score_val_v2.log
```
- It first checks that the adapter (v2, from `run_config.json`) and the windows (tagged v2) match, and stops with an `ERROR` if not.
- It prints one line per window (`[n/276] ...`) and ends with `Wrote outputs/val_forecasts.csv and outputs/val_metrics.json`.
- Then check:
```bash
grep '"variant"' outputs/val_metrics.json
grep '"parse_failures"' outputs/val_metrics.json
```
The first line must say `"variant": "v2"`.

## 6. Compare v1 and v2 and get the decision

Use the folder you found in step 0.2 for v1. If both files are directly in `~/fedflow_data/val_v1`, or in its `outputs/` or
`training/outputs/` subfolder, this works as is:
```bash
cd ~/fedflow_data/local_val_v2/training
python3 compare_variants.py --v1 ~/fedflow_data/val_v1 --v2 ~/fedflow_data/local_val_v2/training \
    --out ~/fedflow_data/local_val_v2/compare_attempt1.md
```
If it says it found no file pair, name the two v1 files directly, using the paths from step 0.2:
```bash
python3 compare_variants.py --v1-metrics PATH/val_metrics.json --v1-forecasts PATH/val_forecasts.csv \
    --v2 ~/fedflow_data/local_val_v2/training --out ~/fedflow_data/local_val_v2/compare_attempt1.md
```
The last line is **ADOPT** or **DO NOT ADOPT**. The full report is in `~/fedflow_data/local_val_v2/compare_attempt1.md`.

## 7a. If it says ADOPT

`PLAN_V2.md` says: score v1 and v2 **once** on the test block on the 4060, then rerun the experiment for cells 7511 and 8336
with the v2 forecasts.

**v2 on the test block:**
```bash
cd ~/fedflow_data/local_val_v2/training
cp data/telecom_v2/test_windows.json data/test_windows.json
~/fedflow_data/venv_gpu/bin/python infer.py --eval 2>&1 | tee score_test_v2.log
```
It writes `outputs/forecasts.csv` and `outputs/metrics.json` next to the validation files.

**v1 on the test block:** in the v1 frozen folder (the one holding v1's `val_metrics.json`), run the same `infer.py --eval`
with v1's own `data/test_windows.json`. Before scoring, check that its checksum equals
`~/fedflow/training/data/telecom/test_windows.json`:
```bash
sha256sum ~/fedflow/training/data/telecom/test_windows.json
```
Claude didn't set that folder up, so ask Claude to check the exact commands against it first.

**Experiments with v2 forecasts.** These need `sudo` and Mininet, so you run them, from `~/fedflow` on branch `v2`. Each takes about 50 minutes:
```bash
cd ~/fedflow
sudo python3 experiments/run_experiment.py --dataset telecom --cell 7511 \
    --forecasts ~/fedflow_data/local_val_v2/training/outputs/forecasts.csv --out-root results/experiment_telecom_v2
python3 experiments/summarize.py --dataset telecom --cell 7511 --out-root results/experiment_telecom_v2
sudo python3 experiments/run_experiment.py --dataset telecom --cell 8336 \
    --forecasts ~/fedflow_data/local_val_v2/training/outputs/forecasts.csv --out-root results/experiment_telecom_v2
python3 experiments/summarize.py --dataset telecom --cell 8336 --out-root results/experiment_telecom_v2
```
The script refuses to write v2 forecasts into `results/experiment_telecom` (the v1 results), so `--out-root` is required.
Then ask Claude to write up v1 vs v2 using only these files.

## 7b. If it says DO NOT ADOPT

`PLAN_V2.md` allows exactly one more attempt: identical, except 400 training examples.

1. In Colab, use a **fresh runtime** (Runtime → Disconnect and delete runtime). Run the GPU check, install and clone cells at
   the top. Then run the **"ATTEMPT 2 ONLY IF PLAN_V2 SAYS SO"** section, top to bottom. It downloads **`v2_attempt2_outputs.zip`**.
2. Repeat steps 4 to 6 with these changes, so attempt 1 stays frozen:
   - folder `~/fedflow_data/local_val_v2_attempt2` instead of `~/fedflow_data/local_val_v2`
   - `unzip /mnt/c/Users/shash/Downloads/v2_attempt2_outputs.zip`
   - in 4.3, `run_config.json` must say `"n_examples": 400` and `"fedflow_n_examples_env": "400"`
   - in 6, `--v2 ~/fedflow_data/local_val_v2_attempt2/training --out ~/fedflow_data/local_val_v2_attempt2/compare_attempt2.md`
3. **ADOPT** → do 7a with the attempt-2 folder in place of `local_val_v2`.
   **DO NOT ADOPT** → v1 stays final and v2 is reported as a negative result. There is no attempt 3.
   Ask Claude to write the report with every metric from both comparison reports.

## Never

- Change anything in `PLAN_V2.md`, the frozen folders or their outputs after seeing a result.
- Use the test block (`--eval` without `--split val`) before the decision in step 6.
- Run attempt 2 after an ADOPT, or any third attempt.
