# PLAN_V2: telecom training pipeline v2

Written and committed on branch `v2` before any v2 code, and before any v2 model exists.
Nothing below may be changed after a v2 result has been seen.

## What v2 is

v2 = v1 plus exactly three changes:

- (a) **Answer-only loss.** Training loss is computed on the answer tokens only (the output text and the end token).
  Prompt tokens and padding tokens get label -100.
- (b) **Time in the prompt.** The prompt gains one line stating the Milan local time and day type of the LAST history value,
  for example `Time of last value: Thursday 14:30 (working day)`. It is computed from the slot number and the calendar only,
  never from traffic values: slot 0 = Friday 1 November 2013 00:00 Milan time, 144 slots per day. Non-working days are
  1 November (public holiday) and every Saturday and Sunday.
- (c) **Onset-balanced sampling** of the training examples. Onset window = last history value <= 100 and maximum of the
  6 future values > 100. Per cell the target is half onset windows and half other windows, distinct starts only, never a
  duplicated window. A cell with too few onset windows takes all of them and is filled with other windows; the shortfall is
  recorded per cell. The build aborts if the training pool holds fewer than 60 onset windows in total. Training windows never
  touch validation or test slots.

Everything else is identical to v1: model Qwen/Qwen2.5-1.5B-Instruct, LoRA settings (r=8, alpha=16, q_proj/v_proj,
dropout 0.05), 25 rounds, 6 clients (client i trains only on cell i), FedAvg, seeds, the chronological split
(training slots 0-1439, validation 1440-1727, test 1728-2015), and the validation and test window starts.
The only extra setting v2 needs is its training length limit (MAX_LENGTH), measured from the real tokenizer and recorded
in `training/data/telecom_v2/token_audit_v2.txt` and `V2_STATUS.md`. It is chosen by rule (smallest multiple of 32 at
least 16 above the longest example), not tuned.

## Allowed attempts

At most 2.

- Attempt 1: v2 with 200 training examples (`FEDFLOW_N_EXAMPLES=200`, the default).
- Attempt 2: identical except `FEDFLOW_N_EXAMPLES=400`. It is run only if attempt 1 is not adopted.

No other change of any kind is allowed after seeing results.

## Scoring

- v1 and v2 are both scored on the **validation** block on the same local RTX 4060, each with its own frozen copy of the code
  (v1: `~/fedflow_data/val_v1/`, v2: `~/fedflow_data/local_val_v2/`).
- The test block is not used for any decision.

## Adoption rule

v2 is adopted only if **all four** hold, on validation:

1. Overall model MAE is lower than v1's (strictly).
2. Pooled onset recall over all six cells is higher than v1's (strictly).
3. False-alarm rate on calm windows is at most 5 percentage points above v1's.
4. Parse failures are no more than v1's.

Definitions, the same as `results/final/analyze_peaks.py`: onset window = last history value <= 100 and true peak > 100;
calm window = last history value <= 100 and true peak <= 100; a window is flagged when the model's predicted peak (max of its
6 predictions) is > 100; recall and false-alarm rate count only windows whose model prediction parsed.
Pooled recall = flagged onset windows / onset windows, summed over all six cells.
False-alarm rate = flagged calm windows / calm windows, summed over all six cells.

`training/compare_variants.py` applies these four rules mechanically and prints ADOPT or DO NOT ADOPT.

## After the decision

- **If adopted:** score v1 and v2 once each on the test block on the 4060, then rerun the experiment for cells 7511 and 8336
  with the v2 forecasts.
- **If not adopted:** v1 stays final and v2 is reported as a negative result.
- Every metric is reported either way.

## Note on training loss

Training loss is not comparable between v1 and v2. v2 averages the loss over the answer tokens only, while v1 averages it over
every token, including prompt and padding.
