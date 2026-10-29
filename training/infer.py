"""
FedFlow-LLM — infer.py
-----------------------
Loads the base model + the federated LoRA adapter saved by server.py
(outputs/adapter.npz) and forecasts the next 6 traffic values.

Usage (from inside training/):
  python infer.py --window "v1,...,v12"   forecast one window, print PEAK
  python infer.py --eval [--limit N]      score held-out test windows
  python infer.py --eval --split val      score the validation windows instead (writes val_* files)
  python infer.py --selftest              test parser + metrics, no model needed
  add --max-new-tokens N to --window/--eval to change the generation limit

If the test windows carry a "cell" field (telecom data), forecasts.csv gets a
"cell" column and metrics.json gets a "per_cell" block. Synthetic output is unchanged.
A "slope" baseline (least-squares line through the 12 history values) is scored next to
the model; its forecast is the last column of the forecasts CSV.

torch/transformers/peft are imported only inside the functions that need
them, so --selftest runs on a machine without them.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

from generate_data import build_instruction

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "outputs/adapter.npz"
TEST_WINDOWS_PATH = "data/test_windows.json"
FORECASTS_PATH = "outputs/forecasts.csv"
METRICS_PATH = "outputs/metrics.json"
VAL_WINDOWS_PATH = "data/val_windows.json"
VAL_FORECASTS_PATH = "outputs/val_forecasts.csv"
VAL_METRICS_PATH = "outputs/val_metrics.json"
ONSET_THRESHOLD = 100.0  # onset = last history value <= this and true peak above it (same as analyze_peaks.py)
MAX_NEW_TOKENS = 160  # default; override with --max-new-tokens
HISTORY_LEN = 12
HORIZON = 6


# ---------------------------------------------------------------- parsing / metrics

def parse_prediction(text, truncated=False):
    """Return the 6 floats on the final "PREDICTION:" line, or None.
    If generation was cut off by the token limit, the last line may be a
    half-written number (e.g. "12" from "123.45"), so it is ignored."""
    if truncated:
        text = text.rsplit("\n", 1)[0] if "\n" in text else ""
    matches = re.findall(r"(?m)^PREDICTION:\s*(.+)$", text)
    if not matches:
        return None
    parts = matches[-1].split(",")
    if len(parts) != HORIZON:
        return None
    try:
        values = [float(p.strip()) for p in parts]
    except ValueError:
        return None
    # float() accepts "nan"/"inf"; those are not usable forecasts.
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def mae_rmse(predictions, truths):
    """MAE and RMSE over every value of every window. Returns (None, None) if empty."""
    errors = [p - t for pred, true in zip(predictions, truths) for p, t in zip(pred, true)]
    if not errors:
        return None, None
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return mae, rmse


METHODS = ("model", "persistence", "moving_avg", "slope")


def score_methods(scored):
    """{method: {"mae", "rmse"}} for every method in METHODS over the windows collected in `scored`."""
    return {name: dict(zip(("mae", "rmse"), mae_rmse(scored[name], scored["true"]))) for name in METHODS}


def per_cell_metrics(windows, scored, scored_cells):
    """Same metrics as the overall ones, split by cell. `scored_cells[i]` is the cell of the i-th scored window."""
    out = {}
    for cell in sorted({w["cell"] for w in windows}):
        keep = [i for i, c in enumerate(scored_cells) if c == cell]            # scored windows of this cell
        subset = {name: [scored[name][i] for i in keep] for name in scored}
        entry = {
            "square_id": next(w["square_id"] for w in windows if w["cell"] == cell and "square_id" in w),
            "windows_used": sum(1 for w in windows if w["cell"] == cell),
            "windows_scored": len(keep),
        }
        entry.update(score_methods(subset))
        entry["slope"]["onset"] = onset_stats(subset["history"], subset["true"], subset["slope"])
        out[str(cell)] = entry
    return out


def onset_stats(histories, truths, predictions):
    """Onset recall of a forecaster. An onset window starts at or below the threshold and its true peak goes above it.
    Recall = share of onset windows whose predicted peak is above the threshold.
    mean_peak_error = mean of (predicted peak - true peak) over onset windows (negative = under-predicts)."""
    onsets = [(max(true), max(pred)) for hist, true, pred in zip(histories, truths, predictions)
              if hist[-1] <= ONSET_THRESHOLD and max(true) > ONSET_THRESHOLD]
    flagged = sum(1 for _, pred_peak in onsets if pred_peak > ONSET_THRESHOLD)
    return {
        "onset_windows": len(onsets),
        "flagged": flagged,
        "recall": flagged / len(onsets) if onsets else None,   # None = no onset windows, nothing to measure
        "mean_peak_error": sum(p - t for t, p in onsets) / len(onsets) if onsets else None,
    }


def persistence_forecast(history):
    """Baseline: repeat the last observed value 6 times."""
    return [history[-1]] * HORIZON


def moving_average_forecast(history):
    """Baseline: mean of the last 3 observed values, repeated 6 times."""
    return [sum(history[-3:]) / 3] * HORIZON


def slope_forecast(history):
    """Baseline: least-squares line through the 12 history values at x = 0..11, continued 6 steps and clipped at 0."""
    x_mid = (HISTORY_LEN - 1) / 2                       # mean of x = 0..11, i.e. 5.5
    y_mean = sum(history) / HISTORY_LEN
    slope = sum((x - x_mid) * (y - y_mean) for x, y in enumerate(history)) / sum((x - x_mid) ** 2 for x in range(HISTORY_LEN))
    # Step h sits at x = 11 + h; traffic cannot be negative, so clip at 0.
    return [max(0.0, y_mean + slope * (HISTORY_LEN - 1 + h - x_mid)) for h in range(1, HORIZON + 1)]


def split_paths(split):
    """(windows file, forecasts csv, metrics json) for a split. Val gets its own output names so test files are never overwritten."""
    if split == "val":
        return VAL_WINDOWS_PATH, VAL_FORECASTS_PATH, VAL_METRICS_PATH
    return TEST_WINDOWS_PATH, FORECASTS_PATH, METRICS_PATH


def load_split_windows(split):
    """Read the windows of one split. Opens only that split's own file."""
    with open(split_paths(split)[0], "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- model

def load_model():
    """Base model + LoRA wrapper, with outputs/adapter.npz loaded into it."""
    # Skip transformers' optional vision/audio imports, same as client.py.
    os.environ.setdefault("USE_TORCHVISION", "0")
    os.environ.setdefault("USE_TORCHAUDIO", "0")
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 on GPU like training; CPU fp16 is slow/unsupported for many ops, so use fp32 there.
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)

    # These LoRA settings MUST match client.py, or the saved arrays won't fit.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Same key order the clients used when sending weights to the server.
    current_state = get_peft_model_state_dict(model)
    keys = list(current_state.keys())

    if not os.path.exists(ADAPTER_PATH):
        sys.exit(f"ERROR: {ADAPTER_PATH} not found. Run server.py first.")
    with np.load(ADAPTER_PATH) as npz:
        # np.savez names arrays arr_0, arr_1, ...; read them back by index to keep order.
        arrays = [npz[f"arr_{i}"] for i in range(len(npz.files))]

    if len(arrays) != len(keys):
        sys.exit(
            f"ERROR: adapter has {len(arrays)} arrays but the model expects {len(keys)}. "
            "LoRA config in infer.py probably differs from client.py."
        )
    new_state = {}
    for key, array in zip(keys, arrays):
        expected_shape = tuple(current_state[key].shape)
        if tuple(array.shape) != expected_shape:
            sys.exit(f"ERROR: shape mismatch for {key}: adapter {array.shape}, model {expected_shape}.")
        # Match each parameter's existing dtype (LoRA weights may be fp32 even on a fp16 base).
        new_state[key] = torch.tensor(array, dtype=current_state[key].dtype)
    set_peft_model_state_dict(model, new_state)

    model.eval()
    return model, tokenizer, device


def eos_token_ids(model, tokenizer):
    """All token ids that count as end-of-sequence (the model config may list several)."""
    ids = model.generation_config.eos_token_id
    ids = set(ids) if isinstance(ids, (list, tuple)) else ({ids} if ids is not None else set())
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def predict(model, tokenizer, device, history, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy-decode one forecast. Returns (6 floats or None, raw generated text)."""
    import torch

    # Exactly client.py's training format: instruction text, then a newline.
    prompt = build_instruction(history) + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    # Generation stops early only on an end token; using all max_new_tokens means it was cut off.
    truncated = len(new_ids) >= max_new_tokens
    # Separately report whether the model actually emitted an end-of-sequence token.
    ended_with_eos = len(new_ids) > 0 and int(new_ids[-1]) in eos_token_ids(model, tokenizer)
    print(f"new tokens: {len(new_ids)}/{max_new_tokens}" + ("" if ended_with_eos else "  HIT TOKEN LIMIT"))
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return parse_prediction(text, truncated=truncated), text


# ---------------------------------------------------------------- modes

def run_window(window_arg, max_new_tokens):
    history = [float(v) for v in window_arg.split(",")]
    if len(history) != HISTORY_LEN:
        sys.exit(f"ERROR: --window needs {HISTORY_LEN} comma-separated values, got {len(history)}.")
    model, tokenizer, device = load_model()
    prediction, text = predict(model, tokenizer, device, history, max_new_tokens)
    if prediction is None:
        print("Could not parse a PREDICTION line from the model output:")
        print(text)
        sys.exit(1)
    print("PREDICTION: " + ", ".join(f"{v:.2f}" for v in prediction))
    print(f"PEAK: {max(prediction):.2f}")


def run_eval(limit, max_new_tokens, split="test"):
    windows_path, forecasts_path, metrics_path = split_paths(split)
    windows = load_split_windows(split)
    if limit is not None:
        windows = windows[:limit]

    model, tokenizer, device = load_model()

    rows = []
    scored = {"history": [], "true": [], "model": [], "persistence": [], "moving_avg": [], "slope": []}
    scored_cells = []   # cell of each scored window (telecom windows only)
    has_cells = all("cell" in w for w in windows)   # synthetic windows have no cell, so their outputs stay as before
    parse_failures = 0
    for n, w in enumerate(windows, start=1):
        history, true = w["history"], w["true"]
        prediction, raw_output = predict(model, tokenizer, device, history, max_new_tokens)
        if n <= 2:
            # Show the first couple of raw generations so format problems are visible in the log.
            print(f"--- raw output, window start={w['start']} ---\n{raw_output}\n--- end raw output ---")
        persistence = persistence_forecast(history)
        moving_avg = moving_average_forecast(history)
        slope = slope_forecast(history)
        if prediction is None:
            parse_failures += 1
        else:
            # Only windows the model parsed are scored, for every method.
            scored["history"].append(history)
            scored["true"].append(true)
            scored["model"].append(prediction)
            scored["persistence"].append(persistence)
            scored["moving_avg"].append(moving_avg)
            scored["slope"].append(slope)
            if has_cells:
                scored_cells.append(w["cell"])
        for step in range(HORIZON):
            rows.append({
                **({"cell": w["cell"]} if has_cells else {}),   # first column, only when the windows carry a cell
                "start": w["start"],
                "step": step + 1,
                "true": true[step],
                "model": "" if prediction is None else prediction[step],  # blank = parse failure
                "persistence": persistence[step],
                "moving_avg": round(moving_avg[step], 4),
                "parsed": prediction is not None,
                "raw_output": raw_output,  # same full text on all 6 rows of a window
                "slope": round(slope[step], 4),  # appended last so older columns keep their order
            })
        print(f"[{n}/{len(windows)}] start={w['start']} parsed={prediction is not None}")

    os.makedirs("outputs", exist_ok=True)
    with open(forecasts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["start"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "windows_used": len(windows),
        "parse_failures": parse_failures,
        "windows_scored": len(scored["model"]),
    }
    for name in METHODS:
        mae, rmse = mae_rmse(scored[name], scored["true"])
        metrics[name] = {"mae": mae, "rmse": rmse}
    metrics["slope"]["onset"] = onset_stats(scored["history"], scored["true"], scored["slope"])
    if has_cells:
        metrics["per_cell"] = per_cell_metrics(windows, scored, scored_cells)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {forecasts_path} and {metrics_path}")


def run_selftest():
    good = "Trend direction: increasing.\nJustification: x.\nPREDICTION: 1.5, 2, 3.25, 4, 5, 6"
    assert parse_prediction(good) == [1.5, 2.0, 3.25, 4.0, 5.0, 6.0]
    # Uses the final PREDICTION line when there are several.
    two = "PREDICTION: 9, 9, 9, 9, 9, 9\nPREDICTION: 1, 2, 3, 4, 5, 6\n"
    assert parse_prediction(two) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert parse_prediction("no forecast here") is None
    assert parse_prediction("PREDICTION: 1, 2, 3, 4, 5") is None             # too few
    assert parse_prediction("PREDICTION: 1, 2, 3, 4, 5, 6, 7") is None       # too many
    assert parse_prediction("PREDICTION: 1, 2, x, 4, 5, 6") is None          # not a number
    assert parse_prediction("PREDICTION: 1, 2, nan, 4, 5, 6") is None        # not finite
    assert parse_prediction("Text PREDICTION: 1, 2, 3, 4, 5, 6") is None     # not at line start
    # Cut off by the token limit: last line may be a partial number, so reject it.
    assert parse_prediction("Trend.\nPREDICTION: 1, 2, 3, 4, 5, 12", truncated=True) is None
    # Cut off later, after a complete PREDICTION line: still usable.
    assert parse_prediction("PREDICTION: 1, 2, 3, 4, 5, 6\nextra te", truncated=True) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    mae, rmse = mae_rmse([[1.0, 2.0]], [[1.0, 4.0]])
    assert math.isclose(mae, 1.0) and math.isclose(rmse, math.sqrt(2.0))
    mae, rmse = mae_rmse([[3.0], [5.0]], [[1.0], [1.0]])
    assert math.isclose(mae, 3.0) and math.isclose(rmse, math.sqrt(10.0))
    assert mae_rmse([], []) == (None, None)

    history = [float(v) for v in range(1, 13)]  # 1..12
    assert persistence_forecast(history) == [12.0] * 6
    assert moving_average_forecast(history) == [11.0] * 6

    # Per-cell metrics: cell 0 has errors 1 and 3, cell 1 has error 0, so overall MAE is (1 + 3 + 0) / 3.
    windows = [{"cell": 0, "square_id": 11}, {"cell": 0, "square_id": 11}, {"cell": 1, "square_id": 22}]
    scored = {"true": [[0.0], [0.0], [5.0]], "model": [[1.0], [3.0], [5.0]],
              "persistence": [[0.0], [0.0], [0.0]], "moving_avg": [[0.0], [0.0], [0.0]],
              "slope": [[0.0], [0.0], [5.0]], "history": [[0.0] * 12] * 3}
    result = per_cell_metrics(windows, scored, [0, 0, 1])
    assert result["0"]["square_id"] == 11 and result["0"]["windows_used"] == 2 and result["0"]["windows_scored"] == 2
    assert math.isclose(result["0"]["model"]["mae"], 2.0) and math.isclose(result["1"]["model"]["mae"], 0.0)
    assert math.isclose(result["1"]["persistence"]["mae"], 5.0)
    assert math.isclose(score_methods(scored)["model"]["mae"], 4.0 / 3.0)
    assert math.isclose(result["1"]["slope"]["mae"], 0.0) and result["1"]["slope"]["onset"]["onset_windows"] == 0

    # Slope baseline, worked by hand. History 1..12: x mean 5.5, y mean 6.5, slope exactly 1,
    # so step h = 6.5 + (5.5 + h) = 12 + h -> 13..18.
    assert all(math.isclose(a, b) for a, b in zip(slope_forecast(history), [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]))
    # Constant series -> slope 0 -> repeats the constant.
    assert all(math.isclose(a, 7.0) for a in slope_forecast([7.0] * 12))
    # Falling series 20, 18.5, ..., 3.5 (slope -1.5, y mean 11.75): step h = 3.5 - 1.5h = 2.0, 0.5, then negative -> clipped to 0.
    falling = [20.0 - 1.5 * k for k in range(12)]
    got = slope_forecast(falling)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(got, [2.0, 0.5, 0.0, 0.0, 0.0, 0.0])), got
    # Steeply falling series: every step would be negative, so all six clip to 0.
    assert slope_forecast([24.0 - 2.0 * k for k in range(12)]) == [0.0] * 6

    # Onset stats: window A is an onset (last 50 <= 100, true peak 150) flagged with peak 120 (error -30);
    # B is an onset (true peak 110) missed with peak 60 (error -50); C starts high (not an onset); D stays calm (not an onset).
    stats = onset_stats([[50.0] * 12, [50.0] * 12, [150.0] * 12, [50.0] * 12],
                        [[10, 150, 10, 10, 10, 10], [10, 110, 10, 10, 10, 10], [200] * 6, [20] * 6],
                        [[10, 120, 10, 10, 10, 10], [10, 60, 10, 10, 10, 10], [200] * 6, [90] * 6])
    assert stats["onset_windows"] == 2 and stats["flagged"] == 1 and math.isclose(stats["recall"], 0.5)
    assert math.isclose(stats["mean_peak_error"], -40.0)   # (-30 + -50) / 2
    assert onset_stats([[50.0] * 12], [[20] * 6], [[20] * 6])["recall"] is None   # no onsets -> nothing to measure

    # --split: val reads only the val file and writes val_* outputs; test keeps the original names.
    assert split_paths("test") == (TEST_WINDOWS_PATH, FORECASTS_PATH, METRICS_PATH)
    assert split_paths("val") == (VAL_WINDOWS_PATH, VAL_FORECASTS_PATH, VAL_METRICS_PATH)
    assert not {FORECASTS_PATH, METRICS_PATH} & {VAL_FORECASTS_PATH, VAL_METRICS_PATH}
    import tempfile
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "data"))
        with open(os.path.join(tmp, VAL_WINDOWS_PATH), "w", encoding="utf-8") as f:
            json.dump([{"start": 7}], f)          # only the val file exists in this folder
        os.chdir(tmp)
        try:
            assert load_split_windows("val") == [{"start": 7}]
            try:
                load_split_windows("test")          # must try the test file, which is absent, proving the two never mix
                raise AssertionError("test split read a file that does not exist")
            except FileNotFoundError:
                pass
        finally:
            os.chdir(old_cwd)

    prompt = build_instruction(history)
    assert "History: 1.00, 2.00" in prompt and prompt.endswith("PREDICTION: v1, v2, v3, v4, v5, v6")

    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(description="FedFlow-LLM inference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--window", help='12 comma-separated history values, e.g. "100,101,..."')
    mode.add_argument("--eval", action="store_true", help="score the held-out test windows")
    mode.add_argument("--selftest", action="store_true", help="test parser and metrics without a model")
    parser.add_argument("--limit", type=int, help="with --eval: only use the first N windows")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="with --eval: which windows to score (default test; val writes val_forecasts.csv and val_metrics.json)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"generation limit per window (default {MAX_NEW_TOKENS})")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    if args.split != "test" and not args.eval:
        parser.error("--split only applies with --eval")

    if args.selftest:
        run_selftest()
    elif args.eval:
        run_eval(args.limit, args.max_new_tokens, args.split)
    else:
        run_window(args.window, args.max_new_tokens)


if __name__ == "__main__":
    main()
