"""
FedFlow-LLM — infer.py
-----------------------
Loads the base model + the federated LoRA adapter saved by server.py
(outputs/adapter.npz) and forecasts the next 6 traffic values.

Usage (from inside training/):
  python infer.py --window "v1,...,v12"   forecast one window, print PEAK
  python infer.py --eval [--limit N]      score held-out test windows
  python infer.py --selftest              test parser + metrics, no model needed

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
MAX_NEW_TOKENS = 80
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


def persistence_forecast(history):
    """Baseline: repeat the last observed value 6 times."""
    return [history[-1]] * HORIZON


def moving_average_forecast(history):
    """Baseline: mean of the last 3 observed values, repeated 6 times."""
    return [sum(history[-3:]) / 3] * HORIZON


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


def predict(model, tokenizer, device, history):
    """Greedy-decode one forecast. Returns (6 floats or None, raw generated text)."""
    import torch

    # Exactly client.py's training format: instruction text, then a newline.
    prompt = build_instruction(history) + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    # Generation stops early only on an end token; using all MAX_NEW_TOKENS means it was cut off.
    truncated = len(new_ids) >= MAX_NEW_TOKENS
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return parse_prediction(text, truncated=truncated), text


# ---------------------------------------------------------------- modes

def run_window(window_arg):
    history = [float(v) for v in window_arg.split(",")]
    if len(history) != HISTORY_LEN:
        sys.exit(f"ERROR: --window needs {HISTORY_LEN} comma-separated values, got {len(history)}.")
    model, tokenizer, device = load_model()
    prediction, text = predict(model, tokenizer, device, history)
    if prediction is None:
        print("Could not parse a PREDICTION line from the model output:")
        print(text)
        sys.exit(1)
    print("PREDICTION: " + ", ".join(f"{v:.2f}" for v in prediction))
    print(f"PEAK: {max(prediction):.2f}")


def run_eval(limit):
    with open(TEST_WINDOWS_PATH, "r", encoding="utf-8") as f:
        windows = json.load(f)
    if limit is not None:
        windows = windows[:limit]

    model, tokenizer, device = load_model()

    rows = []
    scored = {"true": [], "model": [], "persistence": [], "moving_avg": []}
    parse_failures = 0
    for n, w in enumerate(windows, start=1):
        history, true = w["history"], w["true"]
        prediction, _ = predict(model, tokenizer, device, history)
        persistence = persistence_forecast(history)
        moving_avg = moving_average_forecast(history)
        if prediction is None:
            parse_failures += 1
        else:
            # Only windows the model parsed are scored, for all three methods.
            scored["true"].append(true)
            scored["model"].append(prediction)
            scored["persistence"].append(persistence)
            scored["moving_avg"].append(moving_avg)
        for step in range(HORIZON):
            rows.append({
                "start": w["start"],
                "step": step + 1,
                "true": true[step],
                "model": "" if prediction is None else prediction[step],  # blank = parse failure
                "persistence": persistence[step],
                "moving_avg": round(moving_avg[step], 4),
                "parsed": prediction is not None,
            })
        print(f"[{n}/{len(windows)}] start={w['start']} parsed={prediction is not None}")

    os.makedirs("outputs", exist_ok=True)
    with open(FORECASTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["start"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "windows_used": len(windows),
        "parse_failures": parse_failures,
        "windows_scored": len(scored["model"]),
    }
    for name in ("model", "persistence", "moving_avg"):
        mae, rmse = mae_rmse(scored[name], scored["true"])
        metrics[name] = {"mae": mae, "rmse": rmse}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {FORECASTS_PATH} and {METRICS_PATH}")


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
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.eval:
        run_eval(args.limit)
    else:
        run_window(args.window)


if __name__ == "__main__":
    main()
