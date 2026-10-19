"""
FedFlow-LLM — generate_data.py
-------------------------------
Builds the synthetic traffic series, holds out a test block (days 10-11),
and turns the rest into instruction-tuning examples.

Writes (relative to training/):
  data/synthetic_traffic.csv  full traffic series
  data/traffic_train.json     200 training examples (no overlap with the test block)
  data/train_starts.json      start timestep of each training window
  data/test_windows.json      every 18-step window inside the test block, step 3
"""
import json
import os
import re

import numpy as np
import pandas as pd

STEPS_PER_DAY = 24 * 6     # 10-minute intervals -> 144 steps per day
HISTORY_LEN = 12           # values the model sees
HORIZON = 6                # values the model must predict
WINDOW_LEN = HISTORY_LEN + HORIZON
TEST_DAYS = [10, 11]       # day 0 = Monday, so these are Thursday and Friday of week 2
TEST_STRIDE = 3            # test windows start every 3 timesteps
N_TRAIN_EXAMPLES = 200


def build_instruction(history_values):
    """The prompt text for one window. infer.py imports this, so training
    and inference always use exactly the same wording."""
    history_text = ", ".join(f"{value:.2f}" for value in history_values)
    return (
        "You are a traffic-forecasting assistant. Predict the next 6 traffic values "
        "given the following 12 historical values measured at 10-minute intervals.\n\n"
        f"History: {history_text}\n\n"
        "Reason step by step about the traffic pattern, then end with exactly this format:\n"
        "PREDICTION: v1, v2, v3, v4, v5, v6"
    )


def build_output(history_values, future_values):
    """The target answer text for one window."""
    # Slope = mean of last 3 history values minus mean of first 3.
    slope = float(np.mean(history_values[-3:]) - np.mean(history_values[:3]))
    if slope > 1:
        trend = "increasing"
    elif slope < -1:
        trend = "decreasing"
    else:
        trend = "stable"

    prediction_text = ", ".join(f"{value:.2f}" for value in future_values)
    return (
        f"Trend direction: {trend}.\n"
        f"Justification: The history slope (mean of last 3 minus mean of first 3) is {slope:.2f}.\n"
        f"PREDICTION: {prediction_text}"
    )


def verify_prediction(output_text, expected, label):
    """Parse the PREDICTION line back out of output_text and check it equals expected."""
    match = re.search(r"(?m)^PREDICTION:\s*(.+)$", output_text)
    if not match:
        raise AssertionError(f"{label} failed: PREDICTION line not found.")

    try:
        parsed = [float(value.strip()) for value in match.group(1).split(",")]
    except ValueError as error:
        raise AssertionError(f"{label} failed: invalid PREDICTION values.") from error

    if len(parsed) != HORIZON or parsed != expected:
        raise AssertionError(
            f"{label} failed: parsed {parsed} does not match expected {expected}."
        )


def generate_traffic():
    """Cell 1: Generate synthetic traffic data (seed 42, unchanged)."""
    rng = np.random.default_rng(42)
    steps_per_day = 24 * 6
    n_steps = 14 * steps_per_day
    timesteps = np.arange(n_steps)

    daily_pattern = 100 + 35 * np.sin(2 * np.pi * (timesteps % steps_per_day) / steps_per_day - np.pi / 2)
    day_index = timesteps // steps_per_day
    weekend_multiplier = np.where((day_index % 7) >= 5, 0.7, 1.0)
    traffic = daily_pattern * weekend_multiplier + rng.normal(0, 7, n_steps)

    for _ in range(15):
        start = rng.integers(0, n_steps - 8)
        duration = rng.integers(2, 9)
        height = rng.uniform(30, 120)
        traffic[start:start + duration] += height * np.hanning(duration + 2)[1:-1]

    traffic = np.clip(traffic, 0, None)
    df = pd.DataFrame({"timestep": timesteps, "traffic": traffic.round(2)})

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_traffic.csv", index=False)

    print(df.head(10))
    print(f"\nMean: {df['traffic'].mean():.2f}")
    print(f"Max: {df['traffic'].max():.2f}")


def main():
    generate_traffic()

    df = pd.read_csv("data/synthetic_traffic.csv", usecols=["timestep", "traffic"])
    traffic = df["traffic"].tolist()

    # Test block = all timesteps of the held-out days. [test_start, test_end) is half-open.
    test_start = min(TEST_DAYS) * STEPS_PER_DAY
    test_end = (max(TEST_DAYS) + 1) * STEPS_PER_DAY
    block_max = max(traffic[test_start:test_end])
    if block_max <= 100:
        raise ValueError(
            f"Test block (days {TEST_DAYS}) has max traffic {block_max:.2f}, which is not above 100. "
            "Pick different held-out days so the test set contains real peaks."
        )

    # Training windows: a window covers [s, s + WINDOW_LEN). Keep it only if it ends
    # before the test block starts, or starts at/after the test block ends.
    all_starts = np.arange(len(df) - WINDOW_LEN + 1)
    no_overlap = (all_starts + WINDOW_LEN <= test_start) | (all_starts >= test_end)
    valid_starts = all_starts[no_overlap]

    if len(valid_starts) < N_TRAIN_EXAMPLES:
        raise ValueError("Not enough non-overlapping dataframe rows to create 200 valid windows.")

    rng = np.random.default_rng(123)
    train_starts = [int(s) for s in rng.choice(valid_starts, size=N_TRAIN_EXAMPLES, replace=False)]

    examples = []
    for index, i in enumerate(train_starts):
        history_values = traffic[i : i + HISTORY_LEN]
        future_values = traffic[i + HISTORY_LEN : i + WINDOW_LEN]
        output = build_output(history_values, future_values)
        examples.append({"instruction": build_instruction(history_values), "input": "", "output": output})
        expected = [float(f"{value:.2f}") for value in future_values]
        verify_prediction(output, expected, f"Training example {index}")

    # Test windows: every window fully inside the test block, stepping TEST_STRIDE.
    # The last valid start is test_end - WINDOW_LEN, so range() stops one past it.
    test_windows = []
    for s in range(test_start, test_end - WINDOW_LEN + 1, TEST_STRIDE):
        history_values = [float(f"{v:.2f}") for v in traffic[s : s + HISTORY_LEN]]
        true_values = [float(f"{v:.2f}") for v in traffic[s + HISTORY_LEN : s + WINDOW_LEN]]
        # Same check as training: the PREDICTION line built from this window must
        # parse back to exactly the real dataframe values.
        verify_prediction(build_output(history_values, true_values), true_values, f"Test window {s}")
        test_windows.append({"start": s, "history": history_values, "true": true_values})

    os.makedirs("data", exist_ok=True)
    with open("data/traffic_train.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    with open("data/train_starts.json", "w", encoding="utf-8") as f:
        json.dump(train_starts, f)
    with open("data/test_windows.json", "w", encoding="utf-8") as f:
        json.dump(test_windows, f, indent=2)

    print("All 200 examples verified: PREDICTION values match real dataframe data.")
    print(f"All {len(test_windows)} test windows verified: PREDICTION values match real dataframe data.")
    for index, example in enumerate(examples[:2], start=1):
        print(f"\n--- Example {index} ---")
        print(json.dumps(example, indent=2))

    print(f"\nHeld-out test days: {TEST_DAYS} (timesteps {test_start}-{test_end - 1})")
    print(f"Max traffic in test block: {block_max:.2f}")
    print(f"Training examples: {len(examples)}")
    print(f"Test windows: {len(test_windows)}")


if __name__ == "__main__":
    main()
