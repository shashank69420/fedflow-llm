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

With --dataset telecom it instead reads datasets/telecom_milan/cells.csv (made by
prepare_telecom.py) and writes the same four JSON files, plus val_windows.json,
into data/telecom/ so the synthetic copies in data/ are never overwritten.
"""
import argparse
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


# ---------------------------------------------------------------- telecom dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELECOM_DIR = os.path.join(REPO, "datasets", "telecom_milan")   # where cells.csv and cells_meta.json live
TELECOM_OUT = os.path.join("data", "telecom")                   # output folder, relative to training/
N_CELLS = 6
N_SLOTS = 2016                 # 14 days x 144 ten-minute slots
TRAIN_END, VAL_END = 1440, 1728  # training = slots 0-1439, validation = 1440-1727, test = 1728-2015
VAL_STRIDE = 6                 # validation windows start every 6 slots, for every cell
TEST_STRIDE_EXPERIMENT = 3     # test windows of the experiment cell start every 3 slots
TEST_STRIDE_OTHER = 6          # test windows of the other cells start every 6 slots
TELECOM_SEED = 123             # training starts of cell c use seed 123 + c


def load_telecom_cells():
    """Read cells.csv and cells_meta.json. Returns (meta, squares, series) with one list of floats per cell."""
    import pandas as pd

    with open(os.path.join(TELECOM_DIR, "cells_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    squares = [int(s) for s in meta["squares"]]
    df = pd.read_csv(os.path.join(TELECOM_DIR, "cells.csv"))
    # The CSV must hold exactly the timestep column plus the six chosen squares, in cell order.
    if list(df.columns) != ["timestep"] + [str(s) for s in squares]:
        raise ValueError(f"cells.csv columns {list(df.columns)} do not match cells_meta.json squares {squares}.")
    if df["timestep"].tolist() != list(range(N_SLOTS)):
        raise ValueError(f"cells.csv must have timesteps 0..{N_SLOTS - 1} in order.")
    # Round-trip every value through .2f so the prompt text and the stored truth agree exactly.
    series = [[float(f"{v:.2f}") for v in df[str(s)].tolist()] for s in squares]
    return meta, squares, series


def telecom_window(series, cell, square, start):
    """One validation/test entry: history = 12 values from `start`, true = the 6 values after them."""
    return {
        "cell": cell,
        "square_id": square,
        "start": start,
        "history": series[cell][start : start + HISTORY_LEN],
        "true": series[cell][start + HISTORY_LEN : start + WINDOW_LEN],
    }


def build_telecom():
    """Build everything for the telecom dataset in memory and return it as a dict.
    Nothing is written here, so token_audit.py can reuse this without touching disk."""
    meta, squares, series = load_telecom_cells()
    experiment_cell = squares.index(int(meta["experiment_cell"]))

    # Training starts: windows must fit completely inside training slots 0-1439.
    valid_starts = np.arange(0, TRAIN_END - WINDOW_LEN + 1)
    # Example j belongs to cell j % 6, so client.py's examples[client_id::6] gives client i only cell i.
    per_cell_count = [len(range(c, N_TRAIN_EXAMPLES, N_CELLS)) for c in range(N_CELLS)]
    starts_by_cell = []
    for c in range(N_CELLS):
        rng = np.random.default_rng(TELECOM_SEED + c)   # one generator per cell
        # replace=False means the starts within a cell are all different.
        starts_by_cell.append([int(s) for s in rng.choice(valid_starts, size=per_cell_count[c], replace=False)])

    examples, train_starts = [], []
    for j in range(N_TRAIN_EXAMPLES):
        c = j % N_CELLS
        start = starts_by_cell[c][j // N_CELLS]         # j // 6 = this example's position inside its cell
        history_values = series[c][start : start + HISTORY_LEN]
        future_values = series[c][start + HISTORY_LEN : start + WINDOW_LEN]
        output = build_output(history_values, future_values)
        examples.append({"instruction": build_instruction(history_values), "input": "", "output": output})
        verify_prediction(output, future_values, f"Telecom training example {j}")
        train_starts.append({"cell": c, "square_id": squares[c], "start": start})

    val_windows = []
    for start in range(TRAIN_END, VAL_END - WINDOW_LEN + 1, VAL_STRIDE):   # last start = 1710, so the window ends at 1727
        for c in range(N_CELLS):
            window = telecom_window(series, c, squares[c], start)
            verify_prediction(build_output(window["history"], window["true"]), window["true"],
                              f"Telecom validation window cell {c} start {start}")
            val_windows.append(window)

    test_windows = []
    for start in range(VAL_END, N_SLOTS - WINDOW_LEN + 1, TEST_STRIDE_EXPERIMENT):   # last start = 1998
        for c in range(N_CELLS):
            # The experiment cell uses every 3rd start; the other cells only every 6th.
            if c != experiment_cell and (start - VAL_END) % TEST_STRIDE_OTHER != 0:
                continue
            window = telecom_window(series, c, squares[c], start)
            verify_prediction(build_output(window["history"], window["true"]), window["true"],
                              f"Telecom test window cell {c} start {start}")
            test_windows.append(window)

    return {"meta": meta, "squares": squares, "experiment_cell": experiment_cell, "series": series,
            "examples": examples, "train_starts": train_starts,
            "val_windows": val_windows, "test_windows": test_windows}


def main_telecom():
    data = build_telecom()
    os.makedirs(TELECOM_OUT, exist_ok=True)
    for name, payload in [("traffic_train.json", data["examples"]), ("train_starts.json", data["train_starts"]),
                          ("val_windows.json", data["val_windows"]), ("test_windows.json", data["test_windows"])]:
        with open(os.path.join(TELECOM_OUT, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"All {len(data['examples'])} training examples verified: PREDICTION values match cells.csv.")
    print(f"All {len(data['val_windows'])} validation windows verified: PREDICTION values match cells.csv.")
    print(f"All {len(data['test_windows'])} test windows verified: PREDICTION values match cells.csv.")
    print(f"\nCells (square IDs): {data['squares']}; experiment cell: square {data['squares'][data['experiment_cell']]}")
    for c, square in enumerate(data["squares"]):
        n_train = sum(1 for t in data["train_starts"] if t["cell"] == c)
        n_val = sum(1 for w in data["val_windows"] if w["cell"] == c)
        n_test = sum(1 for w in data["test_windows"] if w["cell"] == c)
        print(f"  cell {c} (square {square}): training {n_train}, validation {n_val}, test {n_test}")
    for index, example in enumerate(data["examples"][:2], start=1):
        print(f"\n--- Example {index} (cell {data['train_starts'][index - 1]['cell']}) ---")
        print(json.dumps(example, indent=2))
    print(f"\nWrote {TELECOM_OUT}/ traffic_train.json, train_starts.json, val_windows.json, test_windows.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedFlow-LLM data generation")
    parser.add_argument("--dataset", choices=["synthetic", "telecom"], default="synthetic",
                        help="synthetic (default, unchanged) or telecom (reads datasets/telecom_milan/cells.csv)")
    args = parser.parse_args()
    if args.dataset == "telecom":
        main_telecom()
    else:
        main()
