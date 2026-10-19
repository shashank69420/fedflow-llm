# Cell 1: Generate synthetic traffic data.
import os
import numpy as np
import pandas as pd

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


# Create and verify instruction-tuning examples from real dataframe values.
import json
import os
import re
import numpy as np
import pandas as pd

df = pd.read_csv("data/synthetic_traffic.csv", usecols=["timestep", "traffic"])
rng = np.random.default_rng(123)
n_examples = 200
valid_starts = np.arange(len(df) - 18 + 1)

if len(valid_starts) < n_examples:
    raise ValueError("Not enough dataframe rows to create 200 valid windows.")

examples = []
expected_predictions = []

for i in rng.choice(valid_starts, size=n_examples, replace=False):
    history_values = df["traffic"].iloc[i : i + 12].tolist()
    future_values = df["traffic"].iloc[i + 12 : i + 18].tolist()

    slope = float(np.mean(history_values[-3:]) - np.mean(history_values[:3]))
    if slope > 1:
        trend = "increasing"
    elif slope < -1:
        trend = "decreasing"
    else:
        trend = "stable"

    history_text = ", ".join(f"{value:.2f}" for value in history_values)
    prediction_text = ", ".join(f"{value:.2f}" for value in future_values)

    instruction = (
        "You are a traffic-forecasting assistant. Predict the next 6 traffic values "
        "given the following 12 historical values measured at 10-minute intervals.\n\n"
        f"History: {history_text}\n\n"
        "Reason step by step about the traffic pattern, then end with exactly this format:\n"
        "PREDICTION: v1, v2, v3, v4, v5, v6"
    )
    output = (
        f"Trend direction: {trend}.\n"
        f"Justification: The history slope (mean of last 3 minus mean of first 3) is {slope:.2f}.\n"
        f"PREDICTION: {prediction_text}"
    )

    examples.append({"instruction": instruction, "input": "", "output": output})
    expected_predictions.append([float(f"{value:.2f}") for value in future_values])

for index, (example, expected) in enumerate(zip(examples, expected_predictions)):
    match = re.search(r"(?m)^PREDICTION:\s*(.+)$", example["output"])
    if not match:
        raise AssertionError(f"Example {index} failed: PREDICTION line not found.")

    try:
        parsed = [float(value.strip()) for value in match.group(1).split(",")]
    except ValueError as error:
        raise AssertionError(f"Example {index} failed: invalid PREDICTION values.") from error

    if len(parsed) != 6 or parsed != expected:
        raise AssertionError(
            f"Example {index} failed: parsed {parsed} does not match expected {expected}."
        )

os.makedirs("data", exist_ok=True)
with open("data/traffic_train.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)

print("All 200 examples verified: PREDICTION values match real dataframe data.")
for index, example in enumerate(examples[:2], start=1):
    print(f"\n--- Example {index} ---")
    print(json.dumps(example, indent=2))
