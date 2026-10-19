"""
FedFlow-LLM — check_split.py
-----------------------------
Confirms that no training window shares a single timestep with any test window.
Run from inside training/ after generate_data.py.
"""
import json

WINDOW_LEN = 18  # 12 history + 6 future, same as generate_data.py

with open("data/train_starts.json", "r", encoding="utf-8") as f:
    train_starts = json.load(f)
with open("data/test_windows.json", "r", encoding="utf-8") as f:
    test_starts = [w["start"] for w in json.load(f)]

overlap_count = 0
for a in train_starts:
    for b in test_starts:
        # Windows [a, a+18) and [b, b+18) overlap when each starts before the other ends.
        if a < b + WINDOW_LEN and b < a + WINDOW_LEN:
            overlap_count += 1

print(f"training windows: {len(train_starts)}, test windows: {len(test_starts)}")
print(f"overlap count: {overlap_count}")
assert overlap_count == 0, f"FAIL: {overlap_count} training/test window pairs overlap."
print("PASS")
