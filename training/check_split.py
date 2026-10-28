"""
FedFlow-LLM — check_split.py
-----------------------------
Confirms that the data split has no leaks.
Run from inside training/ after generate_data.py.

  python check_split.py                     synthetic: no training window shares a timestep with any test window
  python check_split.py --dataset telecom   telecom: training/validation/test windows share no slot, and
                                            each client's slice holds only its own cell
"""
import argparse
import json

WINDOW_LEN = 18  # 12 history + 6 future, same as generate_data.py


def check_synthetic():
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


def check_telecom():
    # Imported here so the synthetic check keeps working without pandas.
    import generate_data as gd

    folder = gd.TELECOM_OUT
    with open(f"{folder}/traffic_train.json", "r", encoding="utf-8") as f:
        examples = json.load(f)
    with open(f"{folder}/train_starts.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(f"{folder}/val_windows.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open(f"{folder}/test_windows.json", "r", encoding="utf-8") as f:
        test = json.load(f)
    _, squares, series = gd.load_telecom_cells()

    def slots(entries):
        """Every slot number covered by any window in the list (all cells share one timeline)."""
        return {t for e in entries for t in range(e["start"], e["start"] + WINDOW_LEN)}

    train_slots, val_slots, test_slots = slots(train), slots(val), slots(test)
    print(f"training windows: {len(train)}, validation windows: {len(val)}, test windows: {len(test)}")
    print(f"training slots used: {min(train_slots)}-{max(train_slots)}; "
          f"validation: {min(val_slots)}-{max(val_slots)}; test: {min(test_slots)}-{max(test_slots)}")

    # Each set must stay inside its own block of the timeline.
    assert max(train_slots) < gd.TRAIN_END, "FAIL: a training window reaches into slot 1440 or later."
    assert gd.TRAIN_END <= min(val_slots) and max(val_slots) < gd.VAL_END, "FAIL: a validation window leaves slots 1440-1727."
    assert gd.VAL_END <= min(test_slots) and max(test_slots) < gd.N_SLOTS, "FAIL: a test window leaves slots 1728-2015."
    # Direct overlap counts, in case the block checks above are ever loosened.
    assert not (train_slots & val_slots), "FAIL: a training window touches a validation slot."
    assert not (train_slots & test_slots), "FAIL: a training window touches a test slot."
    assert not (val_slots & test_slots), "FAIL: a validation window touches a test slot."
    print("training/validation/test slot overlap: 0 / 0 / 0")

    # Every window must be real data of its own cell (history and truth match cells.csv).
    for name, entries in [("validation", val), ("test", test)]:
        for e in entries:
            c, s = e["cell"], e["start"]
            assert squares[c] == e["square_id"], f"FAIL: {name} window cell {c} has the wrong square ID."
            assert e["history"] == series[c][s : s + 12] and e["true"] == series[c][s + 12 : s + 18], \
                f"FAIL: {name} window cell {c} start {s} does not match cells.csv."

    # Client i trains on examples[i::6] (client.py). That slice must hold only cell i.
    assert len(examples) == len(train), "FAIL: traffic_train.json and train_starts.json have different lengths."
    for client in range(gd.N_CELLS):
        for j in range(client, len(examples), gd.N_CELLS):
            t = train[j]
            assert t["cell"] == client, f"FAIL: example {j} (client {client}) belongs to cell {t['cell']}."
            history = series[client][t["start"] : t["start"] + 12]
            future = series[client][t["start"] + 12 : t["start"] + 18]
            assert examples[j]["instruction"] == gd.build_instruction(history), f"FAIL: example {j} history is not cell {client}'s data."
            assert examples[j]["output"] == gd.build_output(history, future), f"FAIL: example {j} answer is not cell {client}'s data."
        starts = [train[j]["start"] for j in range(client, len(examples), gd.N_CELLS)]
        assert len(set(starts)) == len(starts), f"FAIL: client {client} has repeated window starts."
        print(f"client {client}: {len(starts)} examples, all from cell {client} (square {squares[client]}), starts distinct")
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedFlow-LLM split check")
    parser.add_argument("--dataset", choices=["synthetic", "telecom"], default="synthetic")
    args = parser.parse_args()
    check_telecom() if args.dataset == "telecom" else check_synthetic()
