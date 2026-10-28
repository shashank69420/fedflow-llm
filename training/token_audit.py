"""
FedFlow-LLM — token_audit.py
-----------------------------
Measures how many tokens each example needs, using the real Qwen2.5-1.5B-Instruct
TOKENIZER (no model weights are loaded), so client.py's MAX_LENGTH and infer.py's
generation limit can be checked against the data.

Text measured = exactly what client.py trains on: instruction + "\\n" + output + eos.
Answer tokens  = tokens the model has to generate = full text minus the prompt that
                 infer.py feeds in (instruction + "\\n"). It includes the end token.

Covers synthetic (regenerated in a temp folder) and telecom (built in memory).
Needs: transformers (tokenizer only), numpy, pandas. No GPU, no torch.

Usage (from inside training/):
  python token_audit.py                      prints the table and writes datasets/telecom_milan/token_audit.txt
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile

import generate_data as gd

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # same as client.py / infer.py
MAX_LENGTH = 256                            # client.py's default training length
MAX_NEW_TOKENS = 160                        # infer.py's default generation limit
OUT_PATH = os.path.join(gd.TELECOM_DIR, "token_audit.txt")


def synthetic_sets():
    """Regenerate the synthetic data in a throw-away folder and return {split: [(prompt, output)]}."""
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)                                    # generate_data.main() writes to ./data, so run it in the temp folder
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # hide its printout
                gd.main()
            with open("data/traffic_train.json", "r", encoding="utf-8") as f:
                train = json.load(f)
            with open("data/test_windows.json", "r", encoding="utf-8") as f:
                test = json.load(f)
        finally:
            os.chdir(old_cwd)
    return {
        "train": [(e["instruction"], e["output"]) for e in train],
        "test": [(gd.build_instruction(w["history"]), gd.build_output(w["history"], w["true"])) for w in test],
    }


def telecom_sets():
    """Build the telecom data in memory and return {split: [(prompt, output)]}."""
    data = gd.build_telecom()
    windows = lambda ws: [(gd.build_instruction(w["history"]), gd.build_output(w["history"], w["true"])) for w in ws]
    return {
        "train": [(e["instruction"], e["output"]) for e in data["examples"]],
        "val": windows(data["val_windows"]),
        "test": windows(data["test_windows"]),
    }


def percentile_99(values):
    """99th percentile by the nearest-rank method (the ceil(0.99 * n)-th smallest value)."""
    ordered = sorted(values)
    return ordered[math.ceil(0.99 * len(ordered)) - 1]


def measure(tokenizer, pairs):
    """Returns (full_lengths, answer_lengths) for a list of (instruction, output) pairs."""
    full_lengths, answer_lengths = [], []
    for instruction, output in pairs:
        full = len(tokenizer(instruction + "\n" + output + tokenizer.eos_token)["input_ids"])   # client.py's training text
        prompt = len(tokenizer(instruction + "\n")["input_ids"])                                  # infer.py's prompt
        full_lengths.append(full)
        answer_lengths.append(full - prompt)
    return full_lengths, answer_lengths


def main():
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # downloads tokenizer files only, never weights
    lines = [
        "Token audit (tokenizer only, no model weights)",
        f"Tokenizer: {MODEL_NAME}, transformers {transformers.__version__}, eos token {tokenizer.eos_token!r}",
        f"Full text = instruction + newline + output + eos, as client.py trains. Training length limit: {MAX_LENGTH}.",
        f"Answer tokens = tokens the model must generate (includes the end token). infer.py default limit: {MAX_NEW_TOKENS};",
        "a generation of exactly the limit is treated as truncated, so answers must be shorter than the limit.",
        "99th percentile = nearest rank, the ceil(0.99 n)-th smallest value.",
        "",
        f"{'dataset':<10}{'split':<7}{'n':>5}{'max':>6}{'p99':>6}{'>256':>6}   {'answer max':>10}{'answer p99':>11}{'answer >=160':>13}",
    ]
    results = {}
    for dataset, sets in [("synthetic", synthetic_sets()), ("telecom", telecom_sets())]:
        for split, pairs in sets.items():
            full, answer = measure(tokenizer, pairs)
            results[(dataset, split)] = (full, answer)
            lines.append(f"{dataset:<10}{split:<7}{len(full):>5}{max(full):>6}{percentile_99(full):>6}"
                         f"{sum(1 for n in full if n > MAX_LENGTH):>6}   {max(answer):>10}{percentile_99(answer):>11}"
                         f"{sum(1 for n in answer if n >= MAX_NEW_TOKENS):>13}")

    telecom_max = max(max(results[("telecom", s)][0]) for s in ("train", "val", "test"))
    lines += ["", f"Longest telecom full text (any split): {telecom_max} tokens"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
