"""
FedFlow-LLM — token_audit_v2.py
--------------------------------
Measures v2 token lengths with the real Qwen2.5-1.5B-Instruct TOKENIZER (no model weights,
no torch, no GPU) and picks v2's training length limit by the rule in PLAN_V2.md:
the smallest multiple of 32 that is at least 16 above the longest text measured.

Measured (all built in memory; nothing in data/telecom/ or datasets/ is read back or written):
  v2 training examples, 200 (default) and 400 (attempt 2): full training text
  v2 validation and test windows: the prompt infer.py feeds in, and the full text
  v1 telecom and synthetic training examples: full text, to confirm v1 fits its own 256 limit
Full text = instruction + "\\n" + output + eos (what client.py trains on).
Prompt    = instruction + "\\n" (what infer.py feeds in).

Usage (from inside training/):
  python3 token_audit_v2.py      prints the table and writes data/telecom_v2/token_audit_v2.txt
"""
import math
import os

import generate_data as gd
import token_utils

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # same as client.py / infer.py
MARGIN = 16                                 # PLAN_V2: at least 16 above the longest
MULTIPLE = 32                               # PLAN_V2: a multiple of 32
OUT_PATH = os.path.join(gd.TELECOM_OUT_V2, "token_audit_v2.txt")


def choose_max_length(longest):
    """Smallest multiple of 32 that is >= longest + 16."""
    return MULTIPLE * math.ceil((longest + MARGIN) / MULTIPLE)


def window_example(window):
    """Prompt and answer of a v2 validation/test window, built through the shared build_example."""
    return gd.build_example(window["history"], window["true"], "v2", last_slot=window["start"] + gd.HISTORY_LEN - 1)


def main():
    import transformers
    from transformers import AutoTokenizer

    import token_audit   # v1 audit helpers; only synthetic_sets() is used, which works in a temp folder

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # tokenizer files only, never weights
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    v1 = gd.build_telecom()
    v2_200 = gd.build_telecom_v2(200, v1)
    v2_400 = gd.build_telecom_v2(400, v1)
    val = [window_example(w) for w in v2_200["val_windows"]]
    test = [window_example(w) for w in v2_200["test_windows"]]
    synthetic_train = [{"instruction": p, "output": o} for p, o in token_audit.synthetic_sets()["train"]]

    full = lambda examples: [token_utils.full_length(e, tokenizer) for e in examples]
    prompt = lambda examples: [len(tokenizer(token_utils.prompt_text(e))["input_ids"]) for e in examples]
    rows = [
        ("v2 train (200)", "full", full(v2_200["examples"])),
        ("v2 train (400)", "full", full(v2_400["examples"])),
        ("v2 validation", "prompt", prompt(val)),
        ("v2 validation", "full", full(val)),
        ("v2 test", "prompt", prompt(test)),
        ("v2 test", "full", full(test)),
        ("v1 telecom train", "full", full(v1["examples"])),
        ("v1 synthetic train", "full", full(synthetic_train)),
    ]
    v2_longest = max(max(n) for name, _, n in rows if name.startswith("v2"))
    max_length = choose_max_length(v2_longest)
    v1_longest = max(max(n) for name, _, n in rows if name.startswith("v1"))

    lines = [
        "v2 token audit (tokenizer only, no model weights, no GPU)",
        f"Tokenizer: {MODEL_NAME}, transformers {transformers.__version__}, eos {tokenizer.eos_token!r}, pad {tokenizer.pad_token!r}",
        "full = instruction + newline + output + eos (training text); prompt = instruction + newline (inference input)",
        "",
        f"{'set':<20}{'text':<8}{'n':>5}{'min':>6}{'max':>6}",
    ]
    for name, kind, n in rows:
        lines.append(f"{name:<20}{kind:<8}{len(n):>5}{min(n):>6}{max(n):>6}")
    lines += [
        "",
        f"Longest v2 text measured (any row starting 'v2'): {v2_longest} tokens",
        f"Rule: smallest multiple of {MULTIPLE} that is >= longest + {MARGIN} = {v2_longest + MARGIN}",
        f"v2 MAX_LENGTH: {max_length}  (set FEDFLOW_MAX_LENGTH={max_length})",
        f"Longest v1 text measured: {v1_longest} tokens; v1 limit 256; v1 examples over 256: "
        f"{sum(1 for name, _, n in rows if name.startswith('v1') for x in n if x > 256)}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(gd.TELECOM_OUT_V2, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
