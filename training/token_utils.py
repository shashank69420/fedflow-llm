"""
FedFlow-LLM — token_utils.py
-----------------------------
Variant switches and the v2 answer-only labels (PLAN_V2.md, change (a)), kept free of
torch so they can be tested with the tokenizer alone.

  FEDFLOW_VARIANT     v1 (default, unchanged training) or v2
  FEDFLOW_MAX_LENGTH  training length limit in tokens (default 256, v1's value)

v2 labels: -100 for every prompt token and every padding token; the real token id for the
answer tokens and the end token. The prompt is instruction + "\\n", exactly what infer.py feeds in.

Selftest (tokenizer only, no model weights, no GPU), from inside training/:
  python3 token_utils.py --selftest
"""
import os
import sys

VARIANTS = ("v1", "v2")
DEFAULT_MAX_LENGTH = 256   # v1's MAX_LENGTH
IGNORE_INDEX = -100        # label value that PyTorch's cross-entropy loss skips


def variant_from_env():
    """FEDFLOW_VARIANT, default v1. Anything else is an error, never a silent fallback."""
    variant = os.environ.get("FEDFLOW_VARIANT", "v1")
    if variant not in VARIANTS:
        raise SystemExit(f"FEDFLOW_VARIANT={variant!r} is not one of {VARIANTS}.")
    return variant


def max_length_from_env():
    """FEDFLOW_MAX_LENGTH, default 256 (v1)."""
    value = int(os.environ.get("FEDFLOW_MAX_LENGTH", str(DEFAULT_MAX_LENGTH)))
    if value < 1:
        raise SystemExit(f"FEDFLOW_MAX_LENGTH={value} must be at least 1.")
    return value


def prompt_text(example):
    """What infer.py feeds the model: the instruction and a newline."""
    return example["instruction"] + "\n"


def full_text(example, tokenizer):
    """What client.py trains on: prompt, answer, end token (the same string in v1 and v2)."""
    return prompt_text(example) + example["output"] + tokenizer.eos_token


def full_length(example, tokenizer):
    """Token count of the training text, without truncation."""
    return len(tokenizer(full_text(example, tokenizer))["input_ids"])


def too_long(examples, tokenizer, max_length):
    """[(index, length)] of every example whose training text is longer than max_length tokens."""
    lengths = [full_length(e, tokenizer) for e in examples]
    return [(i, n) for i, n in enumerate(lengths) if n > max_length]


def abort_if_truncated(examples, tokenizer, max_length, where):
    """Stop training with a clear message if any example would be cut. Never truncate silently."""
    bad = too_long(examples, tokenizer, max_length)
    if bad:
        shown = ", ".join(f"example {i}: {n} tokens" for i, n in bad[:5])
        raise SystemExit(f"[{where}] ABORT: {len(bad)} training example(s) longer than MAX_LENGTH={max_length} "
                         f"({shown}). Raise FEDFLOW_MAX_LENGTH; nothing was truncated.")


def v2_token_lists(example, tokenizer, max_length):
    """input_ids, attention_mask and labels (plain lists, length max_length) for one v2 example.
    Raises ValueError instead of truncating."""
    padded = tokenizer(full_text(example, tokenizer), padding="max_length", max_length=max_length, truncation=False)
    input_ids, attention_mask = padded["input_ids"], padded["attention_mask"]
    if len(input_ids) > max_length:
        raise ValueError(f"example is {len(input_ids)} tokens, longer than max_length={max_length}; refusing to truncate")
    real_ids = [t for t, m in zip(input_ids, attention_mask) if m == 1]     # the text's own tokens, padding removed
    prompt_ids = tokenizer(prompt_text(example))["input_ids"]
    # The mask is only right if the prompt tokenizes the same alone as inside the full text.
    if real_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens differ inside the full text; the prompt/answer boundary is not clean")
    labels, k = [], 0          # k counts real (non-padding) tokens seen so far
    for token, mask in zip(input_ids, attention_mask):
        if mask == 0:
            labels.append(IGNORE_INDEX)        # padding
        else:
            labels.append(IGNORE_INDEX if k < len(prompt_ids) else token)   # prompt -> ignored; answer and end token -> kept
            k += 1
    return input_ids, attention_mask, labels


def run_selftest():
    """Build one real v2 example and check the labels with the real Qwen tokenizer (no model weights)."""
    from transformers import AutoTokenizer

    import generate_data as gd

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")   # tokenizer files only
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token        # same rule as client.py
    data = gd.build_telecom_v2()                         # in memory; writes nothing
    example = data["examples"][0]
    max_length = 320                                     # any limit that fits the example works for this check
    input_ids, attention_mask, labels = v2_token_lists(example, tokenizer, max_length)

    n_prompt = len(tokenizer(prompt_text(example))["input_ids"])
    n_real = sum(attention_mask)
    n_padding = max_length - n_real
    answer_ids = tokenizer(example["output"])["input_ids"]   # answer tokens on their own
    masked = sum(1 for l in labels if l == IGNORE_INDEX)
    unmasked = [l for l in labels if l != IGNORE_INDEX]
    assert len(input_ids) == len(labels) == max_length
    assert masked == n_prompt + n_padding, (masked, n_prompt, n_padding)
    assert len(unmasked) == len(answer_ids) + 1, (len(unmasked), len(answer_ids))   # answer tokens + the end token
    assert unmasked[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(unmasked[:-1]) == example["output"]                 # the answer text, exactly
    assert tokenizer.decode(unmasked) == example["output"] + tokenizer.eos_token
    print(f"v2 label check: {max_length} positions = {n_prompt} prompt + {n_padding} padding masked, "
          f"{len(answer_ids)} answer + 1 end token kept")

    # Truncation guard: a limit below the example's length must be refused, never cut.
    try:
        v2_token_lists(example, tokenizer, n_real - 1)
        raise AssertionError("v2_token_lists accepted an example longer than max_length")
    except ValueError:
        pass
    assert too_long([example], tokenizer, n_real - 1) == [(0, n_real)] and too_long([example], tokenizer, n_real) == []
    try:
        abort_if_truncated([example], tokenizer, n_real - 1, "selftest")
        raise AssertionError("abort_if_truncated did not abort")
    except SystemExit:
        pass

    # Switches: defaults are v1 / 256, bad values stop.
    saved = {k: os.environ.pop(k, None) for k in ("FEDFLOW_VARIANT", "FEDFLOW_MAX_LENGTH")}
    try:
        assert variant_from_env() == "v1" and max_length_from_env() == DEFAULT_MAX_LENGTH
        os.environ["FEDFLOW_VARIANT"] = "v3"
        try:
            variant_from_env()
            raise AssertionError("bad FEDFLOW_VARIANT accepted")
        except SystemExit:
            pass
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    print("SELFTEST OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        run_selftest()
    else:
        print(__doc__)
