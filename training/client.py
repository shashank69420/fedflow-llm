
"""
FedFlow-LLM — client.py
------------------------
This defines what ONE simulated client (site) does:
  1. Load the shared base model + a small LoRA adapter on top of it.
  2. When the server sends the current adapter weights, load them in.
  3. Train locally, for a few steps, on this client's own slice of data.
  4. Send the updated adapter weights back to the server.

Nothing here is Flower-specific magic — it's a normal PyTorch/HuggingFace
training loop, wrapped in the shape Flower expects (a "NumPyClient").

Variant switches (token_utils.py, PLAN_V2.md):
  FEDFLOW_VARIANT=v1 (default)  unchanged: loss on every token, labels = input_ids
  FEDFLOW_VARIANT=v2            loss on the answer and end token only (prompt and padding labels = -100),
                                and training stops before any example would be truncated
  FEDFLOW_MAX_LENGTH            token limit per example (default 256, v1's value)
"""

# Must be set BEFORE importing transformers/peft — these environment
# variables tell transformers to skip its optional vision/audio code
# paths entirely, rather than trying to import torchvision/torchaudio.
# This project is text-only (traffic forecasting), so none of this is
# needed — and unlike patching torchvision/torchaudio one at a time,
# this blocks the whole multimodal import path in one shot.
import os
os.environ["USE_TORCHVISION"] = "0"
os.environ["USE_TORCHAUDIO"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import json
import sys

import flwr as fl
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

import token_utils

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # matches the Objectives slide's stated target
DATA_PATH = "data/traffic_train.json"
NUM_CLIENTS = 6
LOCAL_EPOCHS = 1          # how many passes over this client's own data slice, per round
LOCAL_BATCH_SIZE = 2
MAX_LENGTH = token_utils.max_length_from_env()   # max tokens per training example; default 256 (v1)
VARIANT = token_utils.variant_from_env()          # "v1" (default) or "v2"


def load_model_and_tokenizer():
    """Load the base model once, and wrap it with a small LoRA adapter.
    Only the LoRA adapter's weights get trained/shared — the base model
    itself stays frozen, which is what keeps this small and fast."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # standard, small, fast LoRA target
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def load_client_data(client_id: int, num_clients: int):
    """Split the shared training JSON into num_clients roughly-equal
    slices, and return only this client's slice — simulating each
    client only ever seeing its own local data."""
    with open(DATA_PATH, "r") as f:
        all_examples = json.load(f)

    # Deterministic split: client_id 0 gets every 0th, num_clients-th example, etc.
    my_examples = all_examples[client_id::num_clients]
    return my_examples


def format_example(example, tokenizer):
    """Turn one {"instruction","input","output"} example into a single
    tokenized training string: instruction followed by the expected output.
    v2 also returns "labels" with the prompt and padding masked out (token_utils.v2_token_lists)."""
    if VARIANT == "v2":
        input_ids, attention_mask, labels = token_utils.v2_token_lists(example, tokenizer, MAX_LENGTH)
        # Same shapes as the v1 tokenizer output below: a batch of one.
        return {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.tensor([attention_mask]),
                "labels": torch.tensor([labels])}
    full_text = example["instruction"] + "\n" + example["output"] + tokenizer.eos_token
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    return tokens


class FedFlowClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model, self.tokenizer = load_model_and_tokenizer()
        self.examples = load_client_data(client_id, NUM_CLIENTS)
        if VARIANT == "v2":
            # Stop before training if any example would not fit; v2 never truncates.
            token_utils.abort_if_truncated(self.examples, self.tokenizer, MAX_LENGTH, f"Client {client_id}")
        print(f"[Client {client_id}] loaded {len(self.examples)} local examples.")

    def get_parameters(self, config):
        """Send the current LoRA adapter weights to the server, as a list of numpy arrays."""
        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for val in state_dict.values()]

    def set_parameters(self, parameters):
        """Receive the server's current (averaged) LoRA weights and load them into this client's model."""
        state_dict = get_peft_model_state_dict(self.model)
        new_state_dict = {
            key: torch.tensor(val) for key, val in zip(state_dict.keys(), parameters)
        }
        set_peft_model_state_dict(self.model, new_state_dict)

    def fit(self, parameters, config):
        """This is the core federated step:
        1. Load the server's current weights.
        2. Train locally on this client's own data slice.
        3. Return the updated weights, plus how many examples were trained on
           (Flower uses this count to weight the averaging — FedAvg)."""
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        total_loss = 0.0
        num_steps = 0

        for epoch in range(LOCAL_EPOCHS):
            for i in range(0, len(self.examples), LOCAL_BATCH_SIZE):
                batch = self.examples[i : i + LOCAL_BATCH_SIZE]
                if not batch:
                    continue

                for example in batch:
                    tokens = format_example(example, self.tokenizer)
                    input_ids = tokens["input_ids"].to("cuda")
                    attention_mask = tokens["attention_mask"].to("cuda")
                    # v1: labels = input_ids (unchanged). v2: answer-only labels from format_example.
                    labels = tokens["labels"].to("cuda") if "labels" in tokens else input_ids

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        print(f"[Client {self.client_id}] local training done. avg_loss={avg_loss:.4f}")

        return self.get_parameters(config={}), len(self.examples), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        """Minimal evaluation — reuses training loss as a rough signal.
        Good enough for a demo-scale project; not a held-out test set."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        num_steps = 0
        with torch.no_grad():
            for example in self.examples[:10]:  # quick check on a small subset
                tokens = format_example(example, self.tokenizer)
                input_ids = tokens["input_ids"].to("cuda")
                attention_mask = tokens["attention_mask"].to("cuda")
                labels = tokens["labels"].to("cuda") if "labels" in tokens else input_ids   # same rule as fit()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += outputs.loss.item()
                num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        return avg_loss, len(self.examples), {"loss": avg_loss}


def main():
    # client_id is passed as a command-line argument: python client.py 0
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    client = FedFlowClient(client_id)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()