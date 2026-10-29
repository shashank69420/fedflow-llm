
"""
FedFlow-LLM — server.py
------------------------
This runs the federated SIMULATION: it starts all 6 clients as
simulated processes on this one Colab GPU, coordinates NUM_ROUNDS
rounds of: send weights out -> clients train locally -> collect
updates back -> average them (FedAvg) -> repeat.

You don't call client.py directly — this file spins up all the
clients itself, in simulation mode, which is the standard way to
run federated learning demos on a single machine.

Outputs (relative to training/):
  outputs/adapter.npz       final-round aggregated LoRA weights, arrays in order
  outputs/loss_history.csv  round,loss for every round of this run
  outputs/run_config.json   v2 runs only (FEDFLOW_VARIANT=v2): variant, max_length, n_examples, rounds,
                            git commit. infer.py reads it; a missing file means a v1 adapter.
                            v2 runs also stop before training if any example would be truncated.
"""

import json
import os
import subprocess

import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from client import DATA_PATH, MAX_LENGTH, MODEL_NAME, VARIANT, FedFlowClient, NUM_CLIENTS
import token_utils

# Round count comes from the environment so Colab can run a 1-round smoke test
# and the 25-round full run without editing this file.
NUM_ROUNDS = int(os.environ.get("FEDFLOW_ROUNDS", "25"))

OUTPUT_DIR = "outputs"
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "adapter.npz")
LOSS_HISTORY_PATH = os.path.join(OUTPUT_DIR, "loss_history.csv")
RUN_CONFIG_PATH = os.path.join(OUTPUT_DIR, "run_config.json")


def git_commit():
    """Commit hash of the code being run, or "unknown" outside a git checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty():
    """True if tracked files differ from the commit (so the hash alone does not describe the code), None if unknown."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def v2_preflight():
    """v2 only: refuse to start if any training example would be truncated, then write run_config.json."""
    from transformers import AutoTokenizer

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # tokenizer files only
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_utils.abort_if_truncated(examples, tokenizer, MAX_LENGTH, "server preflight")
    config = {
        "variant": VARIANT,
        "max_length": MAX_LENGTH,
        "n_examples": len(examples),                         # what data/traffic_train.json actually holds
        "rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "model_name": MODEL_NAME,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "fedflow_n_examples_env": os.environ.get("FEDFLOW_N_EXAMPLES"),   # None when not set
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {RUN_CONFIG_PATH}: {json.dumps(config)}")


class SavingFedAvg(FedAvg):
    """Plain FedAvg, plus: save the final adapter and log each round's loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Start every run with a fresh loss log. This is done here, not at the top
        # of the file, because the strategy is built once per run, while the
        # simulation's worker processes may re-import this module mid-training.
        if os.path.exists(LOSS_HISTORY_PATH):
            os.remove(LOSS_HISTORY_PATH)

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if server_round == NUM_ROUNDS:
            if parameters is None:
                print(f"WARNING: round {server_round} produced no aggregated parameters; adapter not saved.")
            else:
                ndarrays = parameters_to_ndarrays(parameters)
                # savez stores the arrays as arr_0, arr_1, ... in this exact order,
                # which matches the key order of get_peft_model_state_dict.
                np.savez(ADAPTER_PATH, *ndarrays)
                print(f"Saved final adapter ({len(ndarrays)} arrays) to {ADAPTER_PATH}")
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            write_header = not os.path.exists(LOSS_HISTORY_PATH)
            with open(LOSS_HISTORY_PATH, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("round,loss\n")
                f.write(f"{server_round},{loss}\n")
        return loss, metrics


def client_fn(context):
    """Flower calls this once per simulated client to create it.
    context.node_config["partition-id"] tells us which client this is (0 to NUM_CLIENTS-1)."""
    client_id = int(context.node_config["partition-id"])
    return FedFlowClient(client_id).to_client()


def server_fn(context):
    """Defines the FedAvg strategy and how many rounds to run."""
    strategy = SavingFedAvg(
        fraction_fit=1.0,          # use all 6 clients every round
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)


client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp(server_fn=server_fn)


if __name__ == "__main__":
    if VARIANT != "v1":
        v2_preflight()   # v1 (default) skips this, so a v1 run writes exactly the files it always did
    print(f"Starting FedFlow-LLM simulation: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, variant {VARIANT}, "
          f"max_length {MAX_LENGTH}.")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1.0 / NUM_CLIENTS}},
    )
