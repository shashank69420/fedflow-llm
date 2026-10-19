
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
"""

import flwr as fl
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from client import FedFlowClient, NUM_CLIENTS

NUM_ROUNDS = 25  # Real evidence run — both 0.5B and 1.5B confirmed working at this round count.


def client_fn(context):
    """Flower calls this once per simulated client to create it.
    context.node_config["partition-id"] tells us which client this is (0 to NUM_CLIENTS-1)."""
    client_id = int(context.node_config["partition-id"])
    return FedFlowClient(client_id).to_client()


def server_fn(context):
    """Defines the FedAvg strategy and how many rounds to run."""
    strategy = FedAvg(
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
    print(f"Starting FedFlow-LLM simulation: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds.")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1.0 / NUM_CLIENTS}},
    )