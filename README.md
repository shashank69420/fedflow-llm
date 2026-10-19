# FedFlow-LLM

FedFlow-LLM fine-tunes a small instruction-tuned LLM (Qwen2.5-Instruct) for network traffic forecasting with federated learning. A synthetic traffic series (14 days at 10-minute intervals, with daily cycles, weekend dips and random bursts) is turned into 200 instruction-tuning examples, each asking the model to predict the next 6 values from 12 historical ones. Flower simulates 6 clients on a single GPU. Each client trains a LoRA adapter (r=8, on `q_proj`/`v_proj`) on its own slice of the data, and the server combines the adapters with FedAvg each round. Only the adapter weights are trained and exchanged; the base model stays frozen.

## Setup (Google Colab, GPU runtime)

Run these in order:

```bash
pip install -q torch transformers peft accelerate bitsandbytes "flwr[simulation]"
pip uninstall -y torchvision torchaudio
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.46.3 peft==0.13.2
```

## How to run

From inside `training/`:

```bash
python generate_data.py   # writes data/synthetic_traffic.csv and data/traffic_train.json
python server.py          # runs the 6-client, 25-round federated simulation
```

`client.py` currently sets `MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"`. To reproduce the 0.5B run, change it to `Qwen/Qwen2.5-0.5B-Instruct`.

## Results

Table below is from earlier runs that trained on all windows with no held-out test set; it will be replaced by results/final.

Distributed loss after each round of the 25-round runs (full logs in `results/`):

| Round | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
|------:|----------------------:|----------------------:|
| 1     | 1.5758                | 1.6784                |
| 10    | 0.6172                | 0.6147                |
| 25    | 0.5992                | 0.5910                |
| Runtime | 1299.02s           | 2047.74s              |

The reported loss comes from each client's `evaluate`, which reuses the first 10 of that client's training examples. It is not measured on a held-out test set.

## Note

The trained LoRA weights are not saved yet. The final aggregated adapter only exists in memory during the simulation.
