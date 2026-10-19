# FedFlow-LLM: Colab steps

Use a GPU runtime (T4). Run the cells below in order. Step (e) is a smoke test and **must succeed before you start step (f)**. If (e) fails, fix the problem first; don't start the full run.

**(a)** Keeps the browser tab active so Colab doesn't disconnect while it's idle. This is JavaScript, not a notebook cell: paste it into the browser's DevTools console (F12 → Console) on the Colab tab.

```javascript
function keepAlive() {
  // Click the "Connect"/status button every 60 s so Colab sees activity.
  const btn = document.querySelector("colab-connect-button");
  if (btn) btn.shadowRoot.querySelector("#connect").click();
}
setInterval(keepAlive, 60000);
```

**(b)** Installs dependencies. The order matters (see README).

```python
!pip install -q torch transformers peft accelerate bitsandbytes "flwr[simulation]"
!pip uninstall -y torch
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
!pip uninstall -y torchvision torchaudio
!pip install -q transformers==4.46.3 peft==0.13.2
```

**(c)** Clones the repo and moves into `training/`, where every script runs.

```python
!git clone https://github.com/shashank69420/fedflow-llm.git
%cd fedflow-llm/training
```

**(d)** Generates the train/test data and checks that no training window overlaps the test block.

```python
!python generate_data.py && python check_split.py
```

**(e)** SMOKE TEST: 1 round of training, then inference on 5 test windows. It must finish without errors and write `outputs/adapter.npz`, `outputs/loss_history.csv`, `outputs/forecasts.csv` and `outputs/metrics.json`. Its numbers are not results.

```python
!FEDFLOW_ROUNDS=1 python server.py && python infer.py --eval --limit 5
```

**(f)** FULL RUN: 25 rounds of training, then inference on all held-out test windows. Run this only after (e) succeeded.

```python
!FEDFLOW_ROUNDS=25 python server.py && python infer.py --eval
```

**(g)** Zips `outputs/` and downloads it.

```python
!zip -r outputs.zip outputs/
from google.colab import files
files.download("outputs.zip")
```
