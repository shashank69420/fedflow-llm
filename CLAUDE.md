# FedFlow-LLM
Computer Networks course project. Federated LoRA fine-tuning (Flower, FedAvg) of Qwen2.5-1.5B forecasts network traffic. The forecast triggers a reroute in a Mininet + POX network. Goal: show proactive (forecast-driven) routing vs reactive routing on delay and packet loss.

Layout: training/ (client.py, server.py, generate_data.py), controller/ (POX: topo.py, reactive_switch.py, reroute_trigger.py, write_forecast.py), results/, notebooks/.
Topology: h1-s1, h2-s2. Primary path s1-s3-s2 (10 Mbps). Alternate path s1-s4-s2 (50 Mbps). The controller uses fixed forwarding tables and never floods (flooding causes a broadcast storm in this looped topology). Hosts use static ARP. POX only loads files from ~/pox/ext.
Training runs on Colab (T4 GPU), never locally. Colab install order matters, see README.

RULES:
1. Never invent or estimate numbers. Every number in README, results or reports must come from a saved raw output file. If something was not run, write "not run".
2. Never run anything with sudo. Tell me the exact command and I will run it.
3. Before editing files, show me a short plan and wait for my OK.
4. Put plain-English comments on every non-obvious line of code you write.
5. You may commit with clear messages. Never git push.
6. When a task is done, list every file you changed.
