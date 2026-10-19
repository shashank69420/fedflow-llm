from pox.core import core
import pox.openflow.libopenflow_01 as of
import os

log = core.getLogger()

CONGESTION_THRESHOLD = 100.0
FORECAST_FILE = os.path.expanduser("~/fedflow/controller/forecast_value.txt")

# Primary path table (matches Part 2's reactive_switch.py exactly).
PRIMARY_TABLE = {
    1: {"10.0.0.1": 1, "10.0.0.2": 2},
    2: {"10.0.0.1": 2, "10.0.0.2": 1},
    3: {"10.0.0.1": 1, "10.0.0.2": 2},
    4: {"10.0.0.1": 1, "10.0.0.2": 2},
}

# Alternate path table: only s1 and s2 change (they now point at s4 instead of s3).
# Port numbers: confirm against your own `net` output in Step 3.2 -- these assume
# topo.py's addLink() order (s1: eth1=h1, eth2=s3, eth3=s4).
ALT_TABLE = {
    1: {"10.0.0.1": 1, "10.0.0.2": 3},  # s1 now sends h2-bound traffic out port 3 (toward s4)
    2: {"10.0.0.1": 3, "10.0.0.2": 1},  # s2 now sends h1-bound traffic out port 3 (toward s4)
    3: {"10.0.0.1": 1, "10.0.0.2": 2},  # s3 unused on this path, table irrelevant but harmless
    4: {"10.0.0.1": 1, "10.0.0.2": 2},  # s4 now carries the traffic
}

class RerouteSwitch:
    def __init__(self, connection):
        self.connection = connection
        self.dpid = connection.dpid
        connection.addListeners(self)
        log.info(f"Reroute-aware forwarding active on switch dpid={self.dpid}")

    def _handle_PacketIn(self, event):
        packet = event.parsed
        if not packet.parsed:
            return

        ip_pkt = packet.find('ipv4')
        if ip_pkt is None:
            return  # ARP handled via static entries, not the controller -- see Step 2.6a

        dst_ip = str(ip_pkt.dstip)
        forecast = self.read_forecast()
        use_alt_path = forecast > CONGESTION_THRESHOLD

        table = ALT_TABLE if use_alt_path else PRIMARY_TABLE
        out_port = table.get(self.dpid, {}).get(dst_ip)

        if out_port is None:
            return  # no route for this destination on this switch -- drop, never flood

        if use_alt_path:
            log.info(f"[FedFlow] Forecast {forecast} > {CONGESTION_THRESHOLD} on switch {self.dpid} — using alt path, port {out_port}")

        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match.from_packet(packet, event.port)
        msg.idle_timeout = 10
        msg.hard_timeout = 30
        msg.actions.append(of.ofp_action_output(port=out_port))
        msg.data = event.ofp
        self.connection.send(msg)

    def read_forecast(self):
        try:
            with open(FORECAST_FILE, "r") as f:
                return float(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0.0

def launch():
    def start_switch(event):
        RerouteSwitch(event.connection)
    core.openflow.addListenerByName("ConnectionUp", start_switch)
