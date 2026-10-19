from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.recoco import Timer
from datetime import datetime
import os

log = core.getLogger()

THRESHOLD = 100.0  # 100 traffic units = the 10 Mbps primary link capacity
FORECAST_FILE = os.path.expanduser("~/fedflow/controller/forecast_value.txt")
CHECK_INTERVAL = 1.0  # seconds between forecast-file checks

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

# The one routing decision, shared by the timer and every switch: "primary" or "alternate".
current_path = "primary"
last_forecast = None   # last forecast value read successfully (None until the first good read)
_warned = False        # True after a bad-file warning, so it is logged once, not every second


def decide_path(value, threshold):
    """Pure decision rule, no POX needed: strictly above the threshold -> alternate."""
    return "alternate" if value > threshold else "primary"


def read_forecast():
    """Return the forecast as a float, or None if the file is missing or not a number.
    Warns once per run of bad reads; a good read re-arms the warning."""
    global _warned
    try:
        with open(FORECAST_FILE, "r") as f:
            value = float(f.read().strip())
    except (OSError, ValueError) as error:
        if not _warned:
            log.warning(f"[FedFlow] Can't read a number from {FORECAST_FILE} ({error}); "
                        f"keeping current path '{current_path}'")
            _warned = True
        return None
    _warned = False
    return value


def clear_all_flows():
    """Delete every flow on every connected switch, so the next packets go to the
    controller and get flows from the newly selected table."""
    # core.openflow.connections is a dict of dpid -> Connection; copy it in case a
    # switch connects/disconnects while we loop.
    for connection in list(core.openflow.connections.values()):
        connection.send(of.ofp_flow_mod(command=of.OFPFC_DELETE))  # empty match = all flows


def check_forecast():
    """Runs every CHECK_INTERVAL seconds: re-read the forecast and flip paths if needed."""
    global current_path, last_forecast
    try:
        value = read_forecast()
        if value is None:
            return  # bad or missing file: keep the last decision
        last_forecast = value
        new_path = decide_path(value, THRESHOLD)
        if new_path != current_path:
            old_path, current_path = current_path, new_path
            clear_all_flows()
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # milliseconds
            log.info(f"[FedFlow] {stamp} FLIP {old_path} -> {new_path} (forecast {value})")
    except Exception as error:
        # Never let one bad tick raise out of the timer callback.
        log.error(f"[FedFlow] forecast check failed ({error!r}); keeping current path '{current_path}'")


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
        # Use the stored decision; the file is only read by the timer.
        use_alt_path = current_path == "alternate"

        table = ALT_TABLE if use_alt_path else PRIMARY_TABLE
        out_port = table.get(self.dpid, {}).get(dst_ip)

        if out_port is None:
            return  # no route for this destination on this switch -- drop, never flood

        if use_alt_path:
            log.info(f"[FedFlow] Forecast {last_forecast} > {THRESHOLD} on switch {self.dpid} — using alt path, port {out_port}")

        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match.from_packet(packet, event.port)
        msg.idle_timeout = 10
        msg.hard_timeout = 30
        msg.actions.append(of.ofp_action_output(port=out_port))
        msg.data = event.ofp
        self.connection.send(msg)


def launch():
    global current_path, last_forecast
    # Starting decision matches the file; if it can't be read, start on the primary path.
    value = read_forecast()
    if value is not None:
        last_forecast = value
        current_path = decide_path(value, THRESHOLD)
    log.info(f"[FedFlow] Starting on {current_path} path (forecast {last_forecast}, threshold {THRESHOLD})")

    def start_switch(event):
        RerouteSwitch(event.connection)
    core.openflow.addListenerByName("ConnectionUp", start_switch)

    Timer(CHECK_INTERVAL, check_forecast, recurring=True)
