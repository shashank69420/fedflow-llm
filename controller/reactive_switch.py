from pox.core import core
import pox.openflow.libopenflow_01 as of

log = core.getLogger()

# Hardcoded forwarding tables, one per switch (by dpid).
# Format: { dpid: { destination_host_ip: out_port } }
# Default routing uses the primary path: h1 -- s1 -- s3 -- s2 -- h2
# (the 10 Mbps link pair), matching topo.py's link order.
#
# Port numbers below match topo.py's addLink() call order per switch:
#   s1: eth1->h1, eth2->s3, eth3->s4
#   s2: eth1->h2, eth2->s3, eth3->s4
#   s3: eth1->s1, eth2->s2
#   s4: eth1->s1, eth2->s2
# Confirm these against your own `net` output in Step 3.2 before Part 3 —
# if they differ, update the tables below to match your real port numbers.

FORWARDING_TABLE = {
    1: {"10.0.0.1": 1, "10.0.0.2": 2},  # s1: h1 is local (port 1), h2 reached via s3 (port 2)
    2: {"10.0.0.1": 2, "10.0.0.2": 1},  # s2: h2 is local (port 1), h1 reached via s3 (port 2)
    3: {"10.0.0.1": 1, "10.0.0.2": 2},  # s3: h1 direction is port 1 (toward s1), h2 direction is port 2 (toward s2)
    4: {"10.0.0.1": 1, "10.0.0.2": 2},  # s4: same shape as s3, used only if rerouted to alt path in Part 3
}

class DeterministicSwitch:
    def __init__(self, connection):
        self.connection = connection
        self.dpid = connection.dpid
        connection.addListeners(self)
        log.info(f"Deterministic forwarding active on switch dpid={self.dpid}")

    def _handle_PacketIn(self, event):
        packet = event.parsed
        if not packet.parsed:
            return

        # Only handle IPv4 packets we have a table entry for (covers ping/iperf).
        ip_pkt = packet.find('ipv4')
        if ip_pkt is None:
            return  # drop ARP/other silently rather than flood -- see note below

        dst_ip = str(ip_pkt.dstip)
        table = FORWARDING_TABLE.get(self.dpid, {})
        out_port = table.get(dst_ip)

        if out_port is None:
            return  # no known route on this switch for this destination -- drop, never flood

        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match.from_packet(packet, event.port)
        msg.idle_timeout = 10
        msg.hard_timeout = 30
        msg.actions.append(of.ofp_action_output(port=out_port))
        msg.data = event.ofp
        self.connection.send(msg)

def launch():
    def start_switch(event):
        DeterministicSwitch(event.connection)
    core.openflow.addListenerByName("ConnectionUp", start_switch)
