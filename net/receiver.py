"""Receiver for the real-network test. Copy this one file to the second laptop and run:

    python3 receiver.py              (listens on UDP port 9000)

It needs Python 3.8 or newer and nothing else. It answers every data packet with a small
ACK. It can also put an emulated bottleneck in front of itself, like the simulator's link:
a drop-tail queue drained at a fixed rate, extra delay on the return path, and random
loss. The sender sets these before each run, so the receiver only has to be started once.
Stop it with Ctrl+C.
"""
import argparse
import heapq
import json
import random
import select
import socket
import struct
import sys
import time
from collections import deque

PORT = 9000
PACKET_BYTES = 1500
PAYLOAD = 1472  # UDP data per packet: 1472 + 8 (UDP header) + 20 (IPv4 header) = 1500 bytes on the wire
HEADER = struct.Struct("!BIId")  # message kind, flow id, sequence number, sender's send time
DATA, ACK, CONFIG, CONFIG_OK, STATS, STATS_REPLY = range(6)


class Bottleneck:
    """Drop-tail FIFO queue served at `rate_mbps`, then `delay_ms` before the ACK leaves.

    rate_mbps = 0 means no emulated bottleneck: the Wi-Fi path itself is the only limit.
    """

    def __init__(self, rate_mbps=0.0, queue=50, delay_ms=0.0, loss=0.0, seed=0):
        self.pps = rate_mbps * 1e6 / (8 * PACKET_BYTES)
        self.queue = int(queue)
        self.delay = delay_ms / 1000
        self.loss = loss
        self.rng = random.Random(seed)
        self.departures = deque()  # departure times of the packets still in the queue
        self.busy_until = 0.0
        self.stats = {}  # flow id -> [received, dropped by the full queue, dropped at random]

    def arrive(self, now, flow):
        """Return when to send the ACK for a packet arriving now, or None if it is dropped."""
        counts = self.stats.setdefault(flow, [0, 0, 0])
        if self.loss and self.rng.random() < self.loss:
            counts[2] += 1
            return None
        depart = now
        if self.pps:
            while self.departures and self.departures[0] <= now:
                self.departures.popleft()
            if len(self.departures) >= self.queue:
                counts[1] += 1
                return None
            depart = max(now, self.busy_until) + 1 / self.pps
            self.busy_until = depart
            self.departures.append(depart)
        counts[0] += 1
        return depart + self.delay


def local_addresses():
    """Best guess at this machine's LAN address, to tell the user what to type on the sender."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
        return probe.getsockname()[0]
    except OSError:
        return "unknown"
    finally:
        probe.close()


def fine_timers():
    """On Windows, ask for 1 ms timer resolution: by default a waiting program may wake up
    to 15.6 ms late, which would show up as RTT jitter. Other systems need nothing."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    fine_timers()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    sock.bind(("", args.port))
    sock.setblocking(False)
    print(f"Receiver listening on UDP port {args.port}. This laptop's address: {local_addresses()}")
    print("On the sender run:  python -m net.run_network_test --receiver <that address>")

    neck = Bottleneck()
    pending = []  # heap of (time to send, counter, address, ACK bytes)
    counter = 0
    while True:
        now = time.monotonic()
        timeout = min(max(pending[0][0] - now, 0.0), 0.05) if pending else 0.05
        # select() can wake up a few ms late (on Windows by up to a timer tick), so it only waits
        # until 2 ms before the next ACK is due; the last stretch is covered by short sleeps.
        readable, _, _ = select.select([sock], [], [], max(timeout - 0.002, 0.0))
        if not readable and timeout < 0.002:
            time.sleep(0.0002)
        now = time.monotonic()
        while readable:
            try:
                data, addr = sock.recvfrom(2048)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:  # Windows reports an earlier ICMP error this way
                continue
            if len(data) < HEADER.size:
                continue
            kind, flow, seq, sent = HEADER.unpack_from(data)
            if kind == DATA:
                at = neck.arrive(now, flow)
                if at is not None:
                    heapq.heappush(pending, (at, counter, addr, HEADER.pack(ACK, flow, seq, sent)))
                    counter += 1
            elif kind == CONFIG:
                config = json.loads(data[HEADER.size:].decode())
                neck = Bottleneck(**config)
                pending.clear()
                sock.sendto(HEADER.pack(CONFIG_OK, 0, seq, 0.0), addr)
                print(f"{time.strftime('%H:%M:%S')}  new run from {addr[0]}: {config}")
            elif kind == STATS:
                reply = json.dumps({str(k): v for k, v in neck.stats.items()}).encode()
                sock.sendto(HEADER.pack(STATS_REPLY, 0, seq, 0.0) + reply, addr)
        while pending and pending[0][0] <= now:
            _, _, addr, message = heapq.heappop(pending)
            try:
                sock.sendto(message, addr)
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
