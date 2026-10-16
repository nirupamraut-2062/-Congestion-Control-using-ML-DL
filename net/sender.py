"""Run the project's congestion controllers over a real network path, using UDP.

Each `RealFlow` drives one controller exactly as `sim.Flow` does in the simulator: it
sends 1500-byte packets within the controller's window and pacing rate, and turns the
receiver's ACKs into the same `Ack` and `Loss` events. RTTs are measured with the
sender's own clock (the receiver echoes the send time), so the two laptops' clocks do
not need to agree. Losses are detected as TCP does: a packet counts as lost once three
later packets have been acknowledged, or when no ACK has come for a timeout.
"""
import json
import math
import random
import socket
import sys
import time
from collections import OrderedDict

import numpy as np

from sim import PKTS_PER_MBPS
from sim.network import Ack, Loss

from .receiver import ACK, CONFIG, CONFIG_OK, DATA, HEADER, PAYLOAD, PORT, STATS, STATS_REPLY

REORDER = 3        # lost once 3 later packets are acknowledged, like TCP's three duplicate ACKs
MIN_TIMEOUT = 0.2  # ... or after max(3 smoothed RTTs, 0.2 s) without an ACK
BURST_S = 0.002    # a paced flow may catch up on at most 2 ms of sending after a scheduling delay
SLEEP_S = 0.0002   # pause between rounds of the send loop
RECORD_S = 0.1     # history interval, as in the simulator


def _socket(receiver: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    sock.connect((receiver, port))
    return sock


class RealFlow:
    def __init__(self, controller, receiver: str, flow_id: int, start: float = 0.0,
                 stop: float = math.inf, name: str | None = None, port: int = PORT):
        self.cc = controller
        self.id = flow_id
        self.start = start
        self.stop = stop
        self.name = name or getattr(controller, "name", type(controller).__name__)
        self.sock = _socket(receiver, port)
        self.sock.setblocking(False)

        self.sent = self.acked = self.lost = 0
        self.lost_random = 0  # unknown at the sender; the receiver's counts give the true causes
        self.rtt_sum = 0.0
        self.inflight = 0
        self.srtt = 0.0
        self.min_rtt = math.inf

        self._seq = 0
        self._tokens = 0.0
        self._last_send = None
        self._last_ack = None
        self._unacked = OrderedDict()  # seq -> (send time, acked count at send, last ACK time at send)
        self._padding = bytes(PAYLOAD - HEADER.size)
        self._on_tick = getattr(controller, "on_tick", None)
        self._rec = (0, 0, 0, 0, 0.0)
        self.history = {k: [] for k in ("t", "cwnd", "sent", "lost", "lost_random", "throughput_mbps", "rtt_ms")}

    # ------------------------------------------------------------------ feedback

    def receive(self, clock) -> None:
        while True:
            try:
                data = self.sock.recv(2048)
            except (BlockingIOError, InterruptedError):
                return
            except ConnectionResetError:  # Windows reports an earlier ICMP "port unreachable" this way
                continue
            if len(data) < HEADER.size:
                continue
            kind, flow, seq, send_time = HEADER.unpack_from(data)
            if kind != ACK or flow != self.id:
                continue
            info = self._unacked.pop(seq, None)
            if info is None:
                continue  # already declared lost
            now = clock()
            self._on_ack(now, info)
            while self._unacked:
                oldest = next(iter(self._unacked))
                if oldest >= seq - REORDER:
                    break
                self._lose(now, oldest)

    def _on_ack(self, now: float, info) -> None:
        send_time, acked_at_send, last_ack_at_send = info
        self.inflight -= 1
        self.acked += 1
        rtt = now - send_time
        self.rtt_sum += rtt
        self.srtt = rtt if not self.srtt else 0.875 * self.srtt + 0.125 * rtt
        self.min_rtt = min(self.min_rtt, rtt)
        rate = (self.acked - acked_at_send) / max(now - last_ack_at_send, 1e-6)
        self._last_ack = now
        self.cc.on_ack(Ack(now, rtt, send_time, rate, self.inflight))

    def _lose(self, now: float, seq: int) -> None:
        send_time = self._unacked.pop(seq)[0]
        self.inflight -= 1
        self.lost += 1
        self.cc.on_loss(Loss(now, send_time, "unknown"))

    def check_timeouts(self, now: float) -> None:
        timeout = max(3 * self.srtt, MIN_TIMEOUT)
        while self._unacked:
            seq, info = next(iter(self._unacked.items()))
            if now - info[0] < timeout:
                break
            self._lose(now, seq)

    # ------------------------------------------------------------------ sending

    def tick(self, now: float) -> None:
        if self._on_tick is not None:
            self._on_tick(now)

    def send(self, now: float) -> None:
        last, self._last_send = self._last_send, now
        if not self.start <= now < self.stop:
            return
        cc = self.cc
        budget = int(cc.cwnd) - self.inflight
        paced = cc.pacing_rate is not None
        if paced:
            elapsed = now - last if last is not None else 0.0
            self._tokens = min(self._tokens + cc.pacing_rate * elapsed, cc.pacing_rate * BURST_S + 1.0)
            budget = min(budget, int(self._tokens))
        for _ in range(budget):
            seq = self._seq
            try:
                self.sock.send(HEADER.pack(DATA, self.id, seq, now) + self._padding)
            except (BlockingIOError, InterruptedError, ConnectionResetError):
                break
            self._seq += 1
            self.sent += 1
            self.inflight += 1
            if paced:
                self._tokens -= 1
            self._unacked[seq] = (now, self.acked, now if self._last_ack is None else self._last_ack)

    # ------------------------------------------------------------------ history

    def record(self, now: float, span: float) -> None:
        sent0, acked0, lost0, random0, rtt0 = self._rec
        acked = self.acked - acked0
        h = self.history
        h["t"].append(now)
        h["cwnd"].append(min(self.cc.cwnd, 1e6))
        h["sent"].append(self.sent - sent0)
        h["lost"].append(self.lost - lost0)
        h["lost_random"].append(0)
        h["throughput_mbps"].append(acked / span / PKTS_PER_MBPS)
        h["rtt_ms"].append(1000 * (self.rtt_sum - rtt0) / acked if acked else float("nan"))
        self._rec = (self.sent, self.acked, self.lost, self.lost_random, self.rtt_sum)

    def close(self) -> None:
        self.sock.close()


def fine_timers():
    """On Windows, ask for 1 ms timer resolution: by default a waiting program may wake up
    to 15.6 ms late, which would show up as RTT jitter. Other systems need nothing."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)


def run_flows(flows: list[RealFlow], duration: float) -> None:
    """Run the flows in real time for `duration` seconds, then let the last ACKs arrive."""
    fine_timers()
    t0 = time.perf_counter()

    def clock():
        return time.perf_counter() - t0

    next_record, last_record = RECORD_S, 0.0
    while True:
        now = clock()
        if now >= duration:
            break
        for f in flows:
            f.receive(clock)
        now = clock()
        for f in flows:
            f.check_timeouts(now)
            f.tick(now)
            f.send(now)
        if now >= next_record:
            for f in flows:
                f.record(now, now - last_record)
            last_record = now
            next_record = max(next_record + RECORD_S, now + RECORD_S / 2)
        time.sleep(SLEEP_S)
    end = time.perf_counter() + 0.5
    while time.perf_counter() < end:
        for f in flows:
            f.receive(clock)
        time.sleep(0.01)
    for f in flows:
        f.close()


# ---------------------------------------------------------------------- talking to the receiver

def _request(receiver: str, kind: int, reply_kind: int, payload: dict | None = None,
             port: int = PORT, tries: int = 10, timeout: float = 0.5) -> bytes:
    nonce = random.getrandbits(31)
    message = HEADER.pack(kind, 0, nonce, 0.0) + (json.dumps(payload).encode() if payload is not None else b"")
    with _socket(receiver, port) as sock:
        sock.settimeout(timeout)
        for _ in range(tries):
            try:
                sock.send(message)
                while True:
                    data = sock.recv(65536)
                    got_kind, _, seq, _ = HEADER.unpack_from(data)
                    if got_kind == reply_kind and seq == nonce:
                        return data[HEADER.size:]
            except (socket.timeout, ConnectionResetError):
                continue
    raise ConnectionError(
        f"No reply from the receiver at {receiver}:{port}. Check that receiver.py is running on the other "
        "laptop, that both laptops are on the same network, and that the Mac's firewall allows Python.")


def configure(receiver: str, rate_mbps: float = 0.0, queue: int = 50, delay_ms: float = 0.0,
              loss: float = 0.0, seed: int = 0, port: int = PORT) -> None:
    """Set the receiver's emulated bottleneck and reset its counters for a new run."""
    _request(receiver, CONFIG, CONFIG_OK, {"rate_mbps": rate_mbps, "queue": queue, "delay_ms": delay_ms,
                                           "loss": loss, "seed": seed}, port)


def receiver_stats(receiver: str, port: int = PORT) -> dict:
    """Per flow id: [packets received, dropped by the full queue, dropped at random]."""
    return {int(k): v for k, v in json.loads(_request(receiver, STATS, STATS_REPLY, port=port).decode()).items()}


def measure_rtt(receiver: str, count: int = 50, gap: float = 0.02, port: int = PORT) -> list[float]:
    """Send `count` single full-size packets `gap` seconds apart and return their RTTs in seconds."""
    rtts = []
    with _socket(receiver, port) as sock:
        sock.setblocking(False)
        padding = bytes(PAYLOAD - HEADER.size)
        t0 = time.perf_counter()
        for seq in range(count + 1):
            if seq < count:
                sock.send(HEADER.pack(DATA, 0, seq, time.perf_counter() - t0) + padding)
            wait_until = time.perf_counter() + (gap if seq < count else 1.0)  # finally wait for stragglers
            while time.perf_counter() < wait_until:
                try:
                    data = sock.recv(2048)
                except (BlockingIOError, InterruptedError, ConnectionResetError):
                    time.sleep(0.0005)
                    continue
                kind, _, _, sent = HEADER.unpack_from(data)
                if kind == ACK:
                    rtts.append(time.perf_counter() - t0 - sent)
    return rtts


def record_path(receiver: str, seconds: float = 120.0, rate_mbps: float = 5.0, port: int = PORT):
    """Send full-size packets at a steady rate and return (send times, RTTs) in seconds; NaN = lost.

    The rate is kept below what the path carries so that the RTT shows the path's own delay
    variation (Wi-Fi channel access, retransmissions, power saving), not a queue of our own.
    """
    fine_timers()
    gap = 1 / (rate_mbps * PKTS_PER_MBPS)
    padding = bytes(PAYLOAD - HEADER.size)
    sent, rtt = [], {}
    with _socket(receiver, port) as sock:
        sock.setblocking(False)
        t0 = time.perf_counter()
        next_send = 0.0
        while True:
            now = time.perf_counter() - t0
            if now >= seconds + 1.0:
                break
            if next_send < seconds and now >= next_send:
                if now - next_send > 0.01:
                    next_send = now  # fell behind (scheduling delay): skip ahead, do not burst
                try:
                    sock.send(HEADER.pack(DATA, 0, len(sent), now) + padding)
                    sent.append(now)
                except (BlockingIOError, InterruptedError, ConnectionResetError):
                    pass
                next_send += gap
            while True:
                try:
                    data = sock.recv(2048)
                except (BlockingIOError, InterruptedError, ConnectionResetError):
                    break
                kind, _, seq, send_time = HEADER.unpack_from(data)
                if kind == ACK:
                    rtt[seq] = time.perf_counter() - t0 - send_time
            time.sleep(0.0002)
    return np.array(sent), np.array([rtt.get(i, np.nan) for i in range(len(sent))])
