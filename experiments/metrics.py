import numpy as np


def jain_index(values) -> float:
    """Jain's fairness index: 1.0 when all flows get equal throughput, 1/n when one takes all."""
    x = np.asarray(values, dtype=float)
    if not x.any():
        return float("nan")
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))


def flow_summary(flow, t0: float = 0.0, t1: float = float("inf")) -> dict:
    """Per-flow metrics over the recorded intervals ending in (t0, t1]."""
    h = flow.history
    t = np.asarray(h["t"])
    mask = (t > max(t0, flow.start)) & (t <= min(t1, flow.stop))
    throughput = np.asarray(h["throughput_mbps"])[mask]
    rtt = np.asarray(h["rtt_ms"])[mask]
    sent = np.asarray(h["sent"])[mask].sum()
    lost = np.asarray(h["lost"])[mask].sum()
    lost_random = np.asarray(h["lost_random"])[mask].sum()
    seen = ~np.isnan(rtt) & (throughput > 0)
    return {
        "throughput_mbps": float(throughput.mean()) if mask.any() else 0.0,
        "avg_rtt_ms": float(np.average(rtt[seen], weights=throughput[seen])) if seen.any() else float("nan"),
        "p95_rtt_ms": float(np.percentile(rtt[seen], 95)) if seen.any() else float("nan"),
        "loss_pct": float(100 * lost / sent) if sent else 0.0,
        "congestion_loss_pct": float(100 * (lost - lost_random) / sent) if sent else 0.0,
    }


def link_summary(sim, flow, t0: float = 0.0) -> dict:
    """Flow metrics relative to what the link offered: utilisation and delay inflation."""
    stats = flow_summary(flow, t0)
    t = np.asarray(sim.history["t"])
    capacity = np.asarray(sim.history["capacity_mbps"])[t > max(t0, flow.start)].mean()
    base_rtt_ms = 1000 * sim.link.rtt
    return {
        "utilisation_pct": 100 * stats["throughput_mbps"] / capacity,
        "delay_ratio": stats["avg_rtt_ms"] / base_rtt_ms,
        "p95_delay_ratio": stats["p95_rtt_ms"] / base_rtt_ms,
        **stats,
    }


def objective(row) -> float:
    """The balanced training objective evaluated on whole-run metrics (1.0 is ideal)."""
    delay = min(max(row["delay_ratio"] - 1.0, 0.0), 3.0)
    return row["utilisation_pct"] / 100 - 0.5 * delay - 5.0 * row["congestion_loss_pct"] / 100
