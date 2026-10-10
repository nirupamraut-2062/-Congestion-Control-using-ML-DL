"""Train and evaluate an LSTM that forecasts a flow's RTT half a second ahead.

Input: the last 3 s of the flow's own measurements (RTT, throughput, loss, window),
sampled every 100 ms. Output: the RTT 0.5 s from now. Compared against three
non-learned forecasts on scenarios the model never saw.

Usage: python -m ml.rtt_forecaster
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from experiments.style import INK2, MUTED, apply_style

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
MODEL_PATH = MODELS / "rtt_lstm.pt"

STEP_S = 0.1
WINDOW = 30
HORIZON = 5
INPUTS = ["queue_delay_ratio", "rtt_peak_ratio", "throughput_ratio", "loss_fraction", "since_loss", "cwnd_ratio"]
SINCE_LOSS_CAP = 50  # steps
VIOLET = "#4a3aa7"


class RttForecaster(nn.Module):
    """Predicts the change in RTT (in units of minimum RTT) over the forecast horizon."""

    def __init__(self, n_inputs: int = len(INPUTS), hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def flow_inputs(frame: pd.DataFrame):
    """Per-step model inputs for one flow, using only information available at that step."""
    rtt = frame["rtt_ms"].ffill().to_numpy()
    seen = ~np.isnan(rtt)
    frame, rtt = frame[seen], rtt[seen]
    min_rtt = np.minimum.accumulate(rtt)
    throughput = frame["throughput_mbps"].to_numpy()
    cwnd = frame["cwnd"].to_numpy()
    lost = frame["lost"].to_numpy()
    steps = np.arange(len(rtt))
    last_loss = np.maximum.accumulate(np.where(lost > 0, steps, -SINCE_LOSS_CAP))
    inputs = np.stack([
        rtt / min_rtt - 1.0,
        rtt / np.maximum.accumulate(rtt),
        throughput / np.maximum(np.maximum.accumulate(throughput), 1e-6),
        lost / np.maximum(frame["sent"].to_numpy(), 1),
        np.minimum(steps - last_loss, SINCE_LOSS_CAP) / SINCE_LOSS_CAP,
        cwnd / np.maximum.accumulate(cwnd),
    ], axis=1).astype(np.float32)
    return frame["t"].to_numpy(), rtt, min_rtt, inputs


def make_windows(series: pd.DataFrame) -> dict:
    """Slice every flow into (history window, future RTT) examples."""
    parts = {k: [] for k in ("x", "history_ms", "min_rtt", "target_ms", "t", "run_id", "flow_id", "controller")}
    for (run_id, flow_id), frame in series.groupby(["run_id", "flow_id"], sort=True):
        t, rtt, min_rtt, inputs = flow_inputs(frame)
        n = len(rtt) - WINDOW - HORIZON + 1
        if n <= 0:
            continue
        last = np.arange(n) + WINDOW - 1
        parts["x"].append(sliding_window_view(inputs, WINDOW, axis=0)[:n].transpose(0, 2, 1))
        parts["history_ms"].append(sliding_window_view(rtt, WINDOW)[:n])
        parts["min_rtt"].append(min_rtt[last])
        parts["target_ms"].append(rtt[last + HORIZON])
        parts["t"].append(t[last + HORIZON])
        parts["run_id"].append(np.full(n, run_id))
        parts["flow_id"].append(np.full(n, flow_id))
        parts["controller"].append(np.full(n, frame["controller"].iloc[0]))
    return {k: np.concatenate(v) for k, v in parts.items()}


def subset(windows: dict, mask) -> dict:
    return {k: v[mask] for k, v in windows.items()}


def classical_forecasts(history_ms: np.ndarray) -> dict:
    recent = history_ms[:, -5:]
    steps = np.arange(5) - 2.0
    slope = (recent - recent.mean(axis=1, keepdims=True)) @ steps / (steps ** 2).sum()
    return {
        "Last value": recent[:, -1],
        "Moving average (0.5 s)": recent.mean(axis=1),
        "Linear extrapolation": recent.mean(axis=1) + slope * (2 + HORIZON),
    }


def predict(model: nn.Module, windows: dict) -> np.ndarray:
    """LSTM forecast in milliseconds."""
    model.eval()
    with torch.no_grad():
        change = torch.cat([model(batch) for batch in torch.from_numpy(windows["x"]).split(4096)]).numpy()
    return windows["history_ms"][:, -1] + change * windows["min_rtt"]


def train(model: nn.Module, train_set: dict, val_set: dict, epochs: int = 25, patience: int = 4) -> None:
    def tensors(w):
        change = (w["target_ms"] - w["history_ms"][:, -1]) / w["min_rtt"]
        return torch.from_numpy(w["x"]), torch.from_numpy(change.astype(np.float32))

    x, y = tensors(train_set)
    x_val, y_val = tensors(val_set)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(x)).split(512):
            optimizer.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = sum(loss_fn(model(xb), yb).item() * len(xb)
                      for xb, yb in zip(x_val.split(4096), y_val.split(4096))) / len(x_val)
        print(f"epoch {epoch:2d}  validation loss {val:.5f}")
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)


def errors(forecast_ms, w: dict) -> dict:
    err = forecast_ms - w["target_ms"]
    return {
        "mae_ms": float(np.abs(err).mean()),
        "rmse_ms": float(np.sqrt((err ** 2).mean())),
        "mae_pct_of_min_rtt": float(100 * np.abs(err / w["min_rtt"]).mean()),
    }


def plot_models(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    order = metrics.iloc[::-1]
    ax.barh(order["method"], order["mae_ms"], height=0.55, color=VIOLET)
    for name, value in zip(order["method"], order["mae_ms"]):
        ax.annotate(f"{value:.2f} ms", (value, name), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, order["mae_ms"].max() * 1.18)
    ax.set_xlabel(f"Mean absolute error forecasting RTT {HORIZON * STEP_S:g} s ahead (lower is better)")
    ax.set_title("RTT forecaster: LSTM against non-learned forecasts")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "rtt_forecast_models.png")
    plt.close(fig)


def plot_example(test: dict, lstm_ms: np.ndarray) -> None:
    """One held-out loss-based flow, picked for having the most RTT movement to forecast."""
    table = pd.DataFrame({"run_id": test["run_id"], "flow_id": test["flow_id"],
                          "controller": test["controller"], "swing": test["target_ms"] / test["min_rtt"]})
    table = table[table["controller"] != "BBR"]
    run_id, flow_id = table.groupby(["run_id", "flow_id"])["swing"].std().idxmax()
    pick = (test["run_id"] == run_id) & (test["flow_id"] == flow_id)
    t, actual = test["t"][pick], test["target_ms"][pick]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, actual, color=INK2, label="Measured RTT")
    ax.plot(t, test["history_ms"][pick][:, -1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            label="Last-value forecast")
    ax.plot(t, lstm_ms[pick], color=VIOLET, label="LSTM forecast")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Round-trip time (ms)")
    ax.set_title(f"Forecasting RTT {HORIZON * STEP_S:g} s ahead on an unseen {test['controller'][pick][0]} flow")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / "rtt_forecast_example.png")
    plt.close(fig)


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    apply_style()
    torch.manual_seed(0)

    windows = make_windows(pd.read_csv(DATA / "timeseries.csv"))
    # Whole scenarios are held out: 70% train, 15% validation, 15% test.
    run_ids = np.random.default_rng(0).permutation(np.unique(windows["run_id"]))
    cut1, cut2 = int(0.7 * len(run_ids)), int(0.85 * len(run_ids))
    train_set = subset(windows, np.isin(windows["run_id"], run_ids[:cut1]))
    val_set = subset(windows, np.isin(windows["run_id"], run_ids[cut1:cut2]))
    test = subset(windows, np.isin(windows["run_id"], run_ids[cut2:]))
    print(f"windows: {len(train_set['x'])} train, {len(val_set['x'])} validation, {len(test['x'])} test")

    model = RttForecaster()
    train(model, train_set, val_set)

    forecasts = {**classical_forecasts(test["history_ms"]), "LSTM": predict(model, test)}
    metrics = pd.DataFrame([{"method": name, **errors(f, test)} for name, f in forecasts.items()])
    metrics.round(4).to_csv(RESULTS / "rtt_forecast_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    rows = []
    for controller in np.unique(test["controller"]):
        mask = test["controller"] == controller
        part = subset(test, mask)
        rows.append({"controller": controller, "windows": int(mask.sum()),
                     **{f"{name} mae_ms": errors(f[mask], part)["mae_ms"] for name, f in forecasts.items()}})
    by_controller = pd.DataFrame(rows)
    by_controller.round(4).to_csv(RESULTS / "rtt_forecast_by_controller.csv", index=False)
    print("\nMean absolute error (ms) by controller:")
    print(by_controller.round(3).to_string(index=False))

    plot_models(metrics)
    plot_example(test, forecasts["LSTM"])

    torch.save({"state_dict": model.state_dict(), "inputs": INPUTS, "window": WINDOW,
                "horizon": HORIZON, "step_s": STEP_S}, MODEL_PATH)
    print(f"\nSaved LSTM to {MODEL_PATH}")


if __name__ == "__main__":
    main()
