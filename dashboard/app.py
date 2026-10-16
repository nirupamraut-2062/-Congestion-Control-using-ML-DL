"""Interactive demo: pick controllers and a network, then watch them run.

Run from the project folder with:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from experiments.controllers import ALL, EXTRA, available, make
from experiments.metrics import jain_index, link_summary
from experiments.style import AXIS, COLORS, GRID, INK, INK2, MUTED, SURFACE
from sim import Flow, Link, Simulator
from sim.traces import Schedule

RESULTS = ROOT / "results"
SIDE_BY_SIDE = "Side by side: each controller alone on its own copy of the link"
SHARED = "Competing: all controllers share one link"
CAPACITY_PATTERN = (1.0, 0.4, 1.4, 0.7, 1.2)  # multiples of the nominal bandwidth, one per 8 s
LIVE_UPDATES = 30

PRESETS = {
    "Home broadband": {"bandwidth": 20.0, "delay": 15, "queue": 100, "loss": 0.0, "varying": False},
    "Lossy Wi-Fi": {"bandwidth": 10.0, "delay": 20, "queue": 50, "loss": 1.0, "varying": False},
    "Mobile network (capacity keeps changing)": {"bandwidth": 10.0, "delay": 30, "queue": 60, "loss": 0.1,
                                                 "varying": True},
    "Satellite link": {"bandwidth": 10.0, "delay": 250, "queue": 300, "loss": 0.5, "varying": False},
}

GALLERY = [
    ("Classical baselines", [
        ("baselines_single_flow.png", "Reno and CUBIC fill the queue until packets drop; BBR keeps delay near the minimum."),
        ("baselines_random_loss.png", "Loss-based controllers collapse when the link itself drops packets."),
        ("baselines_fairness.png", "Two flows of the same algorithm sharing a link."),
    ]),
    ("Machine learning: loss classifier", [
        ("loss_classifier_models.png", "Telling congestion loss from random loss, on scenarios not seen in training."),
        ("loss_classifier_confusion.png", "Where the deployed classifier is right and wrong."),
        ("loss_classifier_importance.png", "Which sender-side signals the classifier relies on."),
    ]),
    ("Deep learning: RTT forecaster", [
        ("rtt_forecast_models.png", "LSTM against simple forecasts, predicting RTT 0.5 s ahead."),
        ("rtt_forecast_example.png", "Forecast against measurement on a flow not seen in training."),
    ]),
    ("Reinforcement learning and the full comparison", [
        ("training_curves.png", "How the PPO policies improved during training."),
        ("comparison_test_links.png", "Throughput against delay for all six controllers on unseen links."),
        ("comparison_random_loss.png", "Throughput as random loss increases."),
        ("comparison_capacity_trace.png", "Following a link whose capacity changes."),
        ("comparison_fairness.png", "Sharing a link with CUBIC and with a copy of itself."),
        ("comparison_out_of_distribution.png", "Conditions outside the training range."),
        ("comparison_variants.png", "Reward profiles and ablations of the PPO agent."),
    ]),
    ("Final improvements", [
        ("comparison_switching.png", "PPO-Switch: PPO with a competitive mode now gets a fair share beside other flows."),
        ("comparison_switching_timeline.png", "PPO-Switch detects a CUBIC flow, competes with it, and returns to low delay when it leaves."),
        ("comparison_jitter.png", "RTT jitter, as on Wi-Fi: PPO-J was retrained with jitter so it keeps working."),
    ]),
    ("Real-network test (two laptops over Wi-Fi)", [
        ("network_summary.png", "Each controller alone on real Wi-Fi, beside the simulator and the replay of the recorded Wi-Fi."),
        ("wifi_stall_pattern.png", "What the Wi-Fi does: it stalls about twice a second for roughly 75 ms."),
        ("comparison_wifi_trace.png", "The recorded Wi-Fi replayed in the simulator (held-out part); PPO-W was trained on the rest."),
        ("network_single.png", "Each controller alone over real Wi-Fi, next to the simulator's result."),
        ("network_loss.png", "Random loss on the real path."),
        ("network_fairness.png", "Two flows sharing the real path."),
    ]),
]


def build_runs(names, mode, bandwidth, delay, queue, loss_pct, varying, duration, join_gap):
    """Simulators to step through: one per controller, or a single shared one."""
    def link():
        capacity = bandwidth
        if varying:
            capacity = Schedule([(8 * i, bandwidth * CAPACITY_PATTERN[i % len(CAPACITY_PATTERN)])
                                 for i in range(int(duration // 8) + 1)])
        return Link(capacity, delay, queue, loss_pct / 100)

    if mode == SHARED:
        flows = [Flow(make(name, seed=i), start=i * join_gap, name=name) for i, name in enumerate(names)]
        return [(Simulator(link(), flows), flows)]
    runs = []
    for name in names:
        flow = Flow(make(name), name=name)
        runs.append((Simulator(link(), [flow]), [flow]))
    return runs


def figure(runs, mode, duration):
    titles = ("Throughput (Mbps)", "Round-trip time (ms)", "Packets waiting in the bottleneck queue",
              "Congestion window (packets)")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=titles)

    link_history = runs[0][0].history
    fig.add_trace(go.Scatter(x=link_history["t"], y=link_history["capacity_mbps"], name="Link capacity",
                             line={"color": MUTED, "width": 1.5, "dash": "dash", "shape": "hv"},
                             hovertemplate="%{y:.1f} Mbps"), row=1, col=1)
    for sim, flows in runs:
        for flow in flows:
            h = flow.history
            line = {"color": COLORS[flow.name], "width": 2}
            common = {"x": h["t"], "name": flow.name, "legendgroup": flow.name, "line": line}
            fig.add_trace(go.Scatter(y=h["throughput_mbps"], hovertemplate="%{y:.2f} Mbps", **common), row=1, col=1)
            fig.add_trace(go.Scatter(y=h["rtt_ms"], hovertemplate="%{y:.0f} ms", showlegend=False, **common),
                          row=2, col=1)
            # A rate-based controller has no window to plot.
            window = [w if w < 1e6 else None for w in h["cwnd"]]
            fig.add_trace(go.Scatter(y=window, hovertemplate="%{y:.0f} packets", showlegend=False, **common),
                          row=4, col=1)
        shared = mode == SHARED
        name = "Shared queue" if shared else flows[0].name
        fig.add_trace(go.Scatter(x=sim.history["t"], y=sim.history["queue"], name=name, legendgroup=name,
                                 showlegend=shared, hovertemplate="%{y} packets",
                                 line={"color": INK2 if shared else COLORS[name], "width": 2}), row=3, col=1)

    fig.update_xaxes(range=[0, duration], gridcolor=GRID, linecolor=AXIS, tickfont={"color": INK2})
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(rangemode="tozero", gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont={"color": INK2})
    fig.update_annotations(font={"size": 13, "color": INK}, xanchor="left", x=0)
    fig.update_layout(height=860, hovermode="x unified", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font={"family": "Segoe UI, system-ui, sans-serif", "color": INK},
                      margin={"l": 50, "r": 20, "t": 70, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0})
    return fig


def summary(runs, mode, warmup):
    rows = []
    for sim, flows in runs:
        for flow in flows:
            stats = link_summary(sim, flow, warmup)
            rows.append({
                "Controller": flow.name,
                "Throughput (Mbps)": stats["throughput_mbps"],
                "Share of capacity (%)": stats["utilisation_pct"],
                "Average RTT (ms)": stats["avg_rtt_ms"],
                "RTT / base RTT": stats["delay_ratio"],
                "Packets lost (%)": stats["loss_pct"],
            })
    table = pd.DataFrame(rows).round(2)
    fairness = jain_index(table["Throughput (Mbps)"]) if mode == SHARED and len(rows) > 1 else None
    return table, fairness


def show_result(result):
    st.plotly_chart(result["figure"], width="stretch", key="final")
    st.subheader("Summary")
    st.caption(f"Measured after the first {result['warmup']:g} s.")
    st.dataframe(result["table"], hide_index=True, width="stretch")
    if result["fairness"] is not None:
        st.metric("Jain fairness index (1.0 = equal shares)", f"{result['fairness']:.3f}")


st.set_page_config(page_title="Congestion Control using ML/DL", layout="wide")
st.title("Congestion Control using ML/DL")
st.caption("Classical TCP congestion control against three learned controllers, on a simulated bottleneck link.")

live_tab, results_tab = st.tabs(["Live simulation", "Project results"])

with st.sidebar:
    st.header("Network")
    preset_name = st.selectbox("Start from", list(PRESETS))
    preset = PRESETS[preset_name]
    # Widget keys include the preset so choosing another preset resets the sliders.
    bandwidth = st.slider("Bandwidth (Mbps)", 1.0, 50.0, preset["bandwidth"], 0.5, key=f"bw_{preset_name}")
    delay = st.slider("One-way delay (ms)", 5, 300, preset["delay"], 5, key=f"delay_{preset_name}",
                      help="The base round-trip time is twice this.")
    queue = st.slider("Queue size (packets)", 5, 500, preset["queue"], 5, key=f"queue_{preset_name}")
    loss_pct = st.slider("Random packet loss (%)", 0.0, 10.0, preset["loss"], 0.1, key=f"loss_{preset_name}",
                         help="Loss caused by the link itself, such as wireless noise, not by congestion.")
    varying = st.checkbox("Capacity changes every 8 s", preset["varying"], key=f"vary_{preset_name}")
    duration = st.slider("Duration (s)", 10, 60, 30, 5)

    st.header("Controllers")
    names = st.multiselect("Controllers to run", available(ALL + EXTRA), default=["CUBIC", "BBR", "PPO"], max_selections=4)
    mode = st.radio("How they run", [SIDE_BY_SIDE, SHARED])
    join_gap = 0
    if mode == SHARED:
        join_gap = st.slider("Each next flow joins after (s)", 0, 10, 5)
    start = st.button("Run simulation", type="primary", disabled=not names)

with live_tab:
    if start:
        try:
            with st.spinner("Loading controllers"):
                runs = build_runs(names, mode, bandwidth, delay, queue, loss_pct, varying, duration, join_gap)
        except FileNotFoundError as error:
            st.error(f"A trained model is missing ({error.filename}). Train it first; see README.md.")
            st.stop()
        chart = st.empty()
        for update in range(LIVE_UPDATES):
            for sim, _ in runs:
                sim.run(duration / LIVE_UPDATES)
            chart.plotly_chart(figure(runs, mode, duration), width="stretch", key=f"live_{update}")
        chart.empty()
        warmup = min(5.0, duration / 4) + join_gap * (len(names) - 1)
        table, fairness = summary(runs, mode, warmup)
        st.session_state["result"] = {"figure": figure(runs, mode, duration), "table": table,
                                      "fairness": fairness, "warmup": warmup}
    if "result" in st.session_state:
        show_result(st.session_state["result"])
    else:
        st.info("Choose a network and controllers in the sidebar, then press **Run simulation**.")
        st.markdown(
            "- **Reno, CUBIC, BBR**: classical controllers with hand-written rules.\n"
            "- **CUBIC+ML**: CUBIC that asks a trained classifier whether a loss was really congestion.\n"
            "- **PPO**: a reinforcement-learning agent that sets the sending rate directly.\n"
            "- **CUBIC+PPO**: CUBIC with a reinforcement-learning agent adjusting its window.\n"
            "- **PPO-Switch**: the PPO agent with a competitive mode, so it can share a link with CUBIC.\n"
            "- **PPO-J**: the PPO agent retrained with random RTT jitter.\n"
            "- **PPO-W**: the PPO agent trained on recordings of our real Wi-Fi."
        )

with results_tab:
    for heading, figures in GALLERY:
        st.subheader(heading)
        for file, caption in figures:
            if (RESULTS / file).exists():
                st.image(str(RESULTS / file), caption=caption)
