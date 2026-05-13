"""
Lottery Simulator & Analyzer
============================
A production-grade Streamlit application engineered to directly ingest 
Multipasko.pl PDF maps (Lotto 6/49).

Author: Senior Python / Data Engineer
"""

from __future__ import annotations

import io
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pdfplumber

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POOL_MIN: int = 1
POOL_MAX: int = 49
NUMBERS_PER_DRAW: int = 6
POOL: List[int] = list(range(POOL_MIN, POOL_MAX + 1))
MIN_DRAW_ID: int = 1000

# ---------------------------------------------------------------------------
# Page configuration & global styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lottery Simulator & Analyzer",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Inject the application-wide CSS theme."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link
            href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap"
            rel="stylesheet">

        <style>
            :root {
                --bg-deep:        #0b1014;
                --bg-panel:       #141b22;
                --bg-panel-soft:  #1b242d;
                --border:         #243038;
                --text-primary:   #eef2f5;
                --text-secondary: #9aa7b2;
                --accent:         #2ecc8a;
                --accent-soft:    #26a76f;
                --gold:           #d4af37;
                --gold-soft:      #b8962d;
                --danger:         #e25c5c;
                --cold:           #4ea8de;
                --shadow-lg:      0 12px 32px rgba(0, 0, 0, .45);
                --radius:         14px;
            }
            .stApp {
                background:
                    radial-gradient(1200px 600px at 15% -10%, rgba(46, 204, 138, .08), transparent 60%),
                    radial-gradient(900px  500px at 90% 110%, rgba(212, 175, 55, .06), transparent 60%),
                    var(--bg-deep);
                color: var(--text-primary);
                font-family: 'Montserrat', system-ui, sans-serif;
            }
            h1, h2, h3, h4, h5, h6, p, label, span, div { font-family: 'Montserrat', sans-serif; }
            code, pre, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }
            .hero {
                padding: 28px 32px;
                border-radius: var(--radius);
                background: linear-gradient(135deg, #141b22 0%, #1b242d 100%);
                border: 1px solid var(--border);
                box-shadow: var(--shadow-lg);
                margin-bottom: 22px;
            }
            .hero h1 {
                font-size: 2.4rem; margin: 0 0 6px 0;
                background: linear-gradient(90deg, #ffffff 0%, var(--accent) 60%, var(--gold) 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
            }
            .hero p { color: var(--text-secondary); margin: 0; font-size: 1rem; }
            section[data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--border); }
            div[data-testid="stMetric"] {
                background: var(--bg-panel); border: 1px solid var(--border);
                border-radius: var(--radius); padding: 14px 18px;
            }
            div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: .75rem !important; }
            div[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700; }
            .stButton > button {
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%);
                color: #0b1014; font-weight: 700; border: none; border-radius: 10px;
                padding: 10px 22px; transition: all .25s ease;
            }
            .stButton > button:hover { transform: translateY(-1px); color: #0b1014; }
            .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border); }
            .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }
            .ball-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 4px 0; }
            .ball {
                width: 52px; height: 52px; border-radius: 50%; display: flex; align-items: center;
                justify-content: center; font-weight: 700; font-size: 1.1rem; color: #0b1014;
                background: radial-gradient(circle at 30% 30%, #ffe89a 0%, var(--gold) 55%, var(--gold-soft) 100%);
            }
            .ball.hot { background: radial-gradient(circle at 30% 30%, #ffb199 0%, #e25c5c 55%, #b53a3a 100%); color: #fff; }
            .ball.cold { background: radial-gradient(circle at 30% 30%, #b5d8f0 0%, var(--cold) 55%, #2c6fa3 100%); color: #fff; }
            .ticket-card {
                background: var(--bg-panel); border: 1px solid var(--border);
                border-radius: var(--radius); padding: 16px 18px; margin-bottom: 12px;
            }
            .ticket-meta { color: var(--text-secondary); font-size: .8rem; text-transform: uppercase; margin-bottom: 8px; }
            .ticket-stats { color: var(--text-secondary); font-size: .82rem; margin-top: 8px; }
            [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius); }
            [data-testid="stFileUploaderDropzone"] { background: var(--bg-panel-soft); border: 1.5px dashed var(--border); }
            #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Bulletproof PDF Parsing (Multipasko Map Format)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
_INT_RE = re.compile(r"\b\d+\b")

@st.cache_data(show_spinner=False)
def parse_multipasko_pdf(file_bytes: bytes) -> pd.DataFrame:
    """
    Precision parser using pdfplumber to read the PDF preserving its visual layout.
    By reading row-by-row visually, it perfectly captures the Draw ID and its 6 balls.
    """
    records: Dict[int, Tuple[int, ...]] = {}
    
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # Extract text keeping visual spacing intact. This is the silver bullet.
            text = page.extract_text(layout=True)
            if not text:
                continue
                
            for line in text.splitlines():
                # Clean up any potential dates that look like numbers
                clean_line = _DATE_RE.sub(" ", line)
                
                # Extract all standalone integers from the visual line
                tokens = _INT_RE.findall(clean_line)
                if not tokens:
                    continue
                    
                ints = [int(t) for t in tokens]
                
                draw_id = None
                balls = []
                
                for val in ints:
                    # The first number >= 1000 on a line is strictly our Draw ID
                    if val >= MIN_DRAW_ID and draw_id is None:
                        draw_id = val
                    # Any valid number between 1 and 49 is a ball
                    elif POOL_MIN <= val <= POOL_MAX:
                        balls.append(val)
                
                # If we found a valid Draw ID and EXACTLY 6 unique balls on this specific row
                if draw_id is not None:
                    uniq_balls = tuple(sorted(set(balls)))
                    if len(uniq_balls) == NUMBERS_PER_DRAW:
                        # Save the draw. If duplicate IDs exist, first one wins.
                        if draw_id not in records:
                            records[draw_id] = uniq_balls

    if not records:
        return pd.DataFrame(columns=["draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "order"])

    # Build the final dataframe and sort chronologically (oldest first for analysis)
    df = pd.DataFrame(
        [{"draw_id": did, **{f"n{i+1}": v for i, v in enumerate(nums)}}
         for did, nums in records.items()]
    )
    df = df.sort_values("draw_id", ascending=True, kind="mergesort").reset_index(drop=True)
    df["order"] = np.arange(len(df), dtype=np.int64)
    return df


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

@dataclass
class HistoryStats:
    total_draws: int
    frequency: pd.Series
    last_seen: Dict[int, int]
    gaps: Dict[int, int]
    pair_counts: Counter
    triplet_counts: Counter
    gap_histogram: Dict[int, Counter]
    mean_gap: Dict[int, float]

@st.cache_data(show_spinner=False)
def compute_stats(df: pd.DataFrame) -> HistoryStats:
    n_draws = len(df)
    freq = Counter()
    pair_counts: Counter = Counter()
    triplet_counts: Counter = Counter()
    last_seen: Dict[int, int] = {}
    appearances: Dict[int, List[int]] = defaultdict(list)

    num_cols = [f"n{i+1}" for i in range(NUMBERS_PER_DRAW)]
    matrix = df[num_cols].to_numpy(dtype=np.int64)

    for order_idx, row in enumerate(matrix):
        nums = tuple(int(x) for x in row)
        freq.update(nums)
        for n in nums:
            appearances[n].append(order_idx)
            last_seen[n] = order_idx
        for a, b in combinations(nums, 2):
            pair_counts[(a, b)] += 1
        for a, b, c in combinations(nums, 3):
            triplet_counts[(a, b, c)] += 1

    freq_series = pd.Series({n: freq.get(n, 0) for n in POOL}, name="frequency").sort_index()

    gaps: Dict[int, int] = {}
    for n in POOL:
        if n in last_seen:
            gaps[n] = (n_draws - 1) - last_seen[n]
        else:
            gaps[n] = n_draws

    gap_hist: Dict[int, Counter] = {n: Counter() for n in POOL}
    mean_gap: Dict[int, float] = {}
    for n in POOL:
        apps = appearances.get(n, [])
        if len(apps) >= 2:
            deltas = np.diff(apps)
            for d in deltas:
                gap_hist[n][int(d)] += 1
            mean_gap[n] = float(np.mean(deltas))
        else:
            mean_gap[n] = float(n_draws)

    return HistoryStats(
        total_draws=n_draws, frequency=freq_series, last_seen=last_seen,
        gaps=gaps, pair_counts=pair_counts, triplet_counts=triplet_counts,
        gap_histogram=gap_hist, mean_gap=mean_gap,
    )

# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _normalize(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).copy()
    w[w < 0] = 0.0
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(w, 1.0 / len(w))
    return w / total

def _base_weights(stats: HistoryStats, mode: str, intensity: float) -> np.ndarray:
    freq = stats.frequency.reindex(POOL).fillna(0).to_numpy(dtype=np.float64)
    gaps = np.array([stats.gaps[n] for n in POOL], dtype=np.float64)
    mean_gap = np.array([stats.mean_gap[n] for n in POOL], dtype=np.float64)
    uniform = np.full(len(POOL), 1.0 / len(POOL))

    if mode == "hot":
        weighted = _normalize(freq + 1.0)
    elif mode == "cold":
        overdue_ratio = np.where(mean_gap > 0, gaps / mean_gap, 1.0)
        weighted = _normalize((1.0 / (freq + 1.0)) * (1.0 + overdue_ratio))
    elif mode == "balanced":
        empirical = _normalize(freq + 1.0)
        overdue_ratio = np.where(mean_gap > 0, gaps / mean_gap, 1.0)
        overdue = _normalize(overdue_ratio + 0.1)
        weighted = _normalize(0.5 * empirical + 0.5 * overdue)
    else:
        weighted = uniform

    final = intensity * weighted + (1.0 - intensity) * uniform
    return _normalize(final)

def _pair_affinity_boost(chosen: Sequence[int], candidate_weights: np.ndarray, pair_counts: Counter, affinity_strength: float) -> np.ndarray:
    if affinity_strength <= 0 or not chosen:
        return candidate_weights
    boost = np.zeros(len(POOL), dtype=np.float64)
    for i, n in enumerate(POOL):
        if n in chosen: continue
        score = 0.0
        for c in chosen:
            a, b = (c, n) if c < n else (n, c)
            score += pair_counts.get((a, b), 0)
        boost[i] = score
    if boost.max() == 0:
        return candidate_weights
    boost = boost / boost.max()
    multiplier = 1.0 + affinity_strength * boost
    return _normalize(candidate_weights * multiplier)

def simulate_draw(stats: HistoryStats, mode: str, intensity: float, affinity_strength: float, rng: np.random.Generator) -> Tuple[int, ...]:
    base = _base_weights(stats, mode, intensity)
    chosen: List[int] = []
    available = np.ones(len(POOL), dtype=bool)

    for _ in range(NUMBERS_PER_DRAW):
        w = base.copy()
        w = _pair_affinity_boost(chosen, w, stats.pair_counts, affinity_strength)
        w = np.where(available, w, 0.0)
        w = _normalize(w)
        pick_idx = int(rng.choice(len(POOL), p=w))
        chosen.append(POOL[pick_idx])
        available[pick_idx] = False

    return tuple(sorted(chosen))

def simulate_many(stats: HistoryStats, n_tickets: int, mode: str, intensity: float, affinity_strength: float, seed: Optional[int] = None) -> List[Tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    return [simulate_draw(stats, mode, intensity, affinity_strength, rng) for _ in range(n_tickets)]

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Montserrat, sans-serif", color="#eef2f5", size=13),
    margin=dict(l=10, r=10, t=50, b=10),title_font=dict(size=16, color="#eef2f5")) ,
)

def plot_frequency(stats: HistoryStats) -> go.Figure:
    freq = stats.frequency.reindex(POOL).fillna(0)
    fig = go.Figure(go.Bar(
        x=freq.index.astype(str), y=freq.values,
        marker=dict(color=freq.values, colorscale=[[0.0, "#4ea8de"], [0.5, "#2ecc8a"], [1.0, "#e25c5c"]], line=dict(width=0)),
        hovertemplate="Number <b>%{x}</b><br>Drawn %{y} times<extra></extra>",
    ))
    fig.update_layout(title="Frequency of each number across the entire history", xaxis_title="Number", yaxis_title="Draws appeared in", **_PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False, tickangle=0); fig.update_yaxes(gridcolor="#243038", zeroline=False)
    return fig

def plot_hot_cold(stats: HistoryStats, top_n: int = 10) -> go.Figure:
    freq = stats.frequency.sort_values(ascending=False)
    hot = freq.head(top_n)[::-1]; cold = freq.sort_values().head(top_n)[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hot.values, y=[f"#{n}" for n in hot.index], orientation="h", marker_color="#e25c5c", name="Hot", hovertemplate="<b>%{y}</b><br>%{x} appearances<extra></extra>"))
    fig.add_trace(go.Bar(x=-cold.values, y=[f"#{n}" for n in cold.index], orientation="h", marker_color="#4ea8de", name="Cold", hovertemplate="<b>%{y}</b><br>%{customdata} appearances<extra></extra>", customdata=cold.values))
    fig.update_layout(title=f"Top {top_n} hottest and coldest numbers", barmode="overlay", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), **_PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#243038", zeroline=True, zerolinecolor="#243038", tickvals=[-cold.values.max(), 0, hot.values.max()], ticktext=[str(int(cold.values.max())), "0", str(int(hot.values.max()))])
    fig.update_yaxes(showgrid=False)
    return fig

def plot_gaps(stats: HistoryStats) -> go.Figure:
    gaps = pd.Series(stats.gaps).reindex(POOL).fillna(0).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=[f"#{n}" for n in gaps.index], y=gaps.values,
        marker=dict(color=gaps.values, colorscale=[[0.0, "#2ecc8a"], [1.0, "#d4af37"]], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Asleep for %{y} draws<extra></extra>",
    ))
    fig.update_layout(title="Current sleep period — draws since last appearance", xaxis_title="Number (sorted by sleep)", yaxis_title="Draws since last appearance", **_PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False, tickangle=-60); fig.update_yaxes(gridcolor="#243038", zeroline=False)
    return fig

def plot_top_pairs(stats: HistoryStats, top_n: int = 15) -> go.Figure:
    if not stats.pair_counts: return go.Figure()
    items = stats.pair_counts.most_common(top_n)[::-1]
    labels = [f"{a} & {b}" for (a, b), _ in items]; counts = [c for _, c in items]
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker=dict(color=counts, colorscale=[[0.0, "#26a76f"], [1.0, "#d4af37"]], line=dict(width=0)), hovertemplate="<b>%{y}</b><br>Together in %{x} draws<extra></extra>"))
    fig.update_layout(title=f"Top {top_n} most frequently co-drawn pairs", xaxis_title="Joint appearances", yaxis_title="", **_PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#243038"); fig.update_yaxes(showgrid=False)
    return fig

def plot_top_triplets(stats: HistoryStats, top_n: int = 10) -> go.Figure:
    if not stats.triplet_counts: return go.Figure()
    items = stats.triplet_counts.most_common(top_n)[::-1]
    labels = [f"{a} · {b} · {c}" for (a, b, c), _ in items]; counts = [c for _, c in items]
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker=dict(color=counts, colorscale=[[0.0, "#4ea8de"], [1.0, "#2ecc8a"]], line=dict(width=0)), hovertemplate="<b>%{y}</b><br>Together in %{x} draws<extra></extra>"))
    fig.update_layout(title=f"Top {top_n} most frequently co-drawn triplets", xaxis_title="Joint appearances", **_PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#243038"); fig.update_yaxes(showgrid=False)
    return fig

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_balls(numbers: Sequence[int], hot_set: set, cold_set: set) -> str:
    out = ['<div class="ball-row">']
    for n in numbers:
        cls = "ball"
        if n in hot_set: cls += " hot"
        elif n in cold_set: cls += " cold"
        out.append(f'<span class="{cls}">{n}</span>')
    out.append("</div>")
    return "".join(out)

def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🎰 LotusWygranus 2.0: PDF Matrix Core</h1>
            <p>Upload your native Multipasko.pl PDF map directly. The engine will visually parse the grid, 
            learn the patterns, and generate statistically-tuned simulations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    inject_custom_css()
    hero()

    with st.sidebar:
        st.markdown("## Data source")
        uploaded = st.file_uploader(
            "Upload Multipasko Map (.pdf)",
            type=["pdf"],
            help="Upload the raw PDF downloaded directly from Multipasko. The engine will preserve its visual layout for flawless extraction.",
        )
        use_demo = st.checkbox("Use demo history", value=False)

        st.markdown("## Simulation")
        mode = st.radio("Strategy", options=["balanced", "hot", "cold"], format_func=lambda m: {"hot": "🔥 Hot", "cold": "❄️ Cold", "balanced": "⚖️ Balanced"}[m], index=0)
        intensity = st.slider("Strategy intensity", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        affinity = st.slider("Pair affinity strength", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        n_tickets = st.number_input("Tickets to generate", 1, 100, 6, step=1)
        seed_input = st.text_input("Random seed (optional)", value="")
        try: seed: Optional[int] = int(seed_input) if seed_input.strip() else None
        except ValueError: seed = None

        generate = st.button("🎲 Generate tickets", use_container_width=True)

    df = pd.DataFrame()
    
    if uploaded is not None:
        try:
            file_bytes = uploaded.read()
            df = parse_multipasko_pdf(file_bytes)
        except Exception as exc:
            st.error(f"Could not parse the PDF file. Error: {exc}")
            return
    elif use_demo:
        # Generate some synthetic data if no file is present
        rng = np.random.default_rng(42)
        records = []
        for i in range(1000):
            records.append({"draw_id": 6000 + i, **{f"n{j+1}": v for j, v in enumerate(sorted(rng.choice(POOL, 6, replace=False)))}})
        df = pd.DataFrame(records)
        df["order"] = np.arange(len(df))

    if df.empty:
        st.info("👈 Upload your **L1.pdf** (Multipasko Map) in the sidebar to get started.")
        return

    stats = compute_stats(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Draws parsed", f"{stats.total_draws:,}")
    c2.metric("Draw id range", f"{int(df['draw_id'].min())} → {int(df['draw_id'].max())}")
    c3.metric("Hottest number", f"#{int(stats.frequency.idxmax())}", f"{int(stats.frequency.max())} draws")
    c4.metric("Coldest number", f"#{int(stats.frequency.idxmin())}", f"{int(stats.frequency.min())} draws")

    tab_sim, tab_freq, tab_pairs, tab_gaps, tab_history = st.tabs(["🎲 Simulator", "🔥 Frequency", "🔗 Pairs & Triplets", "💤 Sleep", "📜 History"])
    hot_set = set(stats.frequency.sort_values(ascending=False).head(10).index.tolist())
    cold_set = set(stats.frequency.sort_values(ascending=True).head(10).index.tolist())

    with tab_sim:
        st.subheader("Generated tickets")
        if generate or "tickets" not in st.session_state:
            st.session_state["tickets"] = simulate_many(stats, int(n_tickets), mode, intensity, affinity, seed=seed)
            st.session_state["tickets_mode"] = mode
        
        for i, ticket in enumerate(st.session_state["tickets"], start=1):
            ticket_sum, odd_count, spread, avg_freq = sum(ticket), sum(1 for n in ticket if n % 2 == 1), max(ticket) - min(ticket), float(np.mean([stats.frequency[n] for n in ticket]))
            st.markdown(f"""
                <div class="ticket-card">
                    <div class="ticket-meta">Ticket #{i} · {st.session_state['tickets_mode'].title()} strategy</div>
                    {render_balls(ticket, hot_set, cold_set)}
                    <div class="ticket-stats">Sum: <b>{ticket_sum}</b> &nbsp;·&nbsp; Odd / Even: <b>{odd_count} / {NUMBERS_PER_DRAW - odd_count}</b> &nbsp;·&nbsp; Spread: <b>{spread}</b> &nbsp;·&nbsp; Avg. hist. freq: <b>{avg_freq:.1f}</b></div>
                </div>
            """, unsafe_allow_html=True)
            
        out_df = pd.DataFrame(st.session_state["tickets"], columns=[f"n{i+1}" for i in range(NUMBERS_PER_DRAW)])
        st.download_button("⬇️ Download tickets as CSV", data=out_df.to_csv(index=False).encode("utf-8"), file_name="simulated_tickets.csv", mime="text/csv")

    with tab_freq:
        st.plotly_chart(plot_frequency(stats), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a: st.plotly_chart(plot_hot_cold(stats, top_n=10), use_container_width=True)
        with col_b:
            freq_df = stats.frequency.rename_axis("number").reset_index().rename(columns={"frequency": "appearances"})
            freq_df["share_%"] = (freq_df["appearances"] / (stats.total_draws * NUMBERS_PER_DRAW) * 100).round(2)
            st.dataframe(freq_df, use_container_width=True, hide_index=True, height=420)

    with tab_pairs:
        col_a, col_b = st.columns(2)
        with col_a: st.plotly_chart(plot_top_pairs(stats, top_n=15), use_container_width=True)
        with col_b: st.plotly_chart(plot_top_triplets(stats, top_n=10), use_container_width=True)

    with tab_gaps:
        st.plotly_chart(plot_gaps(stats), use_container_width=True)
        gap_df = pd.Series(stats.gaps, name="current_sleep").rename_axis("number").reset_index()
        gap_df["mean_gap"] = gap_df["number"].map(stats.mean_gap).round(2)
        gap_df["overdue_ratio"] = (gap_df["current_sleep"] / gap_df["mean_gap"].replace(0, np.nan)).round(2)
        st.dataframe(gap_df.sort_values("overdue_ratio", ascending=False), use_container_width=True, hide_index=True, height=420)

    with tab_history:
        st.markdown("**Parsed draw history (chronological, newest at the top)**")
        st.dataframe(df.sort_values("draw_id", ascending=False).reset_index(drop=True)[["draw_id", "n1", "n2", "n3", "n4", "n5", "n6"]], use_container_width=True, hide_index=True, height=520)

if __name__ == "__main__":
    main()
