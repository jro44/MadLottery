"""
Lottery Simulator & Analyzer
============================
A production-grade Streamlit application that ingests a raw history of 6/49 lotto
draws, builds a statistical "memory" of the historical patterns, and acts as a
drawing machine that generates simulated future draws with three different
strategies (hot, cold, balanced).

Run with:
    streamlit run app.py

Author: Senior Data Scientist / Principal Streamlit Engineer
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POOL_MIN: int = 1
POOL_MAX: int = 49
NUMBERS_PER_DRAW: int = 6
POOL: List[int] = list(range(POOL_MIN, POOL_MAX + 1))

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

            /* Base layout ------------------------------------------------ */
            .stApp {
                background:
                    radial-gradient(1200px 600px at 15% -10%, rgba(46, 204, 138, .08), transparent 60%),
                    radial-gradient(900px  500px at 90% 110%, rgba(212, 175, 55, .06), transparent 60%),
                    var(--bg-deep);
                color: var(--text-primary);
                font-family: 'Montserrat', system-ui, sans-serif;
            }

            /* Typography ------------------------------------------------- */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                letter-spacing: -0.01em;
                color: var(--text-primary);
            }
            h1 { font-weight: 800; letter-spacing: -0.025em; }
            p, label, span, div { font-family: 'Montserrat', sans-serif; }

            code, pre, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

            /* Hero header ------------------------------------------------ */
            .hero {
                padding: 28px 32px;
                border-radius: var(--radius);
                background: linear-gradient(135deg, #141b22 0%, #1b242d 100%);
                border: 1px solid var(--border);
                box-shadow: var(--shadow-lg);
                margin-bottom: 22px;
            }
            .hero h1 {
                font-size: 2.4rem;
                margin: 0 0 6px 0;
                background: linear-gradient(90deg, #ffffff 0%, var(--accent) 60%, var(--gold) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .hero p {
                color: var(--text-secondary);
                margin: 0;
                font-size: 1rem;
                font-weight: 400;
            }

            /* Sidebar ---------------------------------------------------- */
            section[data-testid="stSidebar"] {
                background: var(--bg-panel);
                border-right: 1px solid var(--border);
            }
            section[data-testid="stSidebar"] .stMarkdown h2 {
                color: var(--accent);
                font-size: 1rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: .08em;
                margin-top: 1.2rem;
            }

            /* Cards & metrics ------------------------------------------- */
            div[data-testid="stMetric"] {
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 14px 18px;
                transition: transform .2s ease, border-color .2s ease;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                border-color: var(--accent-soft);
            }
            div[data-testid="stMetricLabel"] {
                color: var(--text-secondary) !important;
                font-size: .75rem !important;
                text-transform: uppercase;
                letter-spacing: .08em;
            }
            div[data-testid="stMetricValue"] {
                color: var(--text-primary) !important;
                font-weight: 700;
            }

            /* Buttons ---------------------------------------------------- */
            .stButton > button {
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%);
                color: #0b1014;
                font-weight: 700;
                border: none;
                border-radius: 10px;
                padding: 10px 22px;
                letter-spacing: .02em;
                transition: all .25s ease;
                box-shadow: 0 4px 14px rgba(46, 204, 138, .25);
            }
            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 22px rgba(46, 204, 138, .4);
                color: #0b1014;
            }
            .stButton > button:active { transform: translateY(0); }

            /* Tabs ------------------------------------------------------- */
            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
                background: transparent;
                border-bottom: 1px solid var(--border);
            }
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: var(--text-secondary);
                font-weight: 500;
                padding: 10px 18px;
                border-radius: 8px 8px 0 0;
            }
            .stTabs [aria-selected="true"] {
                background: var(--bg-panel);
                color: var(--accent) !important;
                border-bottom: 2px solid var(--accent);
            }

            /* Number "balls" -------------------------------------------- */
            .ball-row {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 8px 0 4px 0;
            }
            .ball {
                width: 52px;
                height: 52px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 1.1rem;
                color: #0b1014;
                background: radial-gradient(circle at 30% 30%, #ffe89a 0%, var(--gold) 55%, var(--gold-soft) 100%);
                box-shadow:
                    inset 0 -4px 6px rgba(0,0,0,.25),
                    0 4px 10px rgba(212, 175, 55, .35);
                font-family: 'Montserrat', sans-serif;
            }
            .ball.hot {
                background: radial-gradient(circle at 30% 30%, #ffb199 0%, #e25c5c 55%, #b53a3a 100%);
                box-shadow: inset 0 -4px 6px rgba(0,0,0,.25), 0 4px 10px rgba(226, 92, 92, .35);
                color: #fff;
            }
            .ball.cold {
                background: radial-gradient(circle at 30% 30%, #b5d8f0 0%, var(--cold) 55%, #2c6fa3 100%);
                box-shadow: inset 0 -4px 6px rgba(0,0,0,.25), 0 4px 10px rgba(78, 168, 222, .35);
                color: #fff;
            }
            .ticket-card {
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 16px 18px;
                margin-bottom: 12px;
                transition: border-color .2s ease, transform .2s ease;
            }
            .ticket-card:hover {
                border-color: var(--gold-soft);
                transform: translateY(-1px);
            }
            .ticket-meta {
                color: var(--text-secondary);
                font-size: .8rem;
                text-transform: uppercase;
                letter-spacing: .08em;
                margin-bottom: 8px;
            }
            .ticket-stats {
                color: var(--text-secondary);
                font-size: .82rem;
                margin-top: 8px;
            }

            /* DataFrames ------------------------------------------------- */
            [data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: var(--radius);
                overflow: hidden;
            }

            /* File uploader --------------------------------------------- */
            [data-testid="stFileUploaderDropzone"] {
                background: var(--bg-panel-soft);
                border: 1.5px dashed var(--border);
                border-radius: var(--radius);
            }

            /* Hide Streamlit chrome ------------------------------------- */
            #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data parsing (Advanced PDF-Extract Token Stream)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
MIN_DRAW_ID = 1000

def _smart_split_token(token_str: str) -> List[int]:
    """
    Splits concatenated numbers (e.g., '1920' -> [19, 20] or '131415' -> [13, 14, 15])
    that occur due to missing whitespace in PDF extraction.
    Prefers splitting if the resulting pieces are valid 1-49 lotto balls.
    """
    n = len(token_str)
    
    # 1 or 2 digits - just parse it
    if n <= 2:
        return [int(token_str)]
        
    # Even length (4, 6, 8) - attempt to split into 2-digit chunks
    if n % 2 == 0:
        chunks = [int(token_str[i:i+2]) for i in range(0, n, 2)]
        if all(POOL_MIN <= c <= POOL_MAX for c in chunks):
            return chunks
            
    # Length 3 (e.g., '708') - try splitting as '7' and '08'
    if n == 3:
        c1, c2 = int(token_str[0]), int(token_str[1:])
        if POOL_MIN <= c1 <= POOL_MAX and POOL_MIN <= c2 <= POOL_MAX:
            return [c1, c2]
        # Or '70' and '8'
        c1, c2 = int(token_str[:2]), int(token_str[2])
        if POOL_MIN <= c1 <= POOL_MAX and POOL_MIN <= c2 <= POOL_MAX:
            return [c1, c2]

    # If it can't be cleanly split into valid balls, return as is
    # (It might be a valid Draw ID >= 1000)
    return [int(token_str)]

@st.cache_data(show_spinner=False)
def parse_history(raw_text: str) -> pd.DataFrame:
    """
    Parse a chaotic, column-major or interleaved text dump into a DataFrame.
    Uses a Token-Stream FIFO queue approach to rebuild draws.
    """
    # 1. Clean explicit text noise that produces false tokens
    text = raw_text.replace("6/49", " ")
    text = re.sub(r'(?i)lotto', ' ', text)
    text = _DATE_RE.sub(" ", text)
    
    # 2. Chunk by page to prevent FIFO drift cascading across the entire file
    # The regex looks for "--- PAGE X ---" or "" patterns
        page_chunks = re.split(r'--- PAGE \d+ ---|\', text)
    if not page_chunks or len(page_chunks) == 1:
        page_chunks = [text]
        
    records: Dict[int, Tuple[int, ...]] = {}
    
    # Process each page chunk independently
    for chunk in page_chunks:
        raw_tokens = re.findall(r"\d+", chunk)
        
        stream: List[int] = []
        for tok in raw_tokens:
            stream.extend(_smart_split_token(tok))
            
        ids_queue: List[int] = []
        balls_queue: List[int] = []
        
        # Route tokens to appropriate queues
        for val in stream:
            if val >= MIN_DRAW_ID:
                ids_queue.append(val)
            elif POOL_MIN <= val <= POOL_MAX:
                balls_queue.append(val)
            # Tokens between 50 and 999 are considered noise and ignored
                
        # Match IDs to Balls sequentially (FIFO)
        while ids_queue and len(balls_queue) >= NUMBERS_PER_DRAW:
            draw_id = ids_queue.pop(0)
            
            # Consume 6 balls from the buffer
            current_balls = balls_queue[:NUMBERS_PER_DRAW]
            balls_queue = balls_queue[NUMBERS_PER_DRAW:]
            
            # Validate the draw (no duplicates within the 6 balls)
            uniq = tuple(sorted(set(current_balls)))
            if len(uniq) == NUMBERS_PER_DRAW:
                # Store the first occurrence (to handle any duplicate ID bugs)
                if draw_id not in records:
                    records[draw_id] = uniq

    if not records:
        return pd.DataFrame(columns=["draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "order"])

    # 3. Build the final dataframe and sort chronologically
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
    """Aggregated, model-ready statistics derived from the draw history."""
    total_draws: int
    frequency: pd.Series          # index = number, value = count
    last_seen: Dict[int, int]     # number -> chronological order index it last appeared
    gaps: Dict[int, int]          # number -> # draws since last appearance (current sleep)
    pair_counts: Counter          # (a,b) -> count, a<b
    triplet_counts: Counter       # (a,b,c) -> count, a<b<c
    gap_histogram: Dict[int, Counter]  # number -> Counter({gap_length: occurrences})
    mean_gap: Dict[int, float]    # number -> average gap between appearances


@st.cache_data(show_spinner=False)
def compute_stats(df: pd.DataFrame) -> HistoryStats:
    """Compute the full set of statistics used by the simulator and the UI.

    Runs in a single pass over the chronologically ordered history.
    """
    n_draws = len(df)
    freq = Counter()
    pair_counts: Counter = Counter()
    triplet_counts: Counter = Counter()
    last_seen: Dict[int, int] = {}
    appearances: Dict[int, List[int]] = defaultdict(list)

    num_cols = [f"n{i+1}" for i in range(NUMBERS_PER_DRAW)]
    matrix = df[num_cols].to_numpy(dtype=np.int64)

    for order_idx, row in enumerate(matrix):
        nums = tuple(int(x) for x in row)  # already sorted ascending
        freq.update(nums)
        for n in nums:
            appearances[n].append(order_idx)
            last_seen[n] = order_idx
        for a, b in combinations(nums, 2):
            pair_counts[(a, b)] += 1
        for a, b, c in combinations(nums, 3):
            triplet_counts[(a, b, c)] += 1

    # Frequency as a complete series across the entire pool (0 for unseen)
    freq_series = pd.Series(
        {n: freq.get(n, 0) for n in POOL}, name="frequency"
    ).sort_index()

    # Current "sleep" for each number = draws since last appearance.
    # If a number was never drawn we use n_draws (max possible sleep).
    gaps: Dict[int, int] = {}
    for n in POOL:
        if n in last_seen:
            gaps[n] = (n_draws - 1) - last_seen[n]
        else:
            gaps[n] = n_draws

    # Gap histograms + mean gap per number, computed from inter-appearance deltas.
    gap_hist: Dict[int, Counter] = {n: Counter() for n in POOL}
    mean_gap: Dict[int, float] = {}
    for n in POOL:
        apps = appearances.get(n, [])
        if len(apps) >= 2:
            deltas = np.diff(apps)
            for d in deltas:
                gap_hist[n][int(d)] += 1
            mean_gap[n] = float(np.mean(deltas))
        elif len(apps) == 1:
            mean_gap[n] = float(n_draws)
        else:
            mean_gap[n] = float(n_draws)

    return HistoryStats(
        total_draws=n_draws,
        frequency=freq_series,
        last_seen=last_seen,
        gaps=gaps,
        pair_counts=pair_counts,
        triplet_counts=triplet_counts,
        gap_histogram=gap_hist,
        mean_gap=mean_gap,
    )


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------


def _normalize(weights: np.ndarray) -> np.ndarray:
    """Return a strictly-positive, normalized probability vector."""
    w = np.asarray(weights, dtype=np.float64).copy()
    w[w < 0] = 0.0
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        # Fall back to uniform if everything collapsed to zero.
        return np.full_like(w, 1.0 / len(w))
    return w / total


def _base_weights(stats: HistoryStats, mode: str, intensity: float) -> np.ndarray:
    """Compute the per-number base weight vector for the given mode.

    ``intensity`` in [0, 1] interpolates between pure uniform (0) and the
    fully mode-driven weighting (1). This is what makes the simulator
    "tunable" – at intensity 0 every mode degenerates into fair random play.
    """
    freq = stats.frequency.reindex(POOL).fillna(0).to_numpy(dtype=np.float64)
    gaps = np.array([stats.gaps[n] for n in POOL], dtype=np.float64)
    mean_gap = np.array([stats.mean_gap[n] for n in POOL], dtype=np.float64)

    uniform = np.full(len(POOL), 1.0 / len(POOL))

    if mode == "hot":
        # Higher historical frequency => higher weight. Smoothed so cold
        # numbers still get a non-zero chance.
        smoothed = freq + 1.0  # Laplace smoothing
        weighted = _normalize(smoothed)
    elif mode == "cold":
        # Favour overdue numbers: weight grows with current sleep and falls
        # with historical frequency. Numbers sleeping longer than their own
        # mean gap get an explicit boost.
        overdue_ratio = np.where(mean_gap > 0, gaps / mean_gap, 1.0)
        inv_freq = 1.0 / (freq + 1.0)
        weighted = _normalize(inv_freq * (1.0 + overdue_ratio))
    elif mode == "balanced":
        # Blend the empirical frequency distribution with the overdue signal.
        empirical = _normalize(freq + 1.0)
        overdue_ratio = np.where(mean_gap > 0, gaps / mean_gap, 1.0)
        overdue = _normalize(overdue_ratio + 0.1)
        weighted = _normalize(0.5 * empirical + 0.5 * overdue)
    else:
        weighted = uniform

    # Blend toward uniform according to intensity.
    final = intensity * weighted + (1.0 - intensity) * uniform
    return _normalize(final)


def _pair_affinity_boost(
    chosen: Sequence[int],
    candidate_weights: np.ndarray,
    pair_counts: Counter,
    affinity_strength: float,
) -> np.ndarray:
    """Boost candidates that historically co-occur with already-chosen numbers."""
    if affinity_strength <= 0 or not chosen:
        return candidate_weights

    boost = np.zeros(len(POOL), dtype=np.float64)
    for i, n in enumerate(POOL):
        if n in chosen:
            continue
        score = 0.0
        for c in chosen:
            a, b = (c, n) if c < n else (n, c)
            score += pair_counts.get((a, b), 0)
        boost[i] = score

    if boost.max() == 0:
        return candidate_weights

    boost = boost / boost.max()  # normalise to [0,1]
    multiplier = 1.0 + affinity_strength * boost
    return _normalize(candidate_weights * multiplier)


def simulate_draw(
    stats: HistoryStats,
    mode: str,
    intensity: float,
    affinity_strength: float,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    """Generate a single simulated 6-number ticket.

    Numbers are drawn one at a time without replacement; after every pick the
    remaining weights are reshaped by the pair-affinity signal so that the
    final ticket reflects historical co-occurrence as well as marginal
    frequencies. Returned tuple is sorted ascending.
    """
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


def simulate_many(
    stats: HistoryStats,
    n_tickets: int,
    mode: str,
    intensity: float,
    affinity_strength: float,
    seed: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """Generate ``n_tickets`` simulated tickets."""
    rng = np.random.default_rng(seed)
    return [
        simulate_draw(stats, mode, intensity, affinity_strength, rng)
        for _ in range(n_tickets)
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Montserrat, sans-serif", color="#eef2f5", size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    title=dict(font=dict(size=16, color="#eef2f5")),
)


def plot_frequency(stats: HistoryStats) -> go.Figure:
    """Bar chart of frequency for all 49 numbers, coloured by hot/cold gradient."""
    freq = stats.frequency.reindex(POOL).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=freq.index.astype(str),
            y=freq.values,
            marker=dict(
                color=freq.values,
                colorscale=[
                    [0.0, "#4ea8de"],
                    [0.5, "#2ecc8a"],
                    [1.0, "#e25c5c"],
                ],
                line=dict(width=0),
            ),
            hovertemplate="Number <b>%{x}</b><br>Drawn %{y} times<extra></extra>",
        )
    )
    fig.update_layout(
        title="Frequency of each number across the entire history",
        xaxis_title="Number",
        yaxis_title="Draws appeared in",
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=False, tickangle=0)
    fig.update_yaxes(gridcolor="#243038", zeroline=False)
    return fig


def plot_hot_cold(stats: HistoryStats, top_n: int = 10) -> go.Figure:
    """Side-by-side hot vs. cold numbers."""
    freq = stats.frequency.sort_values(ascending=False)
    hot = freq.head(top_n)[::-1]
    cold = freq.sort_values().head(top_n)[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hot.values, y=[f"#{n}" for n in hot.index],
        orientation="h", marker_color="#e25c5c", name="Hot",
        hovertemplate="<b>%{y}</b><br>%{x} appearances<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=-cold.values, y=[f"#{n}" for n in cold.index],
        orientation="h", marker_color="#4ea8de", name="Cold",
        hovertemplate="<b>%{y}</b><br>%{customdata} appearances<extra></extra>",
        customdata=cold.values,
    ))
    fig.update_layout(
        title=f"Top {top_n} hottest and coldest numbers",
        barmode="overlay",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#243038", zeroline=True, zerolinecolor="#243038",
        tickvals=[-cold.values.max(), 0, hot.values.max()],
        ticktext=[str(int(cold.values.max())), "0", str(int(hot.values.max()))],
    )
    fig.update_yaxes(showgrid=False)
    return fig


def plot_gaps(stats: HistoryStats) -> go.Figure:
    """Bar chart of current 'sleep' for each number."""
    gaps = pd.Series(stats.gaps).reindex(POOL).fillna(0).sort_values(ascending=False)
    fig = go.Figure(
        go.Bar(
            x=[f"#{n}" for n in gaps.index],
            y=gaps.values,
            marker=dict(
                color=gaps.values,
                colorscale=[[0.0, "#2ecc8a"], [1.0, "#d4af37"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Asleep for %{y} draws<extra></extra>",
        )
    )
    fig.update_layout(
        title="Current sleep period — draws since last appearance",
        xaxis_title="Number (sorted by sleep)",
        yaxis_title="Draws since last appearance",
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=False, tickangle=-60)
    fig.update_yaxes(gridcolor="#243038", zeroline=False)
    return fig


def plot_top_pairs(stats: HistoryStats, top_n: int = 15) -> go.Figure:
    """Top historically co-occurring pairs."""
    if not stats.pair_counts:
        return go.Figure()
    items = stats.pair_counts.most_common(top_n)[::-1]
    labels = [f"{a} & {b}" for (a, b), _ in items]
    counts = [c for _, c in items]
    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker=dict(
            color=counts,
            colorscale=[[0.0, "#26a76f"], [1.0, "#d4af37"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Together in %{x} draws<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} most frequently co-drawn pairs",
        xaxis_title="Joint appearances",
        yaxis_title="",
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#243038")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_top_triplets(stats: HistoryStats, top_n: int = 10) -> go.Figure:
    """Top historically co-occurring triplets."""
    if not stats.triplet_counts:
        return go.Figure()
    items = stats.triplet_counts.most_common(top_n)[::-1]
    labels = [f"{a} · {b} · {c}" for (a, b, c), _ in items]
    counts = [c for _, c in items]
    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color=counts, colorscale=[[0.0, "#4ea8de"], [1.0, "#2ecc8a"]],
                    line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Together in %{x} draws<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} most frequently co-drawn triplets",
        xaxis_title="Joint appearances",
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#243038")
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def render_balls(numbers: Sequence[int], hot_set: set, cold_set: set) -> str:
    """Render a row of HTML "lottery balls", coloured by hot/cold membership."""
    out = ['<div class="ball-row">']
    for n in numbers:
        cls = "ball"
        if n in hot_set:
            cls += " hot"
        elif n in cold_set:
            cls += " cold"
        out.append(f'<span class="{cls}">{n}</span>')
    out.append("</div>")
    return "".join(out)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🎰 Lottery Simulator &amp; Analyzer</h1>
            <p>Upload a raw history of 6/49 draws, let the engine learn the patterns, and generate
            statistically-tuned simulations with hot, cold or balanced strategies.</p>
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

    # ----- Sidebar -------------------------------------------------------
    with st.sidebar:
        st.markdown("## Data source")
        uploaded = st.file_uploader(
            "Upload draw history (.txt)",
            type=["txt", "csv", "tsv", "log"],
            help="A plain-text export with one draw per line. The parser tolerates "
                 "varying separators, dates and column orders.",
        )
        use_demo = st.checkbox("Use a generated demo history instead", value=False,
                               help="Useful for trying the app without your own file.")

        st.markdown("## Simulation")
        mode = st.radio(
            "Strategy",
            options=["balanced", "hot", "cold"],
            format_func=lambda m: {
                "hot":      "🔥 Hot — favour frequent numbers",
                "cold":     "❄️ Cold — chase overdue numbers",
                "balanced": "⚖️ Balanced — frequency + overdue blend",
            }[m],
            index=0,
        )
        intensity = st.slider(
            "Strategy intensity",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="0 = fair random. 1 = fully driven by the chosen strategy.",
        )
        affinity = st.slider(
            "Pair affinity strength",
            min_value=0.0, max_value=1.0, value=0.4, step=0.05,
            help="Bias picks toward numbers that historically appeared together.",
        )
        n_tickets = st.number_input("Tickets to generate", 1, 100, 6, step=1)
        seed_input = st.text_input("Random seed (optional)", value="",
                                   help="Leave empty for non-deterministic runs.")
        try:
            seed: Optional[int] = int(seed_input) if seed_input.strip() else None
        except ValueError:
            seed = None
            st.warning("Seed must be an integer; ignoring.")

        generate = st.button("🎲 Generate tickets", use_container_width=True)

    # ----- Load data -----------------------------------------------------
    raw_text: Optional[str] = None
    if uploaded is not None:
        try:
            raw_bytes = uploaded.read()
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the uploaded file: {exc}")
            return
    elif use_demo:
        raw_text = _generate_demo_history()

    if raw_text is None:
        _empty_state()
        return

    df = parse_history(raw_text)
    if df.empty:
        st.error(
            "No valid 6/49 draws could be extracted from the uploaded file. "
            "Make sure each line contains a draw id followed by six numbers in the range 1–49."
        )
        return

    stats = compute_stats(df)

    # ----- Overview metrics ---------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Draws parsed", f"{stats.total_draws:,}")
    c2.metric("Draw id range",
              f"{int(df['draw_id'].min())} → {int(df['draw_id'].max())}")
    hottest = int(stats.frequency.idxmax())
    coldest = int(stats.frequency.idxmin())
    c3.metric("Hottest number", f"#{hottest}", f"{int(stats.frequency.max())} draws")
    c4.metric("Coldest number", f"#{coldest}", f"{int(stats.frequency.min())} draws")

    # ----- Tabs ----------------------------------------------------------
    tab_sim, tab_freq, tab_pairs, tab_gaps, tab_history = st.tabs(
        ["🎲 Simulator", "🔥 Frequency", "🔗 Pairs & Triplets", "💤 Sleep", "📜 History"]
    )

    hot_set = set(stats.frequency.sort_values(ascending=False).head(10).index.tolist())
    cold_set = set(stats.frequency.sort_values(ascending=True).head(10).index.tolist())

    # --- Simulator tab ---------------------------------------------------
    with tab_sim:
        st.subheader("Generated tickets")
        st.caption(
            "Balls coloured **red** are in the historical top-10 hottest numbers; "
            "**blue** are in the bottom-10 coldest. Gold balls are statistically neutral."
        )

        if generate or "tickets" not in st.session_state:
            tickets = simulate_many(
                stats, int(n_tickets), mode, intensity, affinity, seed=seed
            )
            st.session_state["tickets"] = tickets
            st.session_state["tickets_mode"] = mode

        tickets: List[Tuple[int, ...]] = st.session_state["tickets"]

        for i, ticket in enumerate(tickets, start=1):
            ticket_sum = sum(ticket)
            odd_count = sum(1 for n in ticket if n % 2 == 1)
            spread = max(ticket) - min(ticket)
            avg_freq = float(np.mean([stats.frequency[n] for n in ticket]))
            st.markdown(
                f"""
                <div class="ticket-card">
                    <div class="ticket-meta">Ticket #{i} · {st.session_state['tickets_mode'].title()} strategy</div>
                    {render_balls(ticket, hot_set, cold_set)}
                    <div class="ticket-stats">
                        Sum: <b>{ticket_sum}</b> &nbsp;·&nbsp;
                        Odd / Even: <b>{odd_count} / {NUMBERS_PER_DRAW - odd_count}</b> &nbsp;·&nbsp;
                        Spread: <b>{spread}</b> &nbsp;·&nbsp;
                        Avg. historical frequency: <b>{avg_freq:.1f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Downloadable CSV of the generated tickets
        out_df = pd.DataFrame(tickets, columns=[f"n{i+1}" for i in range(NUMBERS_PER_DRAW)])
        out_df.insert(0, "ticket", np.arange(1, len(tickets) + 1))
        st.download_button(
            "⬇️ Download tickets as CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="simulated_tickets.csv",
            mime="text/csv",
        )

    # --- Frequency tab ---------------------------------------------------
    with tab_freq:
        st.plotly_chart(plot_frequency(stats), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_hot_cold(stats, top_n=10), use_container_width=True)
        with col_b:
            freq_df = (
                stats.frequency
                .rename_axis("number")
                .reset_index()
                .rename(columns={"frequency": "appearances"})
            )
            freq_df["share_%"] = (freq_df["appearances"] /
                                  (stats.total_draws * NUMBERS_PER_DRAW) * 100).round(2)
            st.markdown("**Per-number breakdown**")
            st.dataframe(freq_df, use_container_width=True, hide_index=True, height=420)

    # --- Pairs & triplets tab -------------------------------------------
    with tab_pairs:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_top_pairs(stats, top_n=15), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_top_triplets(stats, top_n=10), use_container_width=True)

        pair_df = pd.DataFrame(
            [{"pair": f"{a} · {b}", "appearances": c}
             for (a, b), c in stats.pair_counts.most_common(50)]
        )
        st.markdown("**Top 50 pairs**")
        st.dataframe(pair_df, use_container_width=True, hide_index=True, height=320)

    # --- Sleep tab -------------------------------------------------------
    with tab_gaps:
        st.plotly_chart(plot_gaps(stats), use_container_width=True)
        gap_df = (
            pd.Series(stats.gaps, name="current_sleep")
            .rename_axis("number")
            .reset_index()
        )
        gap_df["mean_gap"] = gap_df["number"].map(stats.mean_gap).round(2)
        gap_df["overdue_ratio"] = (
            gap_df["current_sleep"] / gap_df["mean_gap"].replace(0, np.nan)
        ).round(2)
        gap_df = gap_df.sort_values("overdue_ratio", ascending=False)
        st.markdown(
            "**Overdue table** — `overdue_ratio = current sleep / historical mean gap`. "
            "Values above 1.0 indicate a number that's currently sleeping longer than usual."
        )
        st.dataframe(gap_df, use_container_width=True, hide_index=True, height=420)

    # --- History tab -----------------------------------------------------
    with tab_history:
        st.markdown("**Parsed draw history (chronological, newest at the top)**")
        display_df = df.sort_values("draw_id", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_df[["draw_id", "n1", "n2", "n3", "n4", "n5", "n6"]],
            use_container_width=True,
            hide_index=True,
            height=520,
        )


# ---------------------------------------------------------------------------
# Empty-state and demo helpers
# ---------------------------------------------------------------------------


def _empty_state() -> None:
    """What the user sees before any data is loaded."""
    st.info(
        "👈 Upload a `.txt` (or `.csv`) draw history in the sidebar to get started, "
        "or tick **Use a generated demo history** to explore the app immediately."
    )
    with st.expander("Expected file format (the parser is very forgiving)"):
        st.code(
            "7351   01.02.2024   3 11 17 22 38 45\n"
            "7350   29.01.2024   1  7 14 28 33 49\n"
            "7349:  5, 9, 19, 27, 31, 44\n"
            "Losowanie 7348 -- 02 10 21 24 36 41\n",
            language="text",
        )
        st.markdown(
            "- Whitespace, commas, colons and dashes between numbers are all fine.\n"
            "- Dates in `dd.mm.yyyy` / `dd-mm-yy` / `yyyy-mm-dd` format are stripped automatically.\n"
            "- The last six integers on each line in the 1–49 range are taken as the drawn numbers.\n"
            "- The first remaining integer (that isn't itself in 1–49) is taken as the draw id."
        )


def _generate_demo_history(n_draws: int = 1500, seed: int = 42) -> str:
    """Build a plausible-looking synthetic 6/49 history for demo purposes."""
    rng = random.Random(seed)
    # Give a few numbers a slightly higher base weight so the analytics aren't flat.
    weights = {n: 1.0 for n in POOL}
    for boosted in (7, 13, 17, 23, 31, 42):
        weights[boosted] = 1.6
    pop = list(weights.keys())
    wts = list(weights.values())

    lines: List[str] = []
    for i in range(n_draws):
        draw_id = 6000 + i
        picked: set = set()
        while len(picked) < NUMBERS_PER_DRAW:
            picked.add(rng.choices(pop, weights=wts, k=1)[0])
        nums = " ".join(f"{n:2d}" for n in sorted(picked))
        lines.append(f"{draw_id}   {nums}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
