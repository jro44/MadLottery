"""
LotusWygranus 4.0 ULTIMATE: Lotto 6/49 AI & Markov Sequences
=============================================================
Profesjonalny system predykcyjny analizujący "szlaki" maszyny losującej
bezpośrednio z mapy wizualnej L1.PDF.

Autor: Principal Data Scientist & Software Architect
"""

from __future__ import annotations

import io
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pdfplumber

# ---------------------------------------------------------------------------
# KONFIGURACJA LOTTO 6/49
# ---------------------------------------------------------------------------

MAIN_POOL_MIN, MAIN_POOL_MAX, MAIN_COUNT = 1, 49, 6
MAIN_POOL = list(range(MAIN_POOL_MIN, MAIN_POOL_MAX + 1))
ID_THRESHOLD = 50  # Numery losowań są znacznie wyższe niż kule (1-49)

# ---------------------------------------------------------------------------
# STYLIZACJA INTERFEJSU
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LotusWygranus 4.0 - Lotto", page_icon="🎰", layout="wide")

def inject_ultimate_css():
    st.markdown("""
        <style>
            :root {
                --bg-main: #0a0e14; --bg-panel: #161b22; --border: #30363d;
                --text-main: #c9d1d9; --text-muted: #8b949e;
                --accent-cyan: #58a6ff; --accent-gold: #d29922; --accent-danger: #f85149;
            }
            .stApp { background-color: var(--bg-main); color: var(--text-main); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
            h1, h2, h3 { color: var(--text-main); font-weight: 600; }
            
            .expert-box {
                background: rgba(88, 166, 255, 0.05);
                border: 1px solid var(--border);
                border-left: 4px solid var(--accent-cyan);
                padding: 16px; border-radius: 6px; margin-bottom: 20px; font-size: 0.9rem;
            }
            
            .ball-container { display: flex; gap: 8px; flex-wrap: wrap; margin: 15px 0; }
            .ball {
                width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
                font-weight: 700; font-size: 1.1rem; color: #0d1117;
                background: radial-gradient(circle at 30% 30%, #f0f6fc, var(--accent-gold));
                box-shadow: 0 4px 10px rgba(0,0,0,0.4);
            }
            
            .ticket-card {
                background: var(--bg-panel); border: 1px solid var(--border);
                border-radius: 8px; padding: 20px; margin-bottom: 12px;
                transition: border-color 0.2s;
            }
            .ticket-card:hover { border-color: var(--accent-cyan); }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PRECYZYJNY PARSER PDF (L1.PDF)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Analizuję mapę wizualną L1.PDF...")
def load_and_parse_lotto_db() -> pd.DataFrame:
    file_path = "L1.PDF"
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    records = {}
    _INT_RE = re.compile(r"\b\d+\b")
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if not text: continue
            
            for line in text.splitlines():
                # Usuwanie dat (np. 2024)
                line = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", line)
                ints = [int(t) for t in _INT_RE.findall(line)]
                if not ints: continue
                
                draw_id, balls = None, []
                for val in ints:
                    if val > ID_THRESHOLD and draw_id is None: draw_id = val
                    elif MAIN_POOL_MIN <= val <= MAIN_POOL_MAX: balls.append(val)
                        
                if draw_id is not None:
                    uniq_balls = tuple(sorted(set(balls)))
                    if len(uniq_balls) == MAIN_COUNT:
                        if draw_id not in records:
                            records[draw_id] = uniq_balls
                            
    df = pd.DataFrame([{"draw_id": did, **{f"n{i+1}": v for i, v in enumerate(nums)}} for did, nums in records.items()])
    if not df.empty:
        df = df.sort_values("draw_id", ascending=True).reset_index(drop=True)
        df["order"] = np.arange(len(df))
    return df

# ---------------------------------------------------------------------------
# SILNIK DATA SCIENCE
# ---------------------------------------------------------------------------

@dataclass
class LottoStats:
    frequency: pd.Series
    gaps: Dict[int, int]
    mean_gaps: Dict[int, float]
    affinity_matrix: pd.DataFrame
    transition_matrix: pd.DataFrame
    last_drawn: Tuple[int, ...]

@st.cache_data(show_spinner=False)
def compute_stats(df: pd.DataFrame) -> LottoStats:
    n_draws = len(df)
    matrix = df[[f"n{i+1}" for i in range(MAIN_COUNT)]].to_numpy()
    
    freq_counter = Counter(matrix.flatten())
    last_seen = {}
    appearances = defaultdict(list)
    aff_matrix = pd.DataFrame(0.0, index=MAIN_POOL, columns=MAIN_POOL)
    trans_matrix = pd.DataFrame(0.0, index=MAIN_POOL, columns=MAIN_POOL)
    
    for i in range(n_draws):
        row = matrix[i]
        for n in row:
            appearances[n].append(i)
            last_seen[n] = i
        for a, b in combinations(row, 2):
            aff_matrix.at[a, b] += 1.0; aff_matrix.at[b, a] += 1.0
            
        if i < n_draws - 1:
            next_row = matrix[i+1]
            for curr in row:
                for nxt in next_row:
                    trans_matrix.at[curr, nxt] += 1.0

    freq_series = pd.Series({n: freq_counter.get(n, 0) for n in MAIN_POOL}).sort_index()
    gaps = {n: (n_draws - 1 - last_seen.get(n, -1)) for n in MAIN_POOL}
    mean_gaps = {n: (float(np.mean(np.diff(appearances[n]))) if len(appearances[n]) > 1 else float(n_draws)) for n in MAIN_POOL}
    
    return LottoStats(freq_series, gaps, mean_gaps, aff_matrix, trans_matrix, tuple(matrix[-1]))

def generate_ticket(stats: LottoStats, mode: str, intensity: float, affinity: float) -> Tuple[int, ...]:
    f = stats.frequency.to_numpy(dtype=float) + 1.0
    g = np.array([stats.gaps[n] for n in MAIN_POOL], dtype=float)
    mg = np.array([stats.mean_gaps[n] for n in MAIN_POOL], dtype=float)
    overdue = np.where(mg > 0, g / mg, 1.0)
    
    if mode == "markov":
        w = np.zeros(len(MAIN_POOL))
        for last in stats.last_drawn: w += stats.transition_matrix.loc[last].to_numpy()
        w += 0.1
    elif mode == "hot": w = f
    elif mode == "cold": w = (1.0 / f) * (overdue ** 2)
    else: w = (f / f.max()) + (overdue / overdue.max()) # Hybrid
    
    current_w = (intensity * (w / w.sum())) + ((1.0 - intensity) * (np.ones_like(w) / len(w)))
    current_w /= current_w.sum()
    
    chosen = []
    mask = np.ones(len(MAIN_POOL), dtype=bool)
    for _ in range(MAIN_COUNT):
        probs = (current_w * mask) / (current_w * mask).sum()
        idx = np.random.choice(len(MAIN_POOL), p=probs)
        val = MAIN_POOL[idx]
        chosen.append(val)
        mask[idx] = False
        if affinity > 0:
            links = stats.affinity_matrix.loc[val].to_numpy()
            if links.max() > 0: current_w = (current_w * (1.0 + affinity * (links / links.max())))
            current_w /= current_w.sum()
            
    return tuple(sorted(chosen))

# ---------------------------------------------------------------------------
# UI MAIN
# ---------------------------------------------------------------------------

def main():
    inject_ultimate_css()
    st.title("🎰 Lotto ULTIMATE 4.0: Machine Path Analysis")
    
    df = load_and_parse_lotto_db()
    if df.empty:
        st.error("Nie znaleziono pliku L1.PDF w folderze aplikacji.")
        st.stop()
        
    stats = compute_stats(df)

    with st.sidebar:
        st.header("⚙️ Konfiguracja Rdzenia")
        
        st.markdown("<div class='expert-box'><b>Szlak Markowa:</b> Analizuje 'pamięć' bębna maszyny. Sprawdza co statystycznie wpada po kulach z ostatniego losowania.</div>", unsafe_allow_html=True)
        
        mode = st.radio("Strategia wag:", ["markov", "hybrid", "hot", "cold"], format_func=lambda x: x.upper())
        intensity = st.slider("Intensywność statystyczna", 0.0, 1.0, 0.75)
        affinity = st.slider("Siła powiązań (Affinity)", 0.0, 1.0, 0.50)
        n_tickets = st.number_input("Liczba zakładów", 1, 100, 6)
        gen_btn = st.button("🚀 GENERUJ ZESTAWY")

    tab1, tab2, tab3 = st.tabs(["🔮 Predykcje", "⬡ Szlaki Przejść", "📁 Baza Danych"])

    with tab1:
        st.write(f"✅ Zsynchronizowano **{len(df)}** losowań. Ostatnie: `{stats.last_drawn}`")
        if gen_btn or "tickets_lotto" not in st.session_state:
            st.session_state["tickets_lotto"] = [generate_ticket(stats, mode, intensity, affinity) for _ in range(n_tickets)]

        for i, t in enumerate(st.session_state["tickets_lotto"], 1):
            html = f"<div class='ticket-card'><div class='ball-container'>"
            for n in t: html += f"<div class='ball'>{n}</div>"
            html += f"</div><div style='font-size:0.8rem; color:#888'>Zakład #{i} | Profil: {mode.upper()}</div></div>"
            st.markdown(html, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Macierz Przejść (Skoki Maszyny)")
        fig = go.Figure(data=go.Heatmap(z=stats.transition_matrix.values, x=stats.transition_matrix.columns, y=stats.transition_matrix.index, colorscale="Jet"))
        fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#fff")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(df.sort_values("draw_id", ascending=False).drop(columns="order"), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
