import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

from core.understat_client import (
    get_understat_matches_season,
    build_understat_team_match_index
)


# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Football Predictor 2", layout="wide")

COMPETITIONS_FD = {
    "Serie A": "SA",
    "Premier League": "PL",
    "LaLiga": "PD",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
}
DEFAULT_SEASON = 2025
HOME_ADV_FACTOR = 1.07

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAP_FILE = Path("team_name_map.json")

API_KEY = st.secrets.get("API_KEY", "")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

# -------------------------------
# UTIL: standardizza colonne partite
# -------------------------------
def standardizza_colonne_partite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantisce presenza colonne:
      casa_fd / trasferta_fd
    accettando nomi alternativi (Casa, Trasferta, home, away, ecc).
    """
    df = df.copy()

    rename_map = {
        "Casa": "casa_fd",
        "Trasferta": "trasferta_fd",
        "home_fd": "casa_fd",
        "away_fd": "trasferta_fd",
        "Home": "casa_fd",
        "Away": "trasferta_fd",
        "home": "casa_fd",
        "away": "trasferta_fd",
        "homeTeam": "casa_fd",
        "awayTeam": "trasferta_fd",
    }

    for col_src, col_dst in rename_map.items():
        if col_src in df.columns and col_dst not in df.columns:
            df[col_dst] = df[col_src]

    # se ancora mancano, crea colonne vuote per evitare KeyError
    if "casa_fd" not in df.columns:
        df["casa_fd"] = np.nan
    if "trasferta_fd" not in df.columns:
        df["trasferta_fd"] = np.nan

    return df


# -------------------------------
# UTIL: mapping squadre
# -------------------------------
def _default_team_map():
    return {"SA": {}, "PL": {}, "PD": {}, "BL1": {}, "FL1": {}}

def load_team_map() -> dict:
    if MAP_FILE.exists():
        try:
            data = json.loads(MAP_FILE.read_text(encoding="utf-8"))
            base = _default_team_map()
            for k, v in data.items():
                if k in base and isinstance(v, dict):
                    base[k].update(v)
            return base
        except Exception:
            return _default_team_map()
    return _default_team_map()

def save_team_map(team_map: dict):
    MAP_FILE.write_text(json.dumps(team_map, ensure_ascii=False, indent=2), encoding="utf-8")

def apply_team_map(comp_code: str, name_fd: str, team_map: dict) -> str:
    if not isinstance(name_fd, str):
        return ""
    if comp_code in team_map and name_fd in team_map[comp_code]:
        return team_map[comp_code][name_fd]
    return name_fd


# -------------------------------
# FOOTBALL-DATA.ORG
# -------------------------------
def fd_get(url: str, params: dict | None = None) -> dict:
    if not API_KEY:
        return {}
    headers = {"X-Auth-Token": API_KEY}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return {}
    return r.json()

@st.cache_data(ttl=60 * 60 * 6)
def get_fd_matches(comp_code: str, season: int, status: str) -> pd.DataFrame:
    url = f"https://api.football-data.org/v4/competitions/{comp_code}/matches"
    data = fd_get(url, params={"season": season, "status": status})
    rows = []
    for m in data.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {}) or {}
        rows.append({
            "data": (m.get("utcDate") or "")[:10],
            "giornata": m.get("matchday"),
            "casa_fd": (m.get("homeTeam") or {}).get("name"),
            "trasferta_fd": (m.get("awayTeam") or {}).get("name"),
            "gol_casa": ft.get("home"),
            "gol_trasferta": ft.get("away"),
            "stato": m.get("status"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
    df = standardizza_colonne_partite(df)
    return df

@st.cache_data(ttl=60 * 60 * 6)
def get_fd_standings(comp_code: str, season: int) -> pd.DataFrame:
    url = f"https://api.football-data.org/v4/competitions/{comp_code}/standings"
    data = fd_get(url, params={"season": season})
    standings = data.get("standings", [])
    table = None
    for s in standings:
        if s.get("type") == "TOTAL":
            table = s.get("table", [])
            break
    if table is None and standings:
        table = standings[0].get("table", [])

    rows = []
    for r in table or []:
        rows.append({
            "posizione": r.get("position"),
            "squadra_fd": (r.get("team") or {}).get("name"),
            "punti": r.get("points"),
            "gf": r.get("goalsFor"),
            "ga": r.get("goalsAgainst"),
            "diff": r.get("goalDifference"),
        })
    return pd.DataFrame(rows)


# -------------------------------
# Rolling xG/xGA da Understat
# -------------------------------
def build_team_long_us(df_us: pd.DataFrame) -> pd.DataFrame:
    if df_us.empty:
        return pd.DataFrame(columns=["data", "squadra", "xg_fatto", "xg_concesso", "squadra_n"])

    home = df_us.rename(columns={
        "casa_us": "squadra",
        "xg_casa": "xg_fatto",
        "xg_trasferta": "xg_concesso",
    })[["data", "squadra", "xg_fatto", "xg_concesso"]]

    away = df_us.rename(columns={
        "trasferta_us": "squadra",
        "xg_trasferta": "xg_fatto",
        "xg_casa": "xg_concesso",
    })[["data", "squadra", "xg_fatto", "xg_concesso"]]

    out = pd.concat([home, away], ignore_index=True)
    out["squadra_n"] = out["squadra"].map(normalize_name)
    out = out.sort_values(["squadra_n", "data"])
    return out

def last_rolling_team_stats(df_us: pd.DataFrame, n: int) -> pd.DataFrame:
    long_df = build_team_long_us(df_us)
    if long_df.empty:
        return pd.DataFrame(columns=["squadra_n", "xg_fatto_N", "xg_concesso_N"]).set_index("squadra_n")

    long_df["xg_fatto_N"] = long_df.groupby("squadra_n")["xg_fatto"].transform(
        lambda s: s.rolling(n, min_periods=1).mean()
    )
    long_df["xg_concesso_N"] = long_df.groupby("squadra_n")["xg_concesso"].transform(
        lambda s: s.rolling(n, min_periods=1).mean()
    )

    last = long_df.groupby("squadra_n").tail(1).set_index("squadra_n")[["xg_fatto_N", "xg_concesso_N"]]
    return last


# -------------------------------
# Merge xG match-level + posizioni + rolling
# -------------------------------
def merge_understat_xg(df_fd: pd.DataFrame, df_us: pd.DataFrame, comp_code: str, team_map: dict):
    out = df_fd.copy()
    out = standardizza_colonne_partite(out)

    out["casa"] = out["casa_fd"].apply(lambda x: apply_team_map(comp_code, x, team_map))
    out["trasferta"] = out["trasferta_fd"].apply(lambda x: apply_team_map(comp_code, x, team_map))

    if df_us.empty or out.empty:
        out["xg_casa"] = np.nan
        out["xg_trasferta"] = np.nan
        return out, 0.0

    us2 = df_us.copy()
    us2["casa_n"] = us2["casa_us"].map(normalize_name)
    us2["trasferta_n"] = us2["trasferta_us"].map(normalize_name)
    us2["_key"] = us2["data"].astype(str) + "||" + us2["casa_n"] + "||" + us2["trasferta_n"]

    out["casa_n"] = out["casa"].map(normalize_name)
    out["trasferta_n"] = out["trasferta"].map(normalize_name)
    out["_key"] = out["data"].astype(str) + "||" + out["casa_n"] + "||" + out["trasferta_n"]

    us_map = us2.set_index("_key")[["xg_casa", "xg_trasferta"]]
    out = out.join(us_map, on="_key")

    out = out.drop(columns=["casa_n", "trasferta_n", "_key"])
    coverage = float(out["xg_casa"].notna().mean() * 100.0) if len(out) else 0.0
    return out, coverage

def add_positions(df: pd.DataFrame, df_std: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty or df_std.empty:
        df["posizione_casa"] = np.nan
        df["posizione_trasferta"] = np.nan
        return df
    pos_map = df_std.set_index("squadra_fd")["posizione"].to_dict()
    df["posizione_casa"] = df["casa_fd"].map(pos_map)
    df["posizione_trasferta"] = df["trasferta_fd"].map(pos_map)
    return df

def add_rolling_stats(df_matches: pd.DataFrame, df_us: pd.DataFrame, n: int) -> pd.DataFrame:
    out = df_matches.copy()
    stats = last_rolling_team_stats(df_us, n=n)

    out["casa_n"] = out["casa"].map(normalize_name)
    out["trasferta_n"] = out["trasferta"].map(normalize_name)

    out["xG_fatti_casa_ultN"] = out["casa_n"].map(stats["xg_fatto_N"])
    out["xG_concessi_casa_ultN"] = out["casa_n"].map(stats["xg_concesso_N"])
    out["xG_fatti_trasferta_ultN"] = out["trasferta_n"].map(stats["xg_fatto_N"])
    out["xG_concessi_trasferta_ultN"] = out["trasferta_n"].map(stats["xg_concesso_N"])

    return out.drop(columns=["casa_n", "trasferta_n"])


# -------------------------------
# Quote (CSV opzionale)
# -------------------------------
def parse_odds_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_data = pick("data", "date")
    c_home = pick("casa", "home")
    c_away = pick("trasferta", "away")
    c_1 = pick("quota_1", "odd_1", "odds_1", "1")
    c_x = pick("quota_x", "odd_x", "odds_x", "x")
    c_2 = pick("quota_2", "odd_2", "odds_2", "2")
    c_u = pick("quota_under25", "odd_under25", "under_2.5", "under25")
    c_o = pick("quota_over25", "odd_over25", "over_2.5", "over25")

    if any(x is None for x in [c_data, c_home, c_away]):
        raise ValueError("CSV quote: servono almeno data/casa/trasferta (anche con nomi equivalenti).")

    out = pd.DataFrame({
        "data": pd.to_datetime(df[c_data], errors="coerce").dt.date,
        "casa_quote": df[c_home].astype(str),
        "trasferta_quote": df[c_away].astype(str),
    })

    if c_1: out["quota_1"] = pd.to_numeric(df[c_1], errors="coerce")
    if c_x: out["quota_x"] = pd.to_numeric(df[c_x], errors="coerce")
    if c_2: out["quota_2"] = pd.to_numeric(df[c_2], errors="coerce")
    if c_u: out["quota_under25"] = pd.to_numeric(df[c_u], errors="coerce")
    if c_o: out["quota_over25"] = pd.to_numeric(df[c_o], errors="coerce")

    out["casa_n"] = out["casa_quote"].map(normalize_name)
    out["trasferta_n"] = out["trasferta_quote"].map(normalize_name)
    out["_key"] = out["data"].astype(str) + "||" + out["casa_n"] + "||" + out["trasferta_n"]
    return out.drop(columns=["casa_n", "trasferta_n"])

def merge_odds(df_matches: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    if df_matches.empty or df_odds.empty:
        return df_matches
    out = df_matches.copy()
    out["casa_n"] = out["casa"].map(normalize_name)
    out["trasferta_n"] = out["trasferta"].map(normalize_name)
    out["_key"] = out["data"].astype(str) + "||" + out["casa_n"] + "||" + out["trasferta_n"]

    odds_map = df_odds.set_index("_key")
    cols_take = [c for c in ["quota_1","quota_x","quota_2","quota_under25","quota_over25"] if c in odds_map.columns]
    out = out.join(odds_map[cols_take], on="_key")
    return out.drop(columns=["casa_n","trasferta_n","_key"])


# -------------------------------
# Poisson + value
# -------------------------------
def poisson_1x2_probs(lam_home: float, lam_away: float, max_goals: int = 10):
    if not np.isfinite(lam_home) or not np.isfinite(lam_away) or lam_home <= 0 or lam_away <= 0:
        return np.nan, np.nan, np.nan

    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)

    fact_h = np.array([math.factorial(int(i)) for i in hg], dtype=float)
    fact_a = np.array([math.factorial(int(i)) for i in ag], dtype=float)

    p_h = np.exp(-lam_home) * (lam_home ** hg) / fact_h
    p_a = np.exp(-lam_away) * (lam_away ** ag) / fact_a

    mat = np.outer(p_h, p_a)
    p_home = np.tril(mat, -1).sum()
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()
    return float(p_home), float(p_draw), float(p_away)

def poisson_under_over_25(lam_home: float, lam_away: float):
    if not np.isfinite(lam_home) or not np.isfinite(lam_away) or lam_home <= 0 or lam_away <= 0:
        return np.nan, np.nan
    lam = lam_home + lam_away
    k = np.arange(0, 3)  # 0,1,2
    fact_k = np.array([math.factorial(int(i)) for i in k], dtype=float)
    p_le_2 = np.exp(-lam) * np.sum((lam ** k) / fact_k)
    p_over = 1.0 - p_le_2
    p_under = 1.0 - p_over
    return float(p_under), float(p_over)

def build_lambdas(row: pd.Series):
    hxgf = row.get("xG_fatti_casa_ultN", np.nan)
    hxga = row.get("xG_concessi_casa_ultN", np.nan)
    axgf = row.get("xG_fatti_trasferta_ultN", np.nan)
    axga = row.get("xG_concessi_trasferta_ultN", np.nan)

    lam_home = float(((hxgf + axga) / 2.0) * HOME_ADV_FACTOR) if np.isfinite(hxgf) and np.isfinite(axga) else np.nan
    lam_away = float(((axgf + hxga) / 2.0)) if np.isfinite(axgf) and np.isfinite(hxga) else np.nan
    return lam_home, lam_away

def implied_prob(odd: float) -> float:
    if odd is None or not np.isfinite(odd) or odd <= 1e-9:
        return np.nan
    return 1.0 / float(odd)

def pct(x: float) -> float:
    return float(x * 100.0) if np.isfinite(x) else np.nan

def pick_best_bet(row: pd.Series, min_prob: float, min_edge: float):
    candidates = []

    for label, pcol, ocol in [
        ("1", "p_1", "quota_1"),
        ("X", "p_X", "quota_x"),
        ("2", "p_2", "quota_2"),
        ("Under 2.5", "p_under25", "quota_under25"),
        ("Over 2.5", "p_over25", "quota_over25"),
    ]:
        p = row.get(pcol, np.nan)
        odd = row.get(ocol, np.nan)
        imp = implied_prob(odd)
        edge = (p - imp) if np.isfinite(p) and np.isfinite(imp) else np.nan
        candidates.append((label, p, edge))

    good = [c for c in candidates if np.isfinite(c[1]) and np.isfinite(c[2]) and c[1] >= min_prob and c[2] >= min_edge]
    if good:
        good.sort(key=lambda x: x[2], reverse=True)
        return good[0][0], float(good[0][1]), float(good[0][2])

    cand_p = [c for c in candidates if np.isfinite(c[1])]
    if not cand_p:
        return "N/D", np.nan, np.nan
    cand_p.sort(key=lambda x: x[1], reverse=True)
    return cand_p[0][0], float(cand_p[0][1]), np.nan

def human_motivation(row: pd.Series) -> str:
    casa = row.get("casa_fd", "")
    trasf = row.get("trasferta_fd", "")
    lam_h = row.get("gol_attesi_casa", np.nan)
    lam_a = row.get("gol_attesi_trasferta", np.nan)
    pos_h = row.get("posizione_casa", np.nan)
    pos_a = row.get("posizione_trasferta", np.nan)

    bits = []
    if np.isfinite(lam_h) and np.isfinite(lam_a):
        if lam_h > lam_a + 0.35:
            bits.append(f"{casa} crea mediamente più occasioni (xG) rispetto a {trasf}.")
        elif lam_a > lam_h + 0.35:
            bits.append(f"{trasf} sembra avere il profilo offensivo migliore nelle ultime partite (xG).")
        else:
            bits.append("Le due squadre risultano piuttosto equilibrate sugli xG recenti.")

        tot = lam_h + lam_a
        if tot >= 2.8:
            bits.append("Ritmo atteso alto → più favorevole a Over 2.5.")
        elif tot <= 2.2:
            bits.append("Ritmo atteso basso → più favorevole a Under 2.5.")

    if np.isfinite(pos_h) and np.isfinite(pos_a):
        if pos_h < pos_a:
            bits.append(f"In classifica {casa} è sopra ({int(pos_h)}ª vs {int(pos_a)}ª): vittoria “da favorita” pesa meno, ma indica solidità.")
        elif pos_a < pos_h:
            bits.append(f"In classifica {trasf} è sopra ({int(pos_a)}ª vs {int(pos_h)}ª): possibile sorpresa in trasferta.")

    return " ".join(bits) if bits else "Motivazione non disponibile (dati insufficienti)."


# -------------------------------
# UI: AUTH
# -------------------------------
st.title("⚽ Football Predictor 2")

with st.expander("🔐 Accesso", expanded=True):
    pwd = st.text_input("Password", type="password", value="")
    if APP_PASSWORD and pwd != APP_PASSWORD:
        st.stop()
    st.success("Accesso consentito ✅" if APP_PASSWORD else "Accesso libero (APP_PASSWORD non impostata).")


# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Impostazioni")

campionato = st.sidebar.selectbox("Campionato", list(COMPETITIONS_FD.keys()))
comp_code = COMPETITIONS_FD[campionato]
season = st.sidebar.number_input("Anno inizio stagione (es. 2025 per 2025/26)", 2014, 2030, DEFAULT_SEASON, step=1)
status_view = st.sidebar.selectbox("Tipo partite", ["SCHEDULED", "FINISHED"], index=0)

n_roll = st.sidebar.slider("Media xG/xGA ultime N", 3, 15, 6)

use_understat = st.sidebar.checkbox("Usa Understat LIVE (xG reali)", value=True)
force_refresh = st.sidebar.button("Forza refresh Understat (oggi)")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Soglie 'Miglior giocata'")
min_prob_pct = st.sidebar.slider("Probabilità minima (%)", 30, 80, 45)
min_edge_pct = st.sidebar.slider("Value minimo (%)", 0, 20, 5)
min_prob = min_prob_pct / 100.0
min_edge = min_edge_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("💸 Quote (opzionale)")
uploaded_odds = st.sidebar.file_uploader("Carica CSV quote", type=["csv"])

team_map = load_team_map()
with st.sidebar.expander("🧩 Mapping nomi squadre", expanded=False):
    st.write("Esempio: 'Inter Milan' → 'Inter'")
    fd_name = st.text_input("Nome football-data")
    us_name = st.text_input("Nome Understat/Quote")
    cA, cB = st.columns(2)
    with cA:
        if st.button("Salva mapping"):
            if fd_name.strip() and us_name.strip():
                team_map.setdefault(comp_code, {})
                team_map[comp_code][fd_name.strip()] = us_name.strip()
                save_team_map(team_map)
                st.success("Salvato ✅ (ricarica pagina)")
            else:
                st.error("Inserisci entrambi i campi.")
    with cB:
        if st.button("Reset mapping campionato"):
            team_map[comp_code] = {}
            save_team_map(team_map)
            st.warning("Reset fatto.")
    st.json(team_map.get(comp_code, {}))


# -------------------------------
# LOAD DATI
# -------------------------------
if not API_KEY:
    st.error("❌ API_KEY mancante in .streamlit/secrets.toml")
    st.stop()

df_fd = get_fd_matches(comp_code, season, status_view)
df_std = get_fd_standings(comp_code, season)

df_us = pd.DataFrame()
if use_understat:
    df_us = get_understat_matches_season(CACHE_DIR, comp_code, season, force_refresh=force_refresh)

df, coverage = merge_understat_xg(df_fd, df_us, comp_code, team_map)
df = add_positions(df, df_std)
df = add_rolling_stats(df, df_us, n=n_roll)

if uploaded_odds is not None:
    try:
        df_odds = parse_odds_csv(uploaded_odds)
        df = merge_odds(df, df_odds)
        st.sidebar.success("Quote caricate ✅")
    except Exception as e:
        st.sidebar.error(f"Errore CSV quote: {e}")


# -------------------------------
# CALCOLI
# -------------------------------
lam = df.apply(lambda r: build_lambdas(r), axis=1, result_type="expand")
df["gol_attesi_casa"] = lam[0]
df["gol_attesi_trasferta"] = lam[1]

p1x2 = df.apply(lambda r: poisson_1x2_probs(r["gol_attesi_casa"], r["gol_attesi_trasferta"]), axis=1, result_type="expand")
df["p_1"] = p1x2[0]
df["p_X"] = p1x2[1]
df["p_2"] = p1x2[2]

pou = df.apply(lambda r: poisson_under_over_25(r["gol_attesi_casa"], r["gol_attesi_trasferta"]), axis=1, result_type="expand")
df["p_under25"] = pou[0]
df["p_over25"] = pou[1]

def pred_secca(row):
    best_1x2 = "N/D"
    if np.isfinite(row.get("p_1")) and np.isfinite(row.get("p_X")) and np.isfinite(row.get("p_2")):
        best_1x2 = max([("1", row["p_1"]), ("X", row["p_X"]), ("2", row["p_2"])], key=lambda x: x[1])[0]
    best_ou = "N/D"
    if np.isfinite(row.get("p_under25")) and np.isfinite(row.get("p_over25")):
        best_ou = max([("Under 2.5", row["p_under25"]), ("Over 2.5", row["p_over25"])], key=lambda x: x[1])[0]
    return f"{best_1x2} + {best_ou}"

df["predizione_secca"] = df.apply(pred_secca, axis=1)

best = df.apply(lambda r: pick_best_bet(r, min_prob=min_prob, min_edge=min_edge), axis=1, result_type="expand")
df["miglior_giocata"] = best[0]
df["prob_miglior_giocata"] = best[1]
df["value_miglior_giocata"] = best[2]

df["motivazione"] = df.apply(human_motivation, axis=1)


# -------------------------------
# HEADER
# -------------------------------
st.subheader(f"{campionato} — {('PROGRAMMATE' if status_view=='SCHEDULED' else 'FINITE')} — {season}/{season+1}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Partite", f"{len(df)}")
c2.metric("Match Understat (stagione)", f"{len(df_us)}")
c3.metric("Copertura xG partita", f"{coverage:.1f}%")
c4.metric("Rolling", f"Ultime {n_roll}")

st.caption("⚠️ Stima statistica: xG reali Understat (ultime N) + modello Poisson. Non è certezza.")


# -------------------------------
# TABELLA (con %)
# -------------------------------
df_show = df.copy()
for c in ["p_1","p_X","p_2","p_under25","p_over25","prob_miglior_giocata"]:
    if c in df_show.columns:
        df_show[c] = df_show[c].apply(pct)
if "value_miglior_giocata" in df_show.columns:
    df_show["value_miglior_giocata"] = df_show["value_miglior_giocata"].apply(pct)

cols = [
    "data","giornata","casa_fd","trasferta_fd",
    "xg_casa","xg_trasferta",
    "xG_fatti_casa_ultN","xG_concessi_casa_ultN","xG_fatti_trasferta_ultN","xG_concessi_trasferta_ultN",
    "posizione_casa","posizione_trasferta",
    "gol_attesi_casa","gol_attesi_trasferta",
    "p_1","p_X","p_2","p_under25","p_over25",
    "predizione_secca",
    "quota_1","quota_x","quota_2","quota_under25","quota_over25",
    "miglior_giocata","prob_miglior_giocata","value_miglior_giocata",
    "motivazione",
]
cols = [c for c in cols if c in df_show.columns]

rename = {
    "data": "Data",
    "giornata": "Giornata",
    "casa_fd": "Casa",
    "trasferta_fd": "Trasferta",
    "xg_casa": "xG (partita) Casa",
    "xg_trasferta": "xG (partita) Trasferta",
    "xG_fatti_casa_ultN": f"xG fatti Casa (ultime {n_roll})",
    "xG_concessi_casa_ultN": f"xG concessi Casa (ultime {n_roll})",
    "xG_fatti_trasferta_ultN": f"xG fatti Trasferta (ultime {n_roll})",
    "xG_concessi_trasferta_ultN": f"xG concessi Trasferta (ultime {n_roll})",
    "posizione_casa": "Posizione Casa",
    "posizione_trasferta": "Posizione Trasferta",
    "gol_attesi_casa": "Gol attesi Casa (λ)",
    "gol_attesi_trasferta": "Gol attesi Trasferta (λ)",
    "p_1": "Prob 1 (%)",
    "p_X": "Prob X (%)",
    "p_2": "Prob 2 (%)",
    "p_under25": "Prob Under 2.5 (%)",
    "p_over25": "Prob Over 2.5 (%)",
    "predizione_secca": "Predizione secca",
    "quota_1": "Quota 1",
    "quota_x": "Quota X",
    "quota_2": "Quota 2",
    "quota_under25": "Quota Under 2.5",
    "quota_over25": "Quota Over 2.5",
    "miglior_giocata": "Miglior giocata",
    "prob_miglior_giocata": "Probabilità miglior giocata (%)",
    "value_miglior_giocata": "Value stimato (%)",
    "motivazione": "Motivazione",
}

st.dataframe(df_show[cols].rename(columns=rename), use_container_width=True)

with st.expander("📌 Classifica (TOTALE)"):
    st.dataframe(df_std, use_container_width=True)

with st.expander("🧩 Debug: partite senza xG (mismatch nomi)"):
    if df.empty:
        st.info("Nessun match.")
    else:
        miss = df[df["xg_casa"].isna()][["data","casa_fd","trasferta_fd","casa","trasferta"]].head(200)
        if miss.empty:
            st.success("Coverage 100%")
        else:
            st.warning("Match senza xG: serve mapping nomi squadra.")
            st.dataframe(miss, use_container_width=True)

st.markdown("---")
csv_bytes = df_show[cols].rename(columns=rename).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Scarica tabella (CSV)", data=csv_bytes, file_name="football_predictor2_output.csv", mime="text/csv")



