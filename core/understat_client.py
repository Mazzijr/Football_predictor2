# core/understat_client.py

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

# ==============================
# NORMALIZZAZIONE NOMI SQUADRE
# ==============================
def normalize_name(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.lower().strip()
    x = re.sub(r"\s+fc$", "", x)
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x


# ==============================
# MAPPA CAMPIONATI
# ==============================
UNDERSTAT_LEAGUES = {
    "SA": "serie-a",
    "PL": "epl",
    "PD": "la-liga",
    "BL1": "bundesliga",
    "FL1": "ligue-1",
}


# ==============================
# DOWNLOAD + CACHE MATCHES
# ==============================
def get_understat_matches_season(
    cache_dir: Path,
    comp_code: str,
    season: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Scarica i match Understat di una stagione (xG reali).
    Usa cache JSON per Streamlit Cloud.
    """

    league = UNDERSTAT_LEAGUES.get(comp_code)
    if not league:
        return pd.DataFrame()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"understat_{comp_code}_{season}.json"

    # --- CACHE ---
    if cache_file.exists() and not force_refresh:
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return _to_df(raw)
        except Exception:
            pass

    # --- DOWNLOAD ---
    url = f"https://understat.com/league/{league}/{season}"

    try:
        html = requests.get(url, timeout=20).text
    except Exception:
        return pd.DataFrame()

    # Estrazione JSON embedded
    marker = "var matchesData = JSON.parse('"
    start = html.find(marker)
    if start == -1:
        return pd.DataFrame()

    start += len(marker)
    end = html.find("');", start)
    if end == -1:
        return pd.DataFrame()

    json_text = html[start:end]
    json_text = bytes(json_text, "utf-8").decode("unicode_escape")

    try:
        raw = json.loads(json_text)
    except Exception:
        return pd.DataFrame()

    cache_file.write_text(json.dumps(raw), encoding="utf-8")
    return _to_df(raw)


# ==============================
# CONVERSIONE JSON → DATAFRAME
# ==============================
def _to_df(raw: Dict) -> pd.DataFrame:
    rows = []

    for m in raw.values():
        rows.append(
            {
                "match_id": int(m["id"]),
                "date": m["datetime"][:10],
                "home": normalize_name(m["h"]["title"]),
                "away": normalize_name(m["a"]["title"]),
                "xg_home": float(m["xG"]["h"]),
                "xg_away": float(m["xG"]["a"]),
                "goals_home": int(m["goals"]["h"]),
                "goals_away": int(m["goals"]["a"]),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ==============================
# INDICE MATCH xG (CASA/TRASFERTA)
# ==============================
def build_understat_team_match_index(df_us: pd.DataFrame) -> Dict:
    """
    Ritorna:
    {(team, match_id): {"xg_for": x, "xg_against": y}}
    """

    index = {}

    for _, r in df_us.iterrows():
        index[(r["home"], r["match_id"])] = {
            "xg_for": r["xg_home"],
            "xg_against": r["xg_away"],
        }
        index[(r["away"], r["match_id"])] = {
            "xg_for": r["xg_away"],
            "xg_against": r["xg_home"],
        }

    return index

