# core/understat_client.py

import json
import asyncio
from pathlib import Path
from datetime import date

import pandas as pd
import aiohttp
from understat import Understat


# =========================================================
# Mappa campionati football-data.org -> slug Understat
# =========================================================
UNDERSTAT_LEAGUES = {
    "SA": "serie_a",
    "PL": "epl",
    "PD": "la_liga",
    "BL1": "bundesliga",
    "FL1": "ligue_1",
}


# =========================================================
# Normalizzazione nomi (serve per join/mapping)
# =========================================================
def normalize_name(x: str) -> str:
    if x is None:
        return ""
    x = str(x).lower().strip()

    # pulizie comuni
    x = x.replace(".", " ")
    x = x.replace("-", " ")
    for t in [
        " fc",
        " cf",
        " calcio",
        " football club",
        " f c",
        " c f",
        " afc",
        " a f c",
    ]:
        x = x.replace(t, " ")

    x = " ".join(x.split())
    return x


# =========================================================
# Fetch Understat (async) -> lista di match
# =========================================================
async def _fetch_understat_season(league_slug: str, season_start_year: int):
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        data = await us.get_league_results(league_slug, season_start_year)
        return data


def _run_async(coro):
    """
    Esegue una coroutine in modo compatibile anche se esiste già un event loop.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# =========================================================
# Conversione raw Understat -> DataFrame standard
# =========================================================
def _understat_to_df(raw: list) -> pd.DataFrame:
    """
    Output columns:
      date | home | away | xg_home | xg_away
    """
    rows = []
    for m in raw or []:
        dt = (m.get("datetime") or "")[:10]  # YYYY-MM-DD
        home = (m.get("h") or {}).get("title")
        away = (m.get("a") or {}).get("title")

        xg = m.get("xG") or {}
        xg_h = xg.get("h", 0)
        xg_a = xg.get("a", 0)

        try:
            xg_h = float(xg_h)
        except Exception:
            xg_h = None

        try:
            xg_a = float(xg_a)
        except Exception:
            xg_a = None

        rows.append(
            {
                "date": dt,
                "home": home,
                "away": away,
                "xg_home": xg_h,
                "xg_away": xg_a,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        df = df.dropna(subset=["date", "home", "away"])

    return df


# =========================================================
# API principale usata da app.py
# =========================================================
def get_understat_matches_season(
    cache_dir: Path,
    comp_code: str,
    season_start_year: int,
    ttl_hours: int = 24,
) -> pd.DataFrame:
    """
    Scarica i match Understat (xG reali) per una stagione.
    Cache su JSON giornaliero:
      data/cache/understat_{comp_code}_{season}_{YYYY-MM-DD}.json
    """
    league_slug = UNDERSTAT_LEAGUES.get(comp_code)
    if not league_slug:
        return pd.DataFrame()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    cache_file = cache_dir / f"understat_{comp_code}_{season_start_year}_{today}.json"

    # --- cache ---
    if cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return _understat_to_df(raw)
        except Exception:
            pass

    # --- live fetch ---
    try:
        raw = _run_async(_fetch_understat_season(league_slug, season_start_year))
        cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return _understat_to_df(raw)
    except Exception:
        # se Understat down o blocco, torna vuoto senza crashare l'app
        return pd.DataFrame()


# =========================================================
# Long format: una riga per squadra per match (per rolling)
# =========================================================
def build_understat_team_match_index(us_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma i match Understat in formato LONG:
      date, team, opponent, xg_for, xg_against
    """
    if us_df is None or us_df.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "xg_for", "xg_against"])

    rows = []
    for _, r in us_df.iterrows():
        # home team
        rows.append(
            {
                "date": r["date"],
                "team": r["home"],
                "opponent": r["away"],
                "xg_for": r["xg_home"],
                "xg_against": r["xg_away"],
            }
        )
        # away team
        rows.append(
            {
                "date": r["date"],
                "team": r["away"],
                "opponent": r["home"],
                "xg_for": r["xg_away"],
                "xg_against": r["xg_home"],
            }
        )

    df_long = pd.DataFrame(rows)
    df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")
    df_long = df_long.dropna(subset=["date", "team", "opponent"])

    return df_long

