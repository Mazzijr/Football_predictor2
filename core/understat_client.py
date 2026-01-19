import json
import asyncio
from pathlib import Path
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import aiohttp
from understat import Understat


# ===============================
# Mappa campionati football-data -> Understat (nomi CORRETTI per la libreria)
# ===============================
UNDERSTAT_LEAGUES = {
    "SA": "Serie A",
    "PL": "EPL",
    "PD": "La liga",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
}


def normalize_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.lower().strip()
    for t in [" fc", " cf", " calcio", " football club", " f.c.", " c.f.", " afc", " cfc"]:
        x = x.replace(t, " ")
    x = x.replace(".", " ").replace("-", " ")
    x = " ".join(x.split())
    return x


def _cache_is_fresh(path: Path, ttl_hours: int = 24) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=ttl_hours)


async def _fetch_understat_season_async(league: str, season: int) -> list:
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        return await us.get_league_results(league, season)


def _run_coro_cloud_safe(coro):
    """
    Streamlit Cloud a volte ha già un event loop attivo -> asyncio.run() può fallire.
    Qui gestiamo entrambi i casi in modo robusto.
    """
    try:
        # se NON c'è un loop attivo, asyncio.run va bene
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    if not has_running_loop:
        return asyncio.run(coro)

    # Se c'è un loop già attivo, eseguiamo in thread separato
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: asyncio.run(coro))
        return fut.result()


def _understat_to_df(raw: list) -> pd.DataFrame:
    """
    Converte JSON Understat in DataFrame standard:
    data | casa_us | trasferta_us | xg_casa | xg_trasferta
    """
    rows = []
    for m in raw or []:
        dt = (m.get("datetime") or "")[:10]
        rows.append(
            {
                "data": dt,
                "casa_us": (m.get("h") or {}).get("title"),
                "trasferta_us": (m.get("a") or {}).get("title"),
                "xg_casa": float((m.get("xG") or {}).get("h", 0) or 0),
                "xg_trasferta": float((m.get("xG") or {}).get("a", 0) or 0),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
        df = df.dropna(subset=["data", "casa_us", "trasferta_us"])
    return df


def get_understat_matches_season(
    cache_dir: Path,
    comp_code: str,
    season_start_year: int,
    ttl_hours: int = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Scarica i match Understat (xG reali) per una stagione.
    Cache giornaliera su file JSON.
    """
    league = UNDERSTAT_LEAGUES.get(comp_code)
    if not league:
        return pd.DataFrame()

    cache_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    cache_file = cache_dir / f"understat_{comp_code}_{season_start_year}_{today}.json"

    # cache
    if (not force_refresh) and _cache_is_fresh(cache_file, ttl_hours=ttl_hours):
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return _understat_to_df(raw)
        except Exception:
            pass

    # fetch live
    try:
        raw = _run_coro_cloud_safe(_fetch_understat_season_async(league, season_start_year))
        cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return _understat_to_df(raw)
    except Exception:
        return pd.DataFrame()


def build_understat_team_match_index(us_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma i match Understat in formato LONG:
    una riga per squadra per match, con:
    data, team, opponent, xg_for, xg_against
    """
    rows = []
    if us_df is None or us_df.empty:
        return pd.DataFrame(columns=["data", "team", "opponent", "xg_for", "xg_against"])

    for _, r in us_df.iterrows():
        rows.append({
            "data": r["data"],
            "team": r["casa_us"],
            "opponent": r["trasferta_us"],
            "xg_for": r["xg_casa"],
            "xg_against": r["xg_trasferta"],
        })
        rows.append({
            "data": r["data"],
            "team": r["trasferta_us"],
            "opponent": r["casa_us"],
            "xg_for": r["xg_trasferta"],
            "xg_against": r["xg_casa"],
        })

    df_long = pd.DataFrame(rows)
    df_long["data"] = pd.to_datetime(df_long["data"], errors="coerce")
    df_long["team_n"] = df_long["team"].map(normalize_name)
    df_long = df_long.sort_values(["team_n", "data"])
    return df_long



