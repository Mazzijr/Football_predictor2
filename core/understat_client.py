import json
import asyncio
from pathlib import Path
from datetime import date

import pandas as pd
import aiohttp
from understat import Understat

# football-data code -> understat league key
UNDERSTAT_LEAGUES = {
    "SA": "serie_a",
    "PL": "epl",
    "PD": "la_liga",
    "BL1": "bundesliga",
    "FL1": "ligue_1",
}


def normalize_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.lower().strip()
    for t in [" fc", " cf", " calcio", " football club", " f.c.", " c.f.", " afc", " cfc"]:
        x = x.replace(t, " ")
    x = x.replace(".", " ").replace("-", " ")
    return " ".join(x.split())


async def _fetch_understat_season(league_key: str, season: int):
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        return await us.get_league_results(league_key, season)


def _understat_to_df(raw: list) -> pd.DataFrame:
    rows = []
    for m in raw or []:
        dt = (m.get("datetime") or "")[:10]
        rows.append({
            "data": pd.to_datetime(dt, errors="coerce").date() if dt else None,
            "casa_us": (m.get("h") or {}).get("title"),
            "trasferta_us": (m.get("a") or {}).get("title"),
            "xg_casa": pd.to_numeric((m.get("xG") or {}).get("h"), errors="coerce"),
            "xg_trasferta": pd.to_numeric((m.get("xG") or {}).get("a"), errors="coerce"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["data", "casa_us", "trasferta_us"])
    df = df.sort_values("data")
    return df


def get_understat_matches_season(
    cache_dir: Path,
    comp_code: str,
    season_start_year: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    xG REALI Understat con cache giornaliera (file JSON per data).
    """
    league_key = UNDERSTAT_LEAGUES.get(comp_code)
    if not league_key:
        return pd.DataFrame()

    cache_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    cache_file = cache_dir / f"understat_{comp_code}_{season_start_year}_{today}.json"

    if cache_file.exists() and not force_refresh:
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return _understat_to_df(raw)
        except Exception:
            pass

    try:
        raw = asyncio.run(_fetch_understat_season(league_key, season_start_year))
        cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return _understat_to_df(raw)
    except Exception:
        return pd.DataFrame()


