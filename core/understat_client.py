# core/understat_client.py
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import aiohttp
from understat import Understat


# ===============================
# football-data competition code -> Understat slug
# (IMPORTANTE: underscore, NON trattini)
# ===============================
UNDERSTAT_LEAGUES = {
    "SA": "serie_a",
    "PL": "epl",
    "PD": "la_liga",
    "BL1": "bundesliga",
    "FL1": "ligue_1",
}


def normalize_name(x: str) -> str:
    """Normalizza nomi squadra per match/merge."""
    if not isinstance(x, str):
        return ""
    x = x.lower().strip()
    for t in [" fc", " cf", " calcio", " football club", " f.c.", " c.f.", " afc", " cfc"]:
        x = x.replace(t, " ")
    x = x.replace(".", " ").replace("-", " ")
    x = " ".join(x.split())
    return x


async def _fetch_understat_season_async(league_slug: str, season_start_year: int) -> list:
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        return await us.get_league_results(league_slug, season_start_year)


def _run_async(coro):
    """Esegue coroutine anche se esiste già un event loop (più robusto su vari ambienti)."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _cache_is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=ttl_hours)


def _understat_to_df(raw: list) -> pd.DataFrame:
    """
    Output standard:
      data | casa_us | trasferta_us | xg_casa | xg_trasferta
    """
    rows = []
    for m in raw or []:
        dt = m.get("datetime") or m.get("date") or ""
        try:
            d = pd.to_datetime(dt).date()
        except Exception:
            continue

        home = (m.get("h") or {}).get("title")
        away = (m.get("a") or {}).get("title")
        xg = m.get("xG") or {}

        def to_float(v):
            try:
                return float(v)
            except Exception:
                return None

        rows.append(
            {
                "data": d,
                "casa_us": home,
                "trasferta_us": away,
                "xg_casa": to_float(xg.get("h")),
                "xg_trasferta": to_float(xg.get("a")),
            }
        )

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
    ttl_hours: int = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Scarica match Understat (xG reali) per stagione.
    Cache su JSON in cache_dir (con TTL su mtime).
    """
    league_slug = UNDERSTAT_LEAGUES.get(comp_code)
    if not league_slug:
        return pd.DataFrame()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"understat_{league_slug}_{season_start_year}.json"

    # --- cache ---
    if (not force_refresh) and _cache_is_fresh(cache_file, ttl_hours=ttl_hours):
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return _understat_to_df(raw)
        except Exception:
            pass

    # --- live fetch ---
    try:
        raw = _run_async(_fetch_understat_season_async(league_slug, season_start_year))
        try:
            cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return _understat_to_df(raw)
    except Exception:
        # se Understat va giù o blocca, non crashiamo: ritorniamo vuoto
        return pd.DataFrame()


def build_understat_team_match_index(us_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match Understat -> formato LONG:
    data | squadra | avversario | xg_fatto | xg_concesso | squadra_n
    """
    if us_df is None or us_df.empty:
        return pd.DataFrame(columns=["data", "squadra", "avversario", "xg_fatto", "xg_concesso", "squadra_n"])

    rows = []
    for _, r in us_df.iterrows():
        rows.append(
            {
                "data": r["data"],
                "squadra": r["casa_us"],
                "avversario": r["trasferta_us"],
                "xg_fatto": r["xg_casa"],
                "xg_concesso": r["xg_trasferta"],
            }
        )
        rows.append(
            {
                "data": r["data"],
                "squadra": r["trasferta_us"],
                "avversario": r["casa_us"],
                "xg_fatto": r["xg_trasferta"],
                "xg_concesso": r["xg_casa"],
            }
        )

    out = pd.DataFrame(rows)
    out["data"] = pd.to_datetime(out["data"], errors="coerce")
    out["squadra_n"] = out["squadra"].map(normalize_name)
    out = out.dropna(subset=["data", "squadra_n"])
    out = out.sort_values(["squadra_n", "data"])
    return out

