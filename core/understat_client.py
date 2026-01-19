import json
import asyncio
from pathlib import Path
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import aiohttp
from understat import Understat


# --- DUE MAPPE LEGA: alcune versioni vogliono slug, altre titolo ---
UNDERSTAT_LEAGUES_SLUG = {
    "SA": "serie_a",
    "PL": "epl",
    "PD": "la_liga",
    "BL1": "bundesliga",
    "FL1": "ligue_1",
}

UNDERSTAT_LEAGUES_TITLE = {
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
    return " ".join(x.split())


def _cache_is_fresh(path: Path, ttl_hours: int = 24) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=ttl_hours)


def _run_coro_cloud_safe(coro):
    """Gestisce il caso in cui Streamlit abbia già un event loop attivo."""
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: asyncio.run(coro))
        return fut.result()


def _understat_to_df(raw: list) -> pd.DataFrame:
    rows = []
    for m in raw or []:
        dt = (m.get("datetime") or "")[:10]
        rows.append(
            {
                "data": dt,
                "casa_us": (m.get("h") or {}).get("title"),
                "trasferta_us": (m.get("a") or {}).get("title"),
                "xg_casa": float(((m.get("xG") or {}).get("h", 0)) or 0),
                "xg_trasferta": float(((m.get("xG") or {}).get("a", 0)) or 0),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
        df = df.dropna(subset=["data", "casa_us", "trasferta_us"])
    return df


async def _fetch_understat_async(league: str, season: int) -> list:
    # Headers “da browser” per evitare risposta vuota su Cloud
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }

    timeout = aiohttp.ClientTimeout(total=40)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        us = Understat(session)
        return await us.get_league_results(league, season)


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

    # prova prima slug, poi title
    candidates = []
    if comp_code in UNDERSTAT_LEAGUES_SLUG:
        candidates.append(UNDERSTAT_LEAGUES_SLUG[comp_code])
    if comp_code in UNDERSTAT_LEAGUES_TITLE:
        candidates.append(UNDERSTAT_LEAGUES_TITLE[comp_code])

    last_err = None
    for league in candidates:
        try:
            raw = _run_coro_cloud_safe(_fetch_understat_async(league, season_start_year))
            # se Understat ti ha risposto ma è vuoto, prova prossimo alias
            if isinstance(raw, list) and len(raw) > 0:
                cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
                return _understat_to_df(raw)
        except Exception as e:
            last_err = e
            continue

    # se arrivi qui: o blocco o veramente vuoto
    # scrivo cache "vuota" per non martellare
    try:
        cache_file.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return pd.DataFrame()


def build_understat_team_match_index(us_df: pd.DataFrame) -> pd.DataFrame:
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
    return df_long.sort_values(["team_n", "data"])

