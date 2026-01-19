# core/understat_client.py
import json
import asyncio
import re
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import aiohttp
import requests
from understat import Understat

# -------------------------------
# Mapping campionati -> Understat (slug sito)
# -------------------------------
# NB: questi sono gli slug usati negli URL di understat.com
UNDERSTAT_LEAGUE_SLUG = {
    "SA": "Serie_A",
    "PL": "EPL",
    "PD": "La_liga",
    "BL1": "Bundesliga",
    "FL1": "Ligue_1",
}


def _cache_file(cache_dir: Path, comp_code: str, season_start_year: int) -> Path:
    today = date.today().isoformat()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"understat_{comp_code}_{season_start_year}_{today}.json"


def _is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=ttl_hours)


async def _fetch_understat_lib(league_slug: str, season: int) -> list:
    """
    Fetch usando libreria 'understat' (scraping-based).
    """
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        # La libreria di solito accetta: "EPL", "Serie A", "La liga" ecc.
        # ma in cloud può essere instabile. Proviamo alcune varianti.
        league_variants = [
            league_slug,  # a volte funziona
            league_slug.replace("_", " "),  # "Serie A"
            league_slug.replace("_", "").lower(),  # fallback
        ]
        last_err = None
        for lg in league_variants:
            try:
                data = await us.get_league_results(lg, season)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception as e:
                last_err = e
                continue
        # se non funziona nulla, rilancia l'ultimo errore
        if last_err:
            raise last_err
        return []


def _decode_understat_json_string(s: str) -> str:
    """
    Understat mette JSON dentro JSON.parse('....') con escape.
    """
    # sostituisce sequenze tipo \xNN e \uNNNN
    s = s.encode("utf-8").decode("unicode_escape")
    # ripulisce eventuali escape di slash
    s = s.replace("\\/", "/")
    return s


def _fetch_understat_site(league_slug: str, season: int) -> list:
    """
    Fallback robusto: scarica HTML da understat.com e parsifica datesData (match con xG).
    """
    url = f"https://understat.com/league/{league_slug}/{season}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    html = r.text

    # cerca: datesData = JSON.parse('....')
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", html, re.DOTALL)
    if not m:
        return []

    raw_str = m.group(1)
    decoded = _decode_understat_json_string(raw_str)

    data = json.loads(decoded)
    # data è una lista di giornate; ogni elemento ha 'matches'
    matches = []
    for day in data:
        for match in day.get("matches", []):
            matches.append(match)

    return matches


def _understat_to_df(raw_matches: list) -> pd.DataFrame:
    """
    Standard:
      date | home | away | xg_home | xg_away
    """
    rows = []
    for m in raw_matches or []:
        # match structure: datetime, h.title, a.title, xG.h, xG.a
        dt = m.get("datetime") or m.get("date") or ""
        dt10 = str(dt)[:10]

        h = (m.get("h") or {}).get("title") or (m.get("home") or {}).get("title")
        a = (m.get("a") or {}).get("title") or (m.get("away") or {}).get("title")

        xg = m.get("xG") or {}
        xg_h = xg.get("h", None)
        xg_a = xg.get("a", None)

        def to_float(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        if h and a and dt10:
            rows.append({
                "date": dt10,
                "home": h,
                "away": a,
                "xg_home": to_float(xg_h),
                "xg_away": to_float(xg_a),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
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
    league_slug = UNDERSTAT_LEAGUE_SLUG.get(comp_code)
    if not league_slug:
        return pd.DataFrame()

    cache_file = _cache_file(cache_dir, comp_code, season_start_year)

    # ---------- cache ----------
    if (not force_refresh) and _is_fresh(cache_file, ttl_hours=ttl_hours):
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return _understat_to_df(raw)
        except Exception:
            pass

    # ---------- live fetch: 1) libreria understat  2) fallback sito ----------
    raw_matches = []
    try:
        raw_matches = asyncio.run(_fetch_understat_lib(league_slug, season_start_year))
    except RuntimeError:
        # event loop già attivo
        loop = asyncio.get_event_loop()
        raw_matches = loop.run_until_complete(_fetch_understat_lib(league_slug, season_start_year))
    except Exception:
        raw_matches = []

    # se libreria torna vuota, prova sito
    if not raw_matches:
        try:
            raw_matches = _fetch_understat_site(league_slug, season_start_year)
        except Exception:
            raw_matches = []

    # salva cache (anche se vuota? meglio NO)
    if raw_matches:
        try:
            cache_file.write_text(json.dumps(raw_matches, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return _understat_to_df(raw_matches)


def build_understat_team_match_index(us_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma i match Understat in formato LONG:
      date, team, opponent, xg_for, xg_against
    """
    rows = []
    if us_df is None or us_df.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "xg_for", "xg_against"])

    for _, r in us_df.iterrows():
        rows.append({
            "date": r["date"],
            "team": r["home"],
            "opponent": r["away"],
            "xg_for": r["xg_home"],
            "xg_against": r["xg_away"],
        })
        rows.append({
            "date": r["date"],
            "team": r["away"],
            "opponent": r["home"],
            "xg_for": r["xg_away"],
            "xg_against": r["xg_home"],
        })

    df_long = pd.DataFrame(rows)
    df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")
    return df_long

