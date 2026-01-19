import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date

UNDERSTAT_LEAGUES = {
    "SA": "serie-a",
    "PL": "epl",
    "PD": "la-liga",
    "BL1": "bundesliga",
    "FL1": "ligue-1",
}


def get_understat_matches_season(cache_dir: Path, comp_code: str, season: int):
    league = UNDERSTAT_LEAGUES.get(comp_code)
    if not league:
        return pd.DataFrame()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"understat_{comp_code}_{season}.json"

    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return _to_df(raw)

    url = f"https://understat.com/league/{league}/{season}"
    html = requests.get(url, timeout=20).text

    start = html.find("var matchesData = JSON.parse('") + 29
    end = html.find("');", start)
    json_text = html[start:end].encode("utf-8").decode("unicode_escape")

    raw = json.loads(json_text)
    cache_file.write_text(json.dumps(raw), encoding="utf-8")

    return _to_df(raw)


def _to_df(raw):
    rows = []
    for m in raw:
        rows.append({
            "date": m["datetime"][:10],
            "home": m["h"]["title"],
            "away": m["a"]["title"],
            "xg_home": float(m["xG"]["h"]),
            "xg_away": float(m["xG"]["a"]),
        })
    return pd.DataFrame(rows)


def build_understat_team_match_index(df):
    rows = []
    for _, r in df.iterrows():
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
    return pd.DataFrame(rows)
