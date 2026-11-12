"""Shared NBA preprocessing for points, rebounds and assists.

Extracted from the three user-provided notebooks; NO models are trained.

Examples:
    python preprocess_nba_props.py --players PlayerStatisticsExtended.csv --output-dir processed
    python preprocess_nba_props.py --kaggle --output-dir processed
    python preprocess_nba_props.py --players PlayerStatisticsExtended.csv --props assists points

Important: notebook-derived `pregame_available` is based on historical game comments,
not a verified pre-tip injury feed. Never treat it as a true historical pregame signal.
The original rebound optional roster feed is disabled unless explicitly supplied.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
import datetime as dt
import json
import logging
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen


def load_player_data(path: str | None = None, kaggle: bool = False) -> pl.LazyFrame:
    overrides = {"numMinutes": pl.String, "seriesGameNumber": pl.String}
    if kaggle:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        result = kagglehub.dataset_load(
            KaggleDatasetAdapter.POLARS,
            "eoinamoore/historical-nba-data-and-player-box-scores",
            "PlayerStatisticsExtended.csv",
            polars_kwargs={
        "schema_overrides": {
            "seriesGameNumber": pl.String,
            "numMinutes": pl.String,

            # Basic box score statistics
            "points": pl.Float64,
            "reboundsDefensive": pl.Float64,
            "reboundsOffensive": pl.Float64,
            "reboundsTotal": pl.Float64,
            "foulsPersonal": pl.Float64,
            "assists": pl.Float64,
            "steals": pl.Float64,
            "blocks": pl.Float64,
            "turnovers": pl.Float64,

            # Shooting
            "fieldGoals": pl.Float64,
            "fieldGoalsAttempted": pl.Float64,
            "fieldGoalsMade": pl.Float64,
            "fieldGoalsPercentage": pl.Float64,

            "threePointersMade": pl.Float64,
            "threePointersAttempted": pl.Float64,
            "threePointersPercentage": pl.Float64,

            "freeThrowsMade": pl.Float64,
            "freeThrowsAttempted": pl.Float64,
            "freeThrowsPercentage": pl.Float64,

            "plusMinusPoints": pl.Float64,

            # Advanced statistics
            "offensiveRating": pl.Float64,
            "defensiveRating": pl.Float64,
            "assistPercentage": pl.Float64,
            "assistToTurnoverRatio": pl.Float64,
            "assistRatio": pl.Float64,
            "offensiveReboundPercentage": pl.Float64,
            "defensiveReboundPercentage": pl.Float64,
            "reboundPercentage": pl.Float64,
            "estimatedTurnoverPercentage": pl.Float64,
            "effectiveFieldGoalPercentage": pl.Float64,
            "usagePercentage": pl.Float64,
            "pace": pl.Float64,
            "pacePer40": pl.Float64,

            # Percentage of team production
            "percentTeamOffensiveRebounds": pl.Float64,
            "percentTeamDefensiveRebounds": pl.Float64,
            "percentTeamBlocks": pl.Float64,
            "percentTeamPoints": pl.Float64,
        }
    },
        )
        return result.lazy() if isinstance(result, pl.DataFrame) else result
    if not path:
        raise ValueError("Supply --players PATH or --kaggle")
    return pl.scan_csv(path, schema_overrides=overrides, infer_schema_length=10000)



def fetch_daily_lineups(api_date: dt.date, timeout: int = 15) -> pl.DataFrame | None:
    """Fetch and validate an NBA daily lineup roster; return None on any failure."""
    url = f"https://stats.nba.com/js/data/leaders/00_daily_lineups_{api_date:%Y%m%d}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Accept": "application/json, text/plain, */*",
    }
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            payload = json.load(response)
        assert isinstance(payload, dict) and isinstance(payload.get("games"), list), "Missing games array"
        assert payload["games"], "No games listed for this date"
        rows = []
        for game in payload["games"]:
            game_id = str(game["gameId"])
            for side, other in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
                team = game[side]
                opponent = game[other]
                for player in team.get("players", []):
                    rows.append({
                        "personId": int(player["personId"]),
                        "gameId": game_id,
                        "gameDateTimeEst": api_date,
                        "playerteamId": int(team["teamId"]),
                        "opponentteamId": int(opponent["teamId"]),
                        "home": side == "homeTeam",
                        "playerName": player.get("playerName"),
                        "rosterStatus": player.get("rosterStatus"),
                        "lineupStatus": player.get("lineupStatus"),
                        "position": player.get("position"),
                        "gameStatus": game.get("gameStatus"),
                    })
        assert rows, "No players returned"
        assert all(r["personId"] > 0 and r["playerteamId"] > 0 for r in rows)
        result = pl.DataFrame(rows).unique(subset=["gameId", "personId"])
        assert result.height > 0
        print(f"Lineup API OK: {url} | {len(payload['games'])} games, {result.height} players")
        return result
    except Exception as exc:
        logging.warning("Lineup API unavailable/invalid (%s). Skipping lineup merge and new rows.", exc)
        return None



# Injury report is a supplementary source, not a complete roster.
NBA_TEAM_ABBR = {
    'ATL':1610612737,'BOS':1610612738,'BKN':1610612751,'CHA':1610612766,
    'CHI':1610612741,'CLE':1610612739,'DAL':1610612742,'DEN':1610612743,
    'DET':1610612765,'GSW':1610612744,'HOU':1610612745,'IND':1610612754,
    'LAC':1610612746,'LAL':1610612747,'MEM':1610612763,'MIA':1610612748,
    'MIL':1610612749,'MIN':1610612750,'NOP':1610612740,'NYK':1610612752,
    'OKC':1610612760,'ORL':1610612753,'PHI':1610612755,'PHX':1610612756,
    'POR':1610612757,'SAC':1610612758,'SAS':1610612759,'TOR':1610612761,
    'UTA':1610612762,'WAS':1610612764,
}

def _norm_player(value):
    import re, unicodedata
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r'[^a-z0-9 ]', ' ', value.lower())
    return ' '.join(value.split())


def extract_injuries(report_time: dt.datetime) -> pl.DataFrame:
    """Fetch an NBA injury-report snapshot and normalize Last, First names."""
    from nbainjuries import injury
    import re

    report = injury.get_reportdata(report_time, return_df=True)
    if report is None or len(report) == 0:
        raise ValueError(f"No injury report returned for {report_time}")
    df = report if isinstance(report, pl.DataFrame) else pl.from_pandas(report)
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}

    def field(*choices):
        return next((cols[re.sub(r"[^a-z0-9]", "", c.lower())]
                     for c in choices if re.sub(r"[^a-z0-9]", "", c.lower()) in cols), None)

    player_col = field("Player Name", "Player", "player_name")
    status_col = field("Current Status", "Status", "Injury Status")
    team_col = field("Team", "Team Abbreviation", "TeamAbbreviation")
    matchup_col = field("Matchup", "Game", "Match Up")
    game_date_col = field("Game Date", "GameDate")
    if not player_col or not status_col or not (team_col or matchup_col):
        raise ValueError(f"Unexpected nbainjuries columns: {df.columns}")

    def first_last(value):
        value = str(value or "").strip()
        if "," in value:
            last, first = value.split(",", 1)
            return f"{first.strip()} {last.strip()}".strip()
        return value

    records = []
    for row in df.to_dicts():
        records.append({
            "injury_player_original": str(row.get(player_col) or ""),
            "playerName": first_last(row.get(player_col)),
            "injuryStatus": str(row.get(status_col) or "").strip(),
            "injury_team": str(row.get(team_col) or "").strip() if team_col else "",
            "injury_matchup": str(row.get(matchup_col) or "").strip() if matchup_col else "",
            "injury_game_date": str(row.get(game_date_col) or "")[:10] if game_date_col else "",
        })
    return pl.DataFrame(records)


def distinct_kaggle_players(historical_raw: pl.LazyFrame,
                            as_of_date: dt.date) -> pl.DataFrame:
    """Distinct player/team/ID combinations, with last observed team-game date."""
    required = {"personId", "playerteamId", "firstName", "lastName", "gameDateTimeEst"}
    missing = required - set(historical_raw.collect_schema().names())
    if missing:
        raise ValueError(f"Missing Kaggle player lookup columns: {sorted(missing)}")
    return (
        historical_raw
        .select("personId", "playerteamId", "firstName", "lastName", "gameDateTimeEst")
        .with_columns(
            pl.col("gameDateTimeEst").cast(pl.String).str.slice(0, 10)
              .str.to_date(strict=False).alias("last_game_date"),
            pl.concat_str([pl.col("firstName"), pl.col("lastName")], separator=" ")
              .alias("playerName"),
        )
        .filter(pl.col("last_game_date") < pl.lit(as_of_date))
        .filter(pl.col("personId").is_not_null() & pl.col("playerteamId").is_not_null())
        .group_by("personId", "playerteamId", "playerName")
        .agg(pl.col("last_game_date").max())
        .collect()
    )


def append_injuries(lineup: pl.DataFrame, historical_raw: pl.LazyFrame,
                    report_time: dt.datetime, output_dir: Path) -> pl.DataFrame:
    """Fuzzy-match injury players by team, resolve game IDs, enrich lineup."""
    import re
    from rapidfuzz import fuzz, process

    injuries = extract_injuries(report_time)
    injuries.write_csv(output_dir / f"nba_injuries_raw_{report_time:%Y%m%d_%H%M}.csv")
    game_date = lineup["gameDateTimeEst"][0]
    if isinstance(game_date, dt.datetime):
        game_date = game_date.date()
    players = distinct_kaggle_players(historical_raw, game_date)
    players.write_csv(output_dir / f"nba_player_id_lookup_{game_date:%Y%m%d}.csv")

    # Team-name variants used in official injury reports.
    team_aliases = {
        "HAWKS":"ATL", "CELTICS":"BOS", "NETS":"BKN", "HORNETS":"CHA",
        "BULLS":"CHI", "CAVALIERS":"CLE", "MAVERICKS":"DAL", "NUGGETS":"DEN",
        "PISTONS":"DET", "WARRIORS":"GSW", "ROCKETS":"HOU", "PACERS":"IND",
        "CLIPPERS":"LAC", "LAKERS":"LAL", "GRIZZLIES":"MEM", "HEAT":"MIA",
        "BUCKS":"MIL", "TIMBERWOLVES":"MIN", "PELICANS":"NOP", "KNICKS":"NYK",
        "THUNDER":"OKC", "MAGIC":"ORL", "76ERS":"PHI", "SIXERS":"PHI",
        "SUNS":"PHX", "TRAIL BLAZERS":"POR", "BLAZERS":"POR", "KINGS":"SAC",
        "SPURS":"SAS", "RAPTORS":"TOR", "JAZZ":"UTA", "WIZARDS":"WAS",
    }
    def team_id(value):
        value = re.sub(r"\s+", " ", str(value or "").upper().strip())
        if value in NBA_TEAM_ABBR:
            return NBA_TEAM_ABBR[value]
        for alias, code in sorted(team_aliases.items(), key=lambda x: -len(x[0])):
            if re.search(r"(?<![A-Z])" + re.escape(alias) + r"(?![A-Z])", value):
                return NBA_TEAM_ABBR[code]
        return None

    def normalized(value):
        return _norm_player(value)

    # Only one game per team on the requested date is permitted.
    games = {}
    for r in lineup.to_dicts():
        tid = int(r["playerteamId"])
        key = (str(r["gameId"]), tid)
        games[key] = r
    by_team = {}
    for (gid, tid), side in games.items():
        by_team.setdefault(tid, []).append((gid, side))

    lookup = {}
    for r in players.to_dicts():
        lookup.setdefault(int(r["playerteamId"]), []).append(r)
    latest_by_id = {}
    for r in players.to_dicts():
        pid_key = int(r["personId"])
        if (pid_key not in latest_by_id or
                r["last_game_date"] > latest_by_id[pid_key]["last_game_date"]):
            latest_by_id[pid_key] = r
    lineup_rows = lineup.to_dicts()
    index = {(str(r["gameId"]), int(r["personId"])): i
             for i, r in enumerate(lineup_rows)}
    unresolved = []
    audit = []
    added = updated = 0

    for injury_row in injuries.to_dicts():
        name = injury_row["playerName"]
        status = injury_row["injuryStatus"]
        matchup = injury_row["injury_matchup"].upper()
        tid = team_id(injury_row["injury_team"])
        if injury_row["injury_game_date"] and injury_row["injury_game_date"] != str(game_date):
            continue
        matchup_match = re.search(r"\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b", matchup)
        allowed = None
        if matchup_match:
            allowed = {NBA_TEAM_ABBR.get(matchup_match.group(1)),
                       NBA_TEAM_ABBR.get(matchup_match.group(2))}
            allowed.discard(None)
        if tid is None:
            unresolved.append({**injury_row, "reason": "Unrecognised injury team"})
            continue
        fixtures = [(gid, side) for gid, side in by_team.get(tid, [])
                    if allowed is None or (tid in allowed and int(side["opponentteamId"]) in allowed)]
        if len(fixtures) != 1:
            unresolved.append({**injury_row, "reason": f"{len(fixtures)} matching fixtures"})
            continue
        gid, side = fixtures[0]

        # Prefer the authoritative NBA lineup IDs when the player is present.
        roster = [r for r in lineup_rows if str(r["gameId"]) == gid and int(r["playerteamId"]) == tid]
        exact = {int(r["personId"]) for r in roster if normalized(r.get("playerName")) == normalized(name)}
        if len(exact) == 1:
            pid = next(iter(exact))
            score = 100.0
            source = "lineup_exact"
        else:
            # Kaggle history: keep only the latest team for each player to
            # reduce stale matches after trades. Do not rely on name alone.
            pool = lookup.get(tid, [])
            pool = [r for r in pool if latest_by_id[int(r["personId"])]["playerteamId"] == tid]
            # Collapse repeated name spellings for the same ID.
            candidates = {}
            for r in pool:
                candidates[normalized(r["playerName"])] = int(r["personId"])
            if not candidates:
                unresolved.append({**injury_row, "reason": "No current-team Kaggle candidates"})
                continue
            results = process.extract(normalized(name), list(candidates), scorer=fuzz.WRatio, limit=2)
            score = float(results[0][1]) if results else 0.0
            runner_up = float(results[1][1]) if len(results) > 1 else 0.0
            if score < 90 or (len(results) > 1 and score - runner_up < 5):
                unresolved.append({**injury_row, "reason": f"Ambiguous/weak fuzzy match: {score:.1f}, second {runner_up:.1f}"})
                continue
            pid = candidates[results[0][0]]
            source = "kaggle_fuzzy"

        status_lower = status.strip().lower()
        availability = (0 if status_lower in ("out", "out for season") else
                        1 if status_lower in ("available", "probable") else None)
        key = (gid, pid)
        if key in index:
            target = lineup_rows[index[key]]
            updated += 1
        else:
            target = dict(side)
            target.update(personId=pid, playerName=name, lineupStatus=None,
                          rosterStatus=None, position=None)
            lineup_rows.append(target)
            index[key] = len(lineup_rows) - 1
            added += 1
        target["injuryStatus"] = status
        target["injuryAvailability"] = availability
        audit.append({**injury_row, "gameId": gid, "personId": pid,
                      "playerteamId": tid, "match_score": score, "match_source": source})

    if audit:
        pl.DataFrame(audit).write_csv(output_dir / f"nba_injuries_matched_{report_time:%Y%m%d_%H%M}.csv")
    if unresolved:
        pl.DataFrame(unresolved).write_csv(output_dir / f"nba_injuries_unresolved_{report_time:%Y%m%d_%H%M}.csv")
    logging.info("Injury merge: %d lineup players updated, %d new rows, %d unresolved",
                 updated, added, len(unresolved))
    return pl.DataFrame(lineup_rows, infer_schema_length=None)


def append_lineup_rows(historical: pl.LazyFrame, lineup: pl.DataFrame) -> pl.LazyFrame:
    """Create future player-game records, leaving game outcomes NULL.

    Use historical schema and retain lineup-supplied team/roster data.
    The roster feed does NOT provide current-game box scores or minutes.
    """
    schema = historical.collect_schema()
    if "injuryAvailability" not in lineup.columns:
        lineup = lineup.with_columns(pl.lit(None, dtype=pl.Int8).alias("injuryAvailability"))
    lineup = lineup.with_columns([
        pl.col("personId").cast(schema["personId"]),
        pl.col("playerteamId").cast(schema["playerteamId"]),
        pl.col("opponentteamId").cast(schema["opponentteamId"]),
        pl.col("gameId").cast(schema["gameId"]),
        pl.col("gameDateTimeEst").cast(schema["gameDateTimeEst"]),
        pl.col("home").cast(schema["home"]),
    ])
    # The API's rosterStatus is the actual pregame status for future rows.
    lineup = lineup.with_columns([
        pl.coalesce([
            pl.col("injuryAvailability").cast(pl.Int8),
            pl.when(pl.col("rosterStatus").str.to_lowercase() == "active")
              .then(pl.lit(1, dtype=pl.Int8))
              .when(pl.col("rosterStatus").str.to_lowercase().is_in(["inactive", "out"]))
              .then(pl.lit(0, dtype=pl.Int8))
              .otherwise(pl.lit(None, dtype=pl.Int8))
        ]).alias("pregame_available"),
        # In the NBA daily-lineups feed, `position` identifies the starting five.
        # `lineupStatus="Confirmed"` is also used for bench players, so it must
        # NOT be interpreted as a starter flag.
        # Only classify players when exactly five active players per team/game
        # have a recognised starting position. Otherwise keep the flag unknown.
        pl.when(
            (
                pl.col("position").fill_null("").str.strip_chars().str.to_uppercase()
                .is_in(["PG", "SG", "SF", "PF", "C"])
                & (pl.col("rosterStatus").fill_null("").str.to_lowercase() == "active")
            ).cast(pl.Int8).sum().over(["gameId", "playerteamId"]) == 5
        )
          .then(
              pl.when(
                  pl.col("position").fill_null("").str.strip_chars().str.to_uppercase()
                  .is_in(["PG", "SG", "SF", "PF", "C"])
                  & (pl.col("rosterStatus").fill_null("").str.to_lowercase() == "active")
              ).then(pl.lit(1, dtype=pl.Int8))
               .otherwise(pl.lit(0, dtype=pl.Int8))
          )
          .otherwise(pl.lit(None, dtype=pl.Int8))
          .alias("is_starter"),
    ])
    # Build schema-matched future records. Never copy a player's last game's
    # box score into a new game: that would leak fabricated outcomes.
    cols = []
    for name, dtype in schema.items():
        if name == "is_prediction_row":
            cols.append(pl.lit(True).cast(dtype).alias(name))
        elif name in lineup.columns:
            cols.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            cols.append(pl.lit(None, dtype=dtype).alias(name))
    future = lineup.lazy().select(cols)
    # Avoid duplicate player-game keys if the historical source already
    # contains rows for the requested fixture.
    existing_keys = historical.select(["gameId", "personId"]).unique()
    future = future.join(existing_keys, on=["gameId", "personId"], how="anti")
    return pl.concat([historical, future], how="vertical")


def shared_clean(lf_player_stats: pl.LazyFrame) -> pl.LazyFrame:
    """Exactly the common basic cleaning from the three notebooks."""
    ## ============================================================
    # BASIC CLEANING OF DATA
    # ============================================================


    # ============================================================
    # 1. CONVERT numMinutes FROM MM:SS TO DECIMAL MINUTES
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns(
        pl.when(
            pl.col("numMinutes").is_not_null()
            & pl.col("numMinutes").str.contains(":")
        )
        .then(
            pl.col("numMinutes")
            .str.split_exact(":", 1)
            .struct.field("field_0")
            .cast(pl.Float64)
            +
            (
                pl.col("numMinutes")
                .str.split_exact(":", 1)
                .struct.field("field_1")
                .cast(pl.Float64)
                / 60
            )
        )
        .otherwise(
            pl.col("numMinutes").cast(pl.Float64, strict=False)
        )
        .alias("numMinutes")
    )


    # ============================================================
    # 2. SELECT RELEVANT COLUMNS
    # ============================================================

    lf_player_stats = lf_player_stats.select(

        # --------------------------------------------------------
        # Identifiers
        # --------------------------------------------------------
        "personId",
        "gameId",
        "gameDateTimeEst",

        # --------------------------------------------------------
        # Team / opponent
        # --------------------------------------------------------
        "playerteamCity",
        "playerteamName",
        "playerteamId",

        "opponentteamCity",
        "opponentteamName",
        "opponentteamId",

        "home",

        # --------------------------------------------------------
        # Basic player box score
        # --------------------------------------------------------
        "numMinutes",
        "points",
        "assists",
        "blocks",
        "steals",

        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "fieldGoalsPercentage",

        "threePointersAttempted",
        "threePointersMade",
        "threePointersPercentage",

        "freeThrowsAttempted",
        "freeThrowsMade",
        "freeThrowsPercentage",

        "reboundsDefensive",
        "reboundsOffensive",
        "reboundsTotal",

        "turnovers",
        "foulsPersonal",
        "plusMinusPoints",

        # --------------------------------------------------------
        # Advanced player statistics
        # --------------------------------------------------------
        "offensiveRating",
        "defensiveRating",

        "assistPercentage",
        "assistToTurnoverRatio",
        "assistRatio",

        "offensiveReboundPercentage",
        "defensiveReboundPercentage",
        "reboundPercentage",

        "estimatedTurnoverPercentage",
        "effectiveFieldGoalPercentage",

        "usagePercentage",

        "pace",
        "pacePer40",

        # --------------------------------------------------------
        # Percentage of team production
        # --------------------------------------------------------
        "percentTeamOffensiveRebounds",
        "percentTeamDefensiveRebounds",
        "percentTeamBlocks",
        "percentTeamPoints",

        # --------------------------------------------------------
        # Player / game context
        # --------------------------------------------------------
        "startingPosition",
        "gameType",
        "comment"
    )


    # ============================================================
    # 3. CONVERT GAME DATE
    # ============================================================

    lf_player_stats = (
        lf_player_stats
        .with_columns(
            pl.col("gameDateTimeEst")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .dt.date()
        )
        .filter(
            pl.col("gameDateTimeEst") >= dt.date(1996, 11, 1)
        )
    )


    # ============================================================
    # 4. CREATE STARTER INDICATOR
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns(
        (
            pl.col("startingPosition").is_not_null()
            &
            (pl.col("startingPosition") != "")
        )
        .cast(pl.Int8)
        .alias("is_starter")
    )


    # ============================================================
    # 5. CREATE PRE-GAME AVAILABILITY INDICATOR
    # ============================================================
    # Temporary historical proxy until NBA lineup rosterStatus
    # data is integrated.
    #
    # 1 = Available
    # 0 = Unavailable due to injury / illness etc.
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns(

        pl.when(
            pl.col("comment")
            .fill_null("")
            .str.to_lowercase()
            .str.contains(
                r"injur|illness|sick|concussion|sprain|strain|soreness|"
                r"fracture|broken|surgery|rehab|recovery|contusion|"
                r"tendinitis|tendonitis|inflammation|infection|"
                r"back spasms|personal reasons"
            )
        )
        .then(0)
        .otherwise(1)
        .cast(pl.Int8)
        .alias("pregame_available")
    )


    # ============================================================
    # 6. KEEP REGULAR-SEASON GAMES ONLY
    # ============================================================

    lf_player_stats = lf_player_stats.filter(
        ~pl.col("gameType").is_in([
            "Pre Season",
            "Post Season",
            "All-Star Game",
            "Preseason",
            "Playoffs"
        ])
    )
    return lf_player_stats

def build_points(lf_player_stats: pl.LazyFrame, lf_roster_availability=None):
    """Notebook-exact points feature pipeline, isolated from other props."""
    # ============================================================
    # PLAYER POINTS MODEL - PRE-GAME FEATURE ENGINEERING
    # ============================================================

    # IMPORTANT:
    # All historical features use shift(1), ensuring the current
    # game's information is never included in its own predictors.


    # ============================================================
    # 1. SORT DATA CHRONOLOGICALLY
    # ============================================================

    lf_player_stats = (
        lf_player_stats
        .sort([
            "personId",
            "gameDateTimeEst",
            "gameId"
        ])
    )


    # ============================================================
    # 2. DEFINE METRICS
    # ============================================================

    # Minutes / playing opportunity
    MINUTES_METRICS = [
        "numMinutes",
    ]

    # Core scoring / shot-volume statistics
    SCORING_METRICS = [
        "points",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "freeThrowsAttempted",
        "freeThrowsMade",
        "threePointersAttempted",
        "threePointersMade",
    ]

    # Offensive context
    CONTEXT_METRICS = [
        "pace",
        "usagePercentage",
    ]

    # All metrics used for historical form features
    FORM_METRICS = (
        MINUTES_METRICS
        + SCORING_METRICS
        + CONTEXT_METRICS
    )


    # ============================================================
    # 3. PREVIOUS GAME VALUES
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns([

        pl.col(metric)
        .shift(1)
        .over("personId")
        .alias(f"{metric}_lag1")

        for metric in FORM_METRICS

    ])


    # ============================================================
    # 4. EWMA FEATURES
    # ============================================================
    # alpha = 0.60
    # More weight is given to recent performances.

    lf_player_stats = lf_player_stats.with_columns([

        pl.col(metric)
        .shift(1)
        .ewm_mean(
            alpha=0.60,
            adjust=False,
            ignore_nulls=True
        )
        .over("personId")
        .alias(f"{metric}_ewma")

        for metric in FORM_METRICS

    ])


    # ============================================================
    # 5. ROLLING 5-GAME AVERAGES
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns([

        pl.col(metric)
        .shift(1)
        .rolling_mean(
            window_size=5,
            min_samples=1
        )
        .over("personId")
        .alias(f"{metric}_avg5")

        for metric in FORM_METRICS

    ])


    # ============================================================
    # 6. ROLLING 10-GAME AVERAGES
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns([

        pl.col(metric)
        .shift(1)
        .rolling_mean(
            window_size=10,
            min_samples=1
        )
        .over("personId")
        .alias(f"{metric}_avg10")

        for metric in FORM_METRICS

    ])


    # ============================================================
    # 7. TREND FEATURES
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns([

        # Points trend relative to EWMA
        (
            pl.col("points_avg5")
            - pl.col("points_ewma")
        )
        .alias("points_trend_5v_ewma"),

        # Minutes / rotation trend
        (
            pl.col("numMinutes_avg5")
            - pl.col("numMinutes_avg10")
        )
        .alias("minutes_trend_5v10"),

        # Usage / offensive role trend
        (
            pl.col("usagePercentage_avg5")
            - pl.col("usagePercentage_avg10")
        )
        .alias("usage_trend_5v10"),

        # Shot-volume trend
        (
            pl.col("fieldGoalsAttempted_avg5")
            - pl.col("fieldGoalsAttempted_avg10")
        )
        .alias("fga_trend_5v10"),

        # Three-point volume trend
        (
            pl.col("threePointersAttempted_avg5")
            - pl.col("threePointersAttempted_avg10")
        )
        .alias("threepa_trend_5v10"),

        # Free-throw volume trend
        (
            pl.col("freeThrowsAttempted_avg5")
            - pl.col("freeThrowsAttempted_avg10")
        )
        .alias("fta_trend_5v10"),

    ])

    # ============================================================
    # DEFENSIVE PROFILE FEATURE ENGINEERING
    # ============================================================


    # ============================================================
    # 1. CREATE NBA SEASON
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns(

        pl.when(
            pl.col("gameDateTimeEst").dt.month() >= 10
        )
        .then(
            pl.col("gameDateTimeEst").dt.year()
        )
        .otherwise(
            pl.col("gameDateTimeEst").dt.year() - 1
        )
        .alias("season")

    )


    # ============================================================
    # 2. PLAYER DEFENSIVE RATING HISTORY
    # ============================================================
    # IMPORTANT:
    # All defensive rating features are shifted by one game.
    # Therefore the current game's defensiveRating is never used
    # to predict the current game.
    #
    # Features reset each NBA season.
    # ============================================================

    lf_player_stats = (
        lf_player_stats
        .sort([
            "personId",
            "season",
            "gameDateTimeEst",
            "gameId",
        ])
    )


    lf_player_stats = lf_player_stats.with_columns([

        # --------------------------------------------------------
        # Previous game
        # --------------------------------------------------------

        pl.col("defensiveRating")
        .shift(1)
        .over([
            "personId",
            "season",
        ])
        .alias("defensiveRating_lag1"),


        # --------------------------------------------------------
        # EWMA
        # --------------------------------------------------------

        pl.col("defensiveRating")
        .shift(1)
        .ewm_mean(
            alpha=0.60,
            adjust=False,
            ignore_nulls=True,
        )
        .over([
            "personId",
            "season",
        ])
        .alias("defensiveRating_ewma"),


        # --------------------------------------------------------
        # Last 5
        # --------------------------------------------------------

        pl.col("defensiveRating")
        .shift(1)
        .rolling_mean(
            window_size=5,
            min_samples=1,
        )
        .over([
            "personId",
            "season",
        ])
        .alias("defensiveRating_avg5"),


        # --------------------------------------------------------
        # Last 10
        # --------------------------------------------------------

        pl.col("defensiveRating")
        .shift(1)
        .rolling_mean(
            window_size=10,
            min_samples=1,
        )
        .over([
            "personId",
            "season",
        ])
        .alias("defensiveRating_avg10"),

    ])


    # ============================================================
    # 3. CREATE TEAM-GAME POINTS SCORED
    # ============================================================
    # Collapse player box scores into one row per team per game.
    # ============================================================

    lf_team_game = (
        lf_player_stats

        .group_by([
            "season",
            "gameId",
            "gameDateTimeEst",
            "playerteamId",
            "opponentteamId",
        ])

        .agg(
            pl.when(pl.col("points").is_not_null().sum() > 0)
            .then(pl.col("points").sum())
            .otherwise(None)
            .alias("team_points")
        )
    )


    # ============================================================
    # 4. CREATE TEAM POINTS ALLOWED
    # ============================================================
    # Take the opponent's team_points and attach it to the
    # corresponding team-game row.
    #
    # Example:
    #
    # BOS 110 vs NYK 105
    #
    # BOS points_allowed = 105
    # NYK points_allowed = 110
    # ============================================================

    lf_opponent_score = (
        lf_team_game

        .select([
            "season",
            "gameId",

            pl.col("playerteamId")
            .alias("opponentteamId"),

            pl.col("team_points")
            .alias("points_allowed"),
        ])
    )


    lf_team_game = (
        lf_team_game

        .join(
            lf_opponent_score,

            on=[
                "season",
                "gameId",
                "opponentteamId",
            ],

            how="left",
        )
    )


    # ============================================================
    # 5. HISTORICAL TEAM POINTS ALLOWED
    # ============================================================
    # All features use PRIOR games only.
    #
    # Reset each season.
    # ============================================================

    lf_team_game = (
        lf_team_game

        .sort([
            "playerteamId",
            "season",
            "gameDateTimeEst",
            "gameId",
        ])
    )


    lf_team_game = lf_team_game.with_columns([

        # Previous game
        pl.col("points_allowed")
        .shift(1)
        .over([
            "playerteamId",
            "season",
        ])
        .alias("points_allowed_lag1"),


        # EWMA
        pl.col("points_allowed")
        .shift(1)
        .ewm_mean(
            alpha=0.60,
            adjust=False,
            ignore_nulls=True,
        )
        .over([
            "playerteamId",
            "season",
        ])
        .alias("points_allowed_ewma"),


        # Last 5
        pl.col("points_allowed")
        .shift(1)
        .rolling_mean(
            window_size=5,
            min_samples=1,
        )
        .over([
            "playerteamId",
            "season",
        ])
        .alias("points_allowed_avg5"),


        # Last 10
        pl.col("points_allowed")
        .shift(1)
        .rolling_mean(
            window_size=10,
            min_samples=1,
        )
        .over([
            "playerteamId",
            "season",
        ])
        .alias("points_allowed_avg10"),


        # Last 20
        pl.col("points_allowed")
        .shift(1)
        .rolling_mean(
            window_size=20,
            min_samples=1,
        )
        .over([
            "playerteamId",
            "season",
        ])
        .alias("points_allowed_avg20"),

    ])


    # ------------------------------------------------------------
    # Recent defensive trend
    #
    # Positive:
    # team is allowing MORE points recently
    #
    # Negative:
    # team is allowing FEWER points recently
    # ------------------------------------------------------------

    lf_team_game = lf_team_game.with_columns(

        (
            pl.col("points_allowed_avg5")
            - pl.col("points_allowed_avg20")
        )
        .alias("points_allowed_trend_5v20")

    )


    # FUTURE PLAYER DEFENCE FALLBACK: season-reset historical features can
    # be null on opening night. Use the last completed historical game for
    # that person, regardless of season, strictly before the prediction date.
    _last_player_defence = (
        lf_player_stats
        .filter(~pl.col("is_prediction_row").fill_null(False))
        .filter(pl.col("numMinutes").is_not_null())
        .select([
            "personId", pl.col("gameDateTimeEst").alias("_history_date"),
            pl.col("defensiveRating").alias("_prior_def_rating"),
            pl.col("numMinutes").alias("_prior_minutes"),
        ])
        .sort("_history_date")
    )
    _future_player_keys = (
        lf_player_stats.filter(pl.col("is_prediction_row").fill_null(False))
        .select(["personId", "gameId", "gameDateTimeEst"])
        .with_columns((pl.col("gameDateTimeEst") - pl.duration(days=1)).alias("_asof_lookup_date"))
        .sort("_asof_lookup_date")
        .join_asof(
            _last_player_defence, left_on="_asof_lookup_date",
            right_on="_history_date", by="personId",
            strategy="backward",
        )
        .select(["personId", "gameId", "_prior_def_rating", "_prior_minutes"])
    )
    lf_player_stats = lf_player_stats.join(
        _future_player_keys, on=["personId", "gameId"], how="left"
    ).with_columns([
        pl.when(pl.col("is_prediction_row").fill_null(False))
          .then(pl.coalesce(["defensiveRating_ewma", "_prior_def_rating"]))
          .otherwise(pl.col("defensiveRating_ewma"))
          .alias("defensiveRating_ewma"),
        pl.when(pl.col("is_prediction_row").fill_null(False))
          .then(pl.coalesce(["numMinutes_ewma", "_prior_minutes"]))
          .otherwise(pl.col("numMinutes_ewma"))
          .alias("numMinutes_ewma"),
    ]).drop(["_prior_def_rating", "_prior_minutes"])

    # ============================================================
    # 6. PLAYER EXPECTED DEFENSIVE CONTRIBUTION
    # ============================================================
    #
    # Historical player defensive quality × expected minutes.
    #
    # numMinutes_ewma must itself be calculated using previous
    # games only.
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns(

        (
            pl.col("defensiveRating_ewma")
            * pl.col("numMinutes_ewma")
        )
        .alias("defensive_rating_weighted_component")

    )


    # ============================================================
    # 7. EXPECTED FULL ROTATION DEFENSIVE RATING
    # ============================================================
    #
    # Represents expected defensive quality BEFORE accounting
    # for today's unavailable players.
    #
    # Weighted by each player's expected minutes.
    # ============================================================

    lf_rotation_defence = (
        lf_player_stats

        .filter(
            pl.col("numMinutes_ewma").is_not_null()
            &
            pl.col("defensiveRating_ewma").is_not_null()
            &
            (pl.col("numMinutes_ewma") > 0)
        )

        .group_by([
            "season",
            "gameId",
            "playerteamId",
        ])

        .agg([

            pl.col("defensive_rating_weighted_component")
            .sum()
            .alias("weighted_drtg_sum"),


            pl.col("numMinutes_ewma")
            .sum()
            .alias("expected_rotation_minutes"),


            pl.col("personId")
            .n_unique()
            .alias("rotation_player_count"),

        ])

        .with_columns(

            pl.when(
                pl.col("expected_rotation_minutes") > 0
            )
            .then(
                pl.col("weighted_drtg_sum")
                / pl.col("expected_rotation_minutes")
            )
            .otherwise(None)
            .alias("expected_rotation_drtg")

        )
    )


    # ============================================================
    # 8. EXPECTED ACTIVE DEFENSIVE ROTATION
    # ============================================================
    #
    # Same calculation as above, except players classified as
    # unavailable are removed.
    #
    # pregame_available:
    #
    #     1 = expected available
    #     0 = expected unavailable
    #
    # CURRENTLY:
    # This may be your historical comment-based proxy.
    #
    # LATER:
    # Replace with NBA daily-lineup rosterStatus.
    # ============================================================

    lf_active_defence = (
        lf_player_stats

        .filter(
            (pl.col("pregame_available") == 1)
            &
            pl.col("numMinutes_ewma").is_not_null()
            &
            pl.col("defensiveRating_ewma").is_not_null()
            &
            (pl.col("numMinutes_ewma") > 0)
        )

        .with_columns(

            (
                pl.col("defensiveRating_ewma")
                * pl.col("numMinutes_ewma")
            )
            .alias("active_drtg_component")

        )

        .group_by([
            "season",
            "gameId",
            "playerteamId",
        ])

        .agg([

            pl.col("active_drtg_component")
            .sum()
            .alias("active_drtg_sum"),


            pl.col("numMinutes_ewma")
            .sum()
            .alias("active_expected_minutes"),


            pl.col("personId")
            .n_unique()
            .alias("expected_active_rotation_players"),

        ])

        .with_columns(

            pl.when(
                pl.col("active_expected_minutes") > 0
            )
            .then(
                pl.col("active_drtg_sum")
                / pl.col("active_expected_minutes")
            )
            .otherwise(None)
            .alias("expected_active_rotation_drtg")

        )
    )


    # ============================================================
    # 9. JOIN EXPECTED ROTATION DRtg TO PLAYER DATA
    # ============================================================
    # Required before calculating individual defensive impact lost.
    # ============================================================

    lf_player_stats = (
        lf_player_stats

        .join(

            lf_rotation_defence.select([
                "season",
                "gameId",
                "playerteamId",
                "expected_rotation_drtg",
            ]),

            on=[
                "season",
                "gameId",
                "playerteamId",
            ],

            how="left",
        )
    )


    # ============================================================
    # 10. DEFENSIVE IMPACT LOST — PLAYER LEVEL
    # ============================================================
    #
    # Lower defensive rating = better defensive performance.
    #
    # Therefore:
    #
    # team DRtg - player DRtg
    #
    # is positive when the player has a better historical
    # defensive rating than the expected team rotation.
    #
    #
    # Example:
    #
    # expected_rotation_drtg = 113
    # Gobert DRtg            = 104
    # expected minutes       = 34
    #
    # 34 × (113 - 104)
    # = 306
    #
    # Losing him therefore generates a large positive value.
    #
    # A poor defender can generate a negative value.
    # ============================================================

    lf_player_stats = lf_player_stats.with_columns(

        pl.when(
            (pl.col("pregame_available") == 0)
            &
            pl.col("numMinutes_ewma").is_not_null()
            &
            pl.col("defensiveRating_ewma").is_not_null()
            &
            pl.col("expected_rotation_drtg").is_not_null()
        )

        .then(
            pl.col("numMinutes_ewma")
            *
            (
                pl.col("expected_rotation_drtg")
                - pl.col("defensiveRating_ewma")
            )
        )

        .otherwise(0.0)

        .alias("defensive_impact_lost_component")

    )


    # ============================================================
    # 11. AGGREGATE DEFENSIVE ABSENCES
    # ============================================================

    lf_defensive_absences = (
        lf_player_stats

        .group_by([
            "season",
            "gameId",
            "playerteamId",
        ])

        .agg([

            # ----------------------------------------------------
            # Total defensive quality lost
            # ----------------------------------------------------

            pl.col("defensive_impact_lost_component")
            .sum()
            .alias("defensive_impact_lost"),


            # ----------------------------------------------------
            # Number of meaningful rotation players unavailable
            #
            # 15 expected minutes is currently the threshold.
            # ----------------------------------------------------

            (
                (
                    (pl.col("pregame_available") == 0)
                    &
                    (pl.col("numMinutes_ewma") >= 15)
                )
                .cast(pl.Int8)
                .sum()
            )
            .alias("defensive_rotation_players_out"),


            # ----------------------------------------------------
            # Total expected minutes unavailable
            # ----------------------------------------------------

            pl.when(
                pl.col("pregame_available") == 0
            )
            .then(
                pl.col("numMinutes_ewma")
                .fill_null(0)
            )
            .otherwise(0.0)
            .sum()
            .alias("defensive_minutes_out"),

        ])
    )


    # ============================================================
    # 12. BUILD TEAM DEFENSIVE PROFILE
    # ============================================================

    lf_team_defensive_profile = (
        lf_team_game

        .join(
            lf_rotation_defence,

            on=[
                "season",
                "gameId",
                "playerteamId",
            ],

            how="left",
        )

        .join(
            lf_active_defence,

            on=[
                "season",
                "gameId",
                "playerteamId",
            ],

            how="left",
        )

        .join(
            lf_defensive_absences,

            on=[
                "season",
                "gameId",
                "playerteamId",
            ],

            how="left",
        )
    )


    # ============================================================
    # 13. DEFENSIVE ROTATION CHANGE
    # ============================================================
    #
    # This is one of the most useful injury-adjusted features.
    #
    # Positive:
    # available rotation is WORSE defensively
    #
    # Negative:
    # available rotation is BETTER defensively
    #
    # Example:
    #
    # Full expected rotation = 109
    # Active rotation        = 113
    #
    # change = +4
    #
    # Defence is expected to deteriorate because of absences.
    # ============================================================

    lf_team_defensive_profile = (
        lf_team_defensive_profile

        .with_columns(

            (
                pl.col("expected_active_rotation_drtg")
                - pl.col("expected_rotation_drtg")
            )
            .alias("rotation_drtg_change")

        )
    )


    # ============================================================
    # 14. CREATE OPPONENT DEFENSIVE PROFILE
    # ============================================================
    #
    # Convert the defensive profile belonging to the opponent into
    # features that can be attached to each offensive player.
    # ============================================================

    lf_opponent_defensive_profile = (
        lf_team_defensive_profile

        .select([

            "season",
            "gameId",

            pl.col("playerteamId")
            .alias("opponentteamId"),


            # ----------------------------------------------------
            # Historical team defence
            # ----------------------------------------------------

            pl.col("points_allowed_lag1")
            .alias("opp_points_allowed_lag1"),

            pl.col("points_allowed_ewma")
            .alias("opp_points_allowed_ewma"),

            pl.col("points_allowed_avg5")
            .alias("opp_points_allowed_avg5"),

            pl.col("points_allowed_avg10")
            .alias("opp_points_allowed_avg10"),

            pl.col("points_allowed_avg20")
            .alias("opp_points_allowed_avg20"),

            pl.col("points_allowed_trend_5v20")
            .alias("opp_points_allowed_trend_5v20"),


            # ----------------------------------------------------
            # Expected rotation defence
            # ----------------------------------------------------

            pl.col("expected_rotation_drtg")
            .alias("opp_expected_rotation_drtg"),

            pl.col("expected_active_rotation_drtg")
            .alias("opp_expected_active_rotation_drtg"),

            pl.col("rotation_drtg_change")
            .alias("opp_rotation_drtg_change"),


            # ----------------------------------------------------
            # Defensive absences
            # ----------------------------------------------------

            pl.col("defensive_impact_lost")
            .alias("opp_defensive_impact_lost"),

            pl.col("defensive_rotation_players_out")
            .alias("opp_defensive_rotation_players_out"),

            pl.col("defensive_minutes_out")
            .alias("opp_defensive_minutes_out"),


            # ----------------------------------------------------
            # Rotation context
            # ----------------------------------------------------

            pl.col("rotation_player_count")
            .alias("opp_rotation_player_count"),

            pl.col("expected_active_rotation_players")
            .alias("opp_expected_active_rotation_players"),

        ])
    )


    # ============================================================
    # 15. JOIN OPPONENT DEFENSIVE PROFILE TO PLAYER DATA
    # ============================================================

    lf_player_stats = (
        lf_player_stats

        .join(
            lf_opponent_defensive_profile,

            on=[
                "season",
                "gameId",
                "opponentteamId",
            ],

            how="left",
        )
    )

    # FUTURE OPPONENT HISTORY: use last COMPLETED game, not a synthetic
    # lineup row (which has no actual points). The season-specific features
    # above can be null at the beginning of a new season.
    _completed_team = (
        lf_team_game
        .filter(pl.col("team_points").is_not_null()
                & pl.col("points_allowed").is_not_null())
        .sort(["playerteamId", "gameDateTimeEst", "gameId"])
        .with_columns([
            pl.col("points_allowed").ewm_mean(alpha=0.60, adjust=False,
                ignore_nulls=True).over("playerteamId").alias("_hist_ewma"),
            *[pl.col("points_allowed").rolling_mean(window_size=n, min_samples=1)
                .over("playerteamId").alias(f"_hist_avg{n}") for n in (5,10,20)],
        ])
        .with_columns((pl.col("_hist_avg5") - pl.col("_hist_avg20"))
                      .alias("_hist_trend"))
        .select([
            pl.col("playerteamId").alias("opponentteamId"),
            pl.col("gameDateTimeEst").alias("_hist_date"),
            pl.col("points_allowed").alias("_hist_lag1"),
            "_hist_ewma", "_hist_avg5", "_hist_avg10", "_hist_avg20",
            "_hist_trend",
        ])
        .sort("_hist_date")
    )
    _future_team_keys = (
        lf_player_stats.filter(pl.col("is_prediction_row").fill_null(False))
        .select(["gameId", "opponentteamId", "gameDateTimeEst"])
        .unique()
        .with_columns((pl.col("gameDateTimeEst") - pl.duration(days=1)).alias("_asof_lookup_date"))
        .sort("_asof_lookup_date")
        .join_asof(
            _completed_team,
            left_on="_asof_lookup_date", right_on="_hist_date",
            by="opponentteamId",
            strategy="backward",
        )
        .select(["gameId", "opponentteamId", "_hist_lag1", "_hist_ewma",
                 "_hist_avg5", "_hist_avg10", "_hist_avg20", "_hist_trend"])
    )
    lf_player_stats = lf_player_stats.join(
        _future_team_keys, on=["gameId", "opponentteamId"], how="left"
    )
    for _feature, _fallback in [
        ("opp_points_allowed_lag1", "_hist_lag1"),
        ("opp_points_allowed_ewma", "_hist_ewma"),
        ("opp_points_allowed_avg5", "_hist_avg5"),
        ("opp_points_allowed_avg10", "_hist_avg10"),
        ("opp_points_allowed_avg20", "_hist_avg20"),
        ("opp_points_allowed_trend_5v20", "_hist_trend"),
    ]:
        lf_player_stats = lf_player_stats.with_columns(
            pl.when(pl.col("is_prediction_row").fill_null(False))
              .then(pl.coalesce([pl.col(_fallback), pl.col(_feature)]))
              .otherwise(pl.col(_feature)).alias(_feature)
        )
    lf_player_stats = lf_player_stats.drop([
        "_hist_lag1", "_hist_ewma", "_hist_avg5", "_hist_avg10",
        "_hist_avg20", "_hist_trend",
    ])

    # ============================================================
    # POINTS MODEL FEATURE SET
    # ============================================================

    TARGET = "points"


    # ============================================================
    # PLAYER OFFENSIVE / FORM FEATURES
    # ============================================================

    PLAYER_FEATURES = [

        # --------------------------------------------------------
        # Minutes / Playing Opportunity
        # --------------------------------------------------------
        "numMinutes_lag1",
        "numMinutes_ewma",
        "numMinutes_avg5",
        "numMinutes_avg10",
        "minutes_trend_5v10",

        # --------------------------------------------------------
        # Points Form
        # --------------------------------------------------------
        "points_lag1",
        "points_ewma",
        "points_avg5",
        "points_avg10",
        "points_trend_5v_ewma",

        # --------------------------------------------------------
        # Field Goals
        # --------------------------------------------------------
        "fieldGoalsAttempted_lag1",
        "fieldGoalsAttempted_ewma",
        "fieldGoalsAttempted_avg5",
        "fieldGoalsAttempted_avg10",

        "fieldGoalsMade_lag1",
        "fieldGoalsMade_ewma",
        "fieldGoalsMade_avg5",
        "fieldGoalsMade_avg10",

        # --------------------------------------------------------
        # Three Pointers
        # --------------------------------------------------------
        "threePointersAttempted_lag1",
        "threePointersAttempted_ewma",
        "threePointersAttempted_avg5",
        "threePointersAttempted_avg10",

        "threePointersMade_lag1",
        "threePointersMade_ewma",
        "threePointersMade_avg5",
        "threePointersMade_avg10",

        # --------------------------------------------------------
        # Free Throws
        # --------------------------------------------------------
        "freeThrowsAttempted_lag1",
        "freeThrowsAttempted_ewma",
        "freeThrowsAttempted_avg5",
        "freeThrowsAttempted_avg10",

        "freeThrowsMade_lag1",
        "freeThrowsMade_ewma",
        "freeThrowsMade_avg5",
        "freeThrowsMade_avg10",

        # --------------------------------------------------------
        # Usage
        # --------------------------------------------------------
        "usagePercentage_lag1",
        "usagePercentage_ewma",
        "usagePercentage_avg5",
        "usagePercentage_avg10",
        "usage_trend_5v10",

        # --------------------------------------------------------
        # Pace
        # --------------------------------------------------------
        "pace_lag1",
        "pace_ewma",
        "pace_avg5",
        "pace_avg10",

        # --------------------------------------------------------
        # Opportunity Trends
        # --------------------------------------------------------
        "fga_trend_5v10",
        "threepa_trend_5v10",
        "fta_trend_5v10",
    ]


    # ============================================================
    # OPPONENT DEFENSIVE PROFILE
    # ============================================================

    DEFENSIVE_FEATURES = [

        # --------------------------------------------------------
        # Points Allowed
        # --------------------------------------------------------
        "opp_points_allowed_ewma",
        "opp_points_allowed_avg5",
        "opp_points_allowed_avg10",
        "opp_points_allowed_avg20",
        "opp_points_allowed_trend_5v20",

        # --------------------------------------------------------
        # Expected Defensive Rotation
        # --------------------------------------------------------
        "opp_expected_rotation_drtg",

        # --------------------------------------------------------
        # Current Expected Active Defence
        # --------------------------------------------------------
        # These require genuine pre-game lineup/availability data.
        "opp_expected_active_rotation_drtg",

        # --------------------------------------------------------
        # Defensive Personnel Missing
        # --------------------------------------------------------
        "opp_defensive_impact_lost",
        "opp_defensive_rotation_players_out",
        "opp_defensive_minutes_out",
    ]


    # ============================================================
    # COMBINE MODEL FEATURES
    # ============================================================

    POINTS_FEATURES = (
        PLAYER_FEATURES
        + DEFENSIVE_FEATURES
    )


    # ============================================================
    # IDENTIFIER / CONTEXT COLUMNS
    # ============================================================

    ID_COLS = [
        "personId",
        "numMinutes",
        "gameId",
        "gameDateTimeEst",
        "season",
        "playerteamId",
        "opponentteamId",
        "home",
        "is_starter",
        "pregame_available"
    ]


    # ============================================================
    # BUILD POINTS MODEL LAZYFRAME
    # ============================================================

    lf_player_stats_points = (
        lf_player_stats

        # Player must actually have played
        .filter(
            (pl.col("points").is_not_null() & (pl.col("numMinutes") > 0))
            | pl.col("is_prediction_row").fill_null(False)
        )

        .select(
            ID_COLS
            + [TARGET]
            + POINTS_FEATURES
        )

        .sort([
            "gameDateTimeEst",
            "gameId",
            "personId"
        ])
    )
    return lf_player_stats_points, POINTS_FEATURES

def build_rebounds(lf_player_stats: pl.LazyFrame, lf_roster_availability=None):
    """Notebook-exact rebounds feature pipeline, isolated from other props."""
    import polars as pl

    # ============================================================
    # CONFIGURATION AND HELPER FUNCTIONS
    # ============================================================
    ALPHA = 0.60
    WINDOWS = (5, 10, 20)
    GAME_KEYS = ["gameId", "playerteamId"]
    PLAYER_SORT = ["personId", "gameDateTimeEst", "gameId"]
    TEAM_SORT = ["playerteamId", "gameDateTimeEst", "gameId"]

    def add_history(lf, metrics, group_col, sort_cols, windows=WINDOWS):
        """Create strictly prior-game lag, EWMA and rolling averages."""
        lf = lf.sort(sort_cols)
        lf = lf.with_columns([
            pl.col(m).shift(1).over(group_col).alias(f"{m}_lag1")
            for m in metrics
        ])
        lf = lf.with_columns([
            pl.col(m).shift(1).ewm_mean(
                alpha=ALPHA, adjust=False, ignore_nulls=True
            ).over(group_col).alias(f"{m}_ewma")
            for m in metrics
        ])
        for w in windows:
            lf = lf.with_columns([
                pl.col(m).shift(1).rolling_mean(
                    window_size=w, min_samples=1
                ).over(group_col).alias(f"{m}_avg{w}")
                for m in metrics
            ])
        return lf

    def ratio(numerator, denominator, name):
        return (
            pl.when(pl.col(denominator) > 0)
            .then(pl.col(numerator) / pl.col(denominator))
            .otherwise(None)
            .alias(name)
        )

    # ============================================================
    # 1. HISTORICAL TEAM-GAME TOTALS
    # ============================================================
    # Include all players in a completed game, not only rotation players.
    # Require both teams to have completed player-box-score records.
    # A scheduled-game row with null outcomes is excluded here.

    lf_completed = lf_player_stats.filter(
        pl.col("reboundsTotal").is_not_null()
        & pl.col("fieldGoalsAttempted").is_not_null()
    )

    SUM_METRICS = [
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "fieldGoalsAttempted", "fieldGoalsMade",
        "threePointersAttempted", "threePointersMade",
        "freeThrowsAttempted", "freeThrowsMade",
    ]

    lf_team_actual = (
        lf_completed
        .group_by(["gameId", "gameDateTimeEst", "playerteamId", "opponentteamId"])
        .agg(
            [pl.col(m).sum().alias(f"team_{m}") for m in SUM_METRICS]
            + [
                pl.col("pace").mean().alias("team_pace"),
                pl.col("personId").n_unique().alias("team_players_used"),
            ]
        )
    )

    # Opponent actual game stats, to compute team rebound rates.
    lf_opp_actual = lf_team_actual.select(
        ["gameId", pl.col("playerteamId").alias("opponentteamId")]
        + [
            pl.col(f"team_{m}").alias(f"opp_{m}")
            for m in SUM_METRICS
        ]
    )

    lf_team_actual = lf_team_actual.join(
        lf_opp_actual, on=["gameId", "opponentteamId"], how="left"
    )

    # ============================================================
    # 2. TEAM SHOOTING, REBOUNDING AND OPPORTUNITY
    # ============================================================
    lf_team_actual = lf_team_actual.with_columns([
        ratio("team_fieldGoalsMade", "team_fieldGoalsAttempted", "team_fg_pct"),
        ratio("team_threePointersMade", "team_threePointersAttempted", "team_3p_pct"),
        ratio("team_freeThrowsMade", "team_freeThrowsAttempted", "team_ft_pct"),
        (pl.col("team_fieldGoalsAttempted") - pl.col("team_fieldGoalsMade")).alias("team_fg_missed"),
        (pl.col("team_threePointersAttempted") - pl.col("team_threePointersMade")).alias("team_3p_missed"),
        (pl.col("team_freeThrowsAttempted") - pl.col("team_freeThrowsMade")).alias("team_ft_missed"),
        (pl.col("opp_fieldGoalsAttempted") - pl.col("opp_fieldGoalsMade")).alias("opp_fg_missed"),
        (pl.col("opp_threePointersAttempted") - pl.col("opp_threePointersMade")).alias("opp_3p_missed"),
        (pl.col("opp_freeThrowsAttempted") - pl.col("opp_freeThrowsMade")).alias("opp_ft_missed"),
    ])

    lf_team_actual = lf_team_actual.with_columns([
        (pl.col("team_fg_missed") + pl.col("opp_fg_missed")).alias("rebound_chances_proxy"),
        pl.col("team_fg_missed").alias("offensive_chances_proxy"),
        pl.col("opp_fg_missed").alias("defensive_chances_proxy"),
        (
            pl.when((pl.col("team_reboundsOffensive") + pl.col("opp_reboundsDefensive")) > 0)
            .then(
                pl.col("team_reboundsOffensive") /
                (pl.col("team_reboundsOffensive") + pl.col("opp_reboundsDefensive"))
            ).otherwise(None)
        ).alias("team_oreb_pct"),
        (
            pl.when((pl.col("team_reboundsDefensive") + pl.col("opp_reboundsOffensive")) > 0)
            .then(
                pl.col("team_reboundsDefensive") /
                (pl.col("team_reboundsDefensive") + pl.col("opp_reboundsOffensive"))
            ).otherwise(None)
        ).alias("team_dreb_pct"),
    ])

    # Missed FGs are a proxy for rebound opportunities, not an exact count.
    # 3P misses are a SUBSET of FG misses: do not add them again.

    TEAM_METRICS = (
        [f"team_{m}" for m in SUM_METRICS]
        + [
            "team_pace", "team_fg_pct", "team_3p_pct", "team_ft_pct",
            "team_fg_missed", "team_3p_missed", "team_ft_missed",
            "opp_fg_missed", "opp_3p_missed", "opp_ft_missed",
            "team_oreb_pct", "team_dreb_pct",
            "rebound_chances_proxy", "offensive_chances_proxy",
            "defensive_chances_proxy",
        ]
    )

    # ============================================================
    # 3. TEAM TIMELINE: COMPLETED + SCHEDULED GAMES
    # ============================================================
    # One row per team/game even for scheduled games. Joining
    # historical actuals by gameId leaves future outcome fields null.
    # With ignore_nulls=True for EWMA, null future rows must not be
    # interspersed before completed games in the training history.

    lf_team_schedule = (
        lf_player_stats.select(
            "gameId", "gameDateTimeEst", "playerteamId", "opponentteamId"
        ).unique()
    )
    lf_team_timeline = lf_team_schedule.join(
        lf_team_actual.select(["gameId", "playerteamId"] + TEAM_METRICS),
        on=GAME_KEYS, how="left"
    )

    # Rolling operations below use only completed games for each team:
    # a scheduled game receives features from earlier completed games.
    # We forward-join the most recent *completed* team snapshot,
    # where the snapshot includes that game's actual statistics.
    #
    # For historical games, we use lagged features computed on the
    # completed-only timeline; for future games, we use the latest
    # available completed snapshot strictly before tipoff.

    lf_team_history = add_history(
        lf_team_actual, TEAM_METRICS, "playerteamId", TEAM_SORT
    )
    TEAM_HIST_COLS = [
        c for c in lf_team_history.collect_schema().names()
        if c.endswith("_lag1") or c.endswith("_ewma")
        or any(c.endswith(f"_avg{w}") for w in WINDOWS)
    ]

    # Historical rows: exact match to own completed game.
    lf_team_historical_features = lf_team_history.select(
        ["gameId", "playerteamId"] + TEAM_HIST_COLS
    )

    # For scheduled rows, recompute snapshots inclusive of the most
    # recently completed game, by shifting the historical features
    # forward one team-game. A simple alternative is using a
    # backward as-of join with *inclusive* snapshots.
    #
    # Build inclusive snapshot metrics directly on completed games.
    lf_team_snapshot = lf_team_actual.sort(TEAM_SORT)
    lf_team_snapshot = lf_team_snapshot.with_columns([
        pl.col(m).ewm_mean(
            alpha=ALPHA, adjust=False, ignore_nulls=True
        ).over("playerteamId").alias(f"{m}_ewma")
        for m in TEAM_METRICS
    ])
    for w in WINDOWS:
        lf_team_snapshot = lf_team_snapshot.with_columns([
            pl.col(m).rolling_mean(window_size=w, min_samples=1)
            .over("playerteamId").alias(f"{m}_avg{w}")
            for m in TEAM_METRICS
        ])
    lf_team_snapshot = lf_team_snapshot.with_columns([
        pl.col(m).alias(f"{m}_lag1") for m in TEAM_METRICS
    ])

    lf_team_snapshot = lf_team_snapshot.select(
        ["playerteamId", "gameDateTimeEst", "gameId"] + TEAM_HIST_COLS
    )

    # Join strictly earlier games, not the current game. We use
    # timestamp + gameId ordering via asof with a strict timestamp
    # requirement; dates tied at the same instant are excluded.
    # Existing cleaning casts gameDateTimeEst to Date.
    # That conservatively excludes all games on the same date.
    lf_team_asof = (
        lf_team_timeline
        .with_columns((pl.col("gameDateTimeEst") - pl.duration(days=1)).alias("_asof_lookup_date"))
        .sort(["_asof_lookup_date", "playerteamId"])
        .join_asof(
            lf_team_snapshot.sort(["gameDateTimeEst", "playerteamId"]),
            left_on="_asof_lookup_date",
            right_on="gameDateTimeEst",
            by="playerteamId",
            strategy="backward",
            suffix="_snapshot",
        )
        .drop("_asof_lookup_date")
    )

    # Use exact historical feature rows for completed games, and
    # as-of snapshots for scheduled rows.
    lf_team_features = lf_team_asof.join(
        lf_team_historical_features,
        on=["gameId", "playerteamId"],
        how="left",
        suffix="_historical",
    )
    lf_team_features = lf_team_features.with_columns([
        pl.coalesce(
            pl.col(f"{c}_historical"),
            pl.col(c)
        ).alias(c)
        for c in TEAM_HIST_COLS
    ]).drop([f"{c}_historical" for c in TEAM_HIST_COLS])

    # ============================================================
    # 4. JOIN BOTH TEAMS' PREGAME HISTORIES TO PLAYERS
    # ============================================================
    lf_own_features = lf_team_features.select(
        ["gameId", "playerteamId"] + TEAM_HIST_COLS
    )
    lf_opponent_features = lf_team_features.select(
        ["gameId", pl.col("playerteamId").alias("opponentteamId")]
        + [pl.col(c).alias(f"opponent_{c}") for c in TEAM_HIST_COLS]
    )

    lf_player_stats = (
        lf_player_stats
        .join(lf_own_features, on=["gameId", "playerteamId"], how="left")
        .join(lf_opponent_features, on=["gameId", "opponentteamId"], how="left")
    )

    # ============================================================
    # 5. PLAYER REBOUND SHARE AND FORM
    # ============================================================
    lf_player_stats = lf_player_stats.join(
        lf_team_actual.select([
            "gameId", "playerteamId",
            "team_reboundsTotal", "team_reboundsOffensive",
            "team_reboundsDefensive",
        ]),
        on=["gameId", "playerteamId"],
        how="left",
    )

    lf_player_stats = lf_player_stats.with_columns([
        ratio("reboundsTotal", "team_reboundsTotal", "player_rebound_share"),
        ratio("reboundsOffensive", "team_reboundsOffensive", "player_oreb_share"),
        ratio("reboundsDefensive", "team_reboundsDefensive", "player_dreb_share"),
        (
            pl.when(pl.col("numMinutes") > 0)
            .then(pl.col("reboundsTotal") * 36 / pl.col("numMinutes"))
            .otherwise(None)
        ).alias("rebounds_per36"),
    ])

    PLAYER_METRICS = [
        "numMinutes", "reboundsTotal", "reboundsOffensive",
        "reboundsDefensive", "player_rebound_share",
        "player_oreb_share", "player_dreb_share",
        "rebounds_per36", "pace",
        "fieldGoalsAttempted", "threePointersAttempted",
    ]

    lf_player_stats = add_history(
        lf_player_stats, PLAYER_METRICS, "personId", PLAYER_SORT,
        windows=(5, 10)
    )
    # Prior-game player rebound-rate features (no current-game leakage).
    REBOUND_RATE_METRICS = (
        "reboundPercentage",
        "offensiveReboundPercentage",
        "defensiveReboundPercentage",
    )
    lf_player_stats = lf_player_stats.with_columns([
        pl.col(m).cast(pl.Float64, strict=False)
        .shift(1)
        .ewm_mean(alpha=ALPHA, adjust=False, ignore_nulls=True)
        .over("personId")
        .alias(f"{m}_ewma")
        for m in REBOUND_RATE_METRICS
    ] + [
        pl.col(m).cast(pl.Float64, strict=False)
        .shift(1)
        .rolling_mean(window_size=10, min_samples=1)
        .over("personId")
        .alias(f"{m}_avg10")
        for m in REBOUND_RATE_METRICS
    ])

    lf_player_stats = lf_player_stats.with_columns([
        (pl.col("reboundsTotal_avg5") - pl.col("reboundsTotal_avg10"))
        .alias("rebounds_trend_5v10"),
        (pl.col("player_rebound_share_avg5") - pl.col("player_rebound_share_avg10"))
        .alias("rebound_share_trend_5v10"),
        (pl.col("numMinutes_avg5") - pl.col("numMinutes_avg10"))
        .alias("minutes_trend_5v10"),
    ])

    print("Team, opponent, player rebound-share and historical features defined.")


    # ============================================================
    # OPTIONAL: CONFIRMED PREGAME ROSTER AVAILABILITY
    # ============================================================
    # Supply lf_roster_availability separately when available.
    #
    # Required columns:
    #   gameId, playerteamId, personId, pregame_available
    #   reboundsTotal_ewma, reboundsOffensive_ewma,
    #   reboundsDefensive_ewma, player_rebound_share_avg5
    #
    # IMPORTANT: these EWMA/share values must be computed using
    # history strictly before the game; availability must reflect
    # a timestamped pre-tipoff status, not final box-score comments.

    VACATED_FEATURES = [
        "inactive_rebounders_count",
        "vacated_rebounds_ewma",
        "vacated_oreb_ewma",
        "vacated_dreb_ewma",
        "vacated_rebound_share",
        "unknown_availability_count",
    ]

    if "lf_roster_availability" in globals():
        required = {
            "gameId", "playerteamId", "personId", "pregame_available",
            "reboundsTotal_ewma", "reboundsOffensive_ewma",
            "reboundsDefensive_ewma", "player_rebound_share_avg5",
        }
        missing = required - set(lf_roster_availability.collect_schema().names())
        if missing:
            raise ValueError(f"Roster availability missing columns: {sorted(missing)}")

        lf_vacated = (
            lf_roster_availability
            .group_by(["gameId", "playerteamId"])
            .agg([
                (
                    (pl.col("pregame_available") == 0)
                    & (pl.col("reboundsTotal_ewma") >= 5)
                ).sum().alias("inactive_rebounders_count"),
                *[
                    pl.when(pl.col("pregame_available") == 0)
                    .then(pl.col(source).fill_null(0))
                    .otherwise(0).sum().alias(dest)
                    for source, dest in [
                        ("reboundsTotal_ewma", "vacated_rebounds_ewma"),
                        ("reboundsOffensive_ewma", "vacated_oreb_ewma"),
                        ("reboundsDefensive_ewma", "vacated_dreb_ewma"),
                        ("player_rebound_share_avg5", "vacated_rebound_share"),
                    ]
                ],
                pl.col("pregame_available").is_null().sum()
                .alias("unknown_availability_count"),
            ])
        )
        lf_player_stats = lf_player_stats.join(
            lf_vacated, on=["gameId", "playerteamId"], how="left"
        )
    else:
        lf_player_stats = lf_player_stats.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c) for c in VACATED_FEATURES
        ])
        print("No verified pregame roster feed: vacated-rebound features set to null.")


    # ============================================================
    # REBOUND MODEL FEATURE ASSEMBLY
    # ============================================================
    TARGET = "reboundsTotal"

    PLAYER_FEATURES = [
        f"{m}_{suffix}"
        for m in PLAYER_METRICS
        for suffix in ("lag1", "ewma", "avg5", "avg10")
    ] + [
        "rebounds_trend_5v10",
        "rebound_share_trend_5v10",
        "minutes_trend_5v10",
    ]

    PLAYER_FEATURES += [
        f"{m}_{suffix}"
        for m in REBOUND_RATE_METRICS
        for suffix in ("ewma", "avg10")
    ]

    TEAM_FEATURES = [
        c for c in TEAM_HIST_COLS
        if c.endswith(("_ewma", "_avg5", "_avg10", "_avg20"))
    ]

    OPPONENT_FEATURES = [f"opponent_{c}" for c in TEAM_FEATURES]

    # Do not enable VACATED_FEATURES until you have a validated
    # pregame availability feed covering historical training rows.
    REBOUNDS_FEATURES = (
        PLAYER_FEATURES + TEAM_FEATURES + OPPONENT_FEATURES
        + ["is_starter", "pregame_available"]
    )

    ID_COLS = [
        "personId", "gameId", "gameDateTimeEst",
        "playerteamId", "opponentteamId", "home",
    ]

    lf_player_stats_rebounds = (
        lf_player_stats
        .filter(
            (pl.col(TARGET).is_not_null() & (pl.col("numMinutes") > 0))
            | pl.col("is_prediction_row").fill_null(False)
        )
        .select(ID_COLS + [TARGET] + REBOUNDS_FEATURES)
        .sort(["gameDateTimeEst", "gameId", "personId"])
    )

    print(f"Rebounds target: {TARGET}")
    print(f"Pregame feature count: {len(REBOUNDS_FEATURES)}")
    print("Model frame: lf_player_stats_rebounds")

    # Optional sanity checks:
    # print(lf_player_stats_rebounds.select(
    #     ["personId", "gameId", TARGET,
    #      "reboundsTotal_ewma", "player_rebound_share_avg5",
    #      "opponent_team_fg_missed_avg5"]
    # ).head(10).collect())

    return lf_player_stats_rebounds, REBOUNDS_FEATURES

def build_assists(lf_player_stats: pl.LazyFrame, lf_roster_availability=None):
    """Notebook-exact assists feature pipeline, isolated from other props."""
    import polars as pl

    ALPHA = 0.60
    WINDOWS = (5, 10)

    # Explicitly sort by game date and game ID for deterministic history.
    def history(lf, metrics, entity, sort_keys, windows=WINDOWS):
        lf = lf.sort(sort_keys)
        lf = lf.with_columns([
            pl.col(c).shift(1).over(entity).alias(f"{c}_lag1")
            for c in metrics
        ])
        lf = lf.with_columns([
            pl.col(c).shift(1).ewm_mean(
                alpha=ALPHA, adjust=False, ignore_nulls=True
            ).over(entity).alias(f"{c}_ewma")
            for c in metrics
        ])
        for w in windows:
            lf = lf.with_columns([
                pl.col(c).shift(1).rolling_mean(
                    window_size=w, min_samples=1
                ).over(entity).alias(f"{c}_avg{w}")
                for c in metrics
            ])
        return lf

    def safe_ratio(n, d, alias):
        return pl.when(pl.col(d) > 0).then(
            pl.col(n) / pl.col(d)
        ).otherwise(None).alias(alias)

    # Only completed, played box-score rows for aggregating actual game stats.
    lf_played = lf_player_stats.filter(
        pl.col("numMinutes").is_not_null()
        & (pl.col("numMinutes") > 0)
        & pl.col("assists").is_not_null()
        & pl.col("fieldGoalsAttempted").is_not_null()
    )

    # TEAM GAME TOTALS — sum player shooting and assists; team pace is approximate.
    TEAM_SUM = [
        "assists", "turnovers", "fieldGoalsAttempted", "fieldGoalsMade",
        "threePointersMade", "threePointersAttempted", "points",
    ]
    lf_team_actual = (
        lf_played.group_by([
            "gameId", "gameDateTimeEst", "playerteamId", "opponentteamId"
        ]).agg(
            [pl.col(c).sum().alias(f"team_{c}") for c in TEAM_SUM]
            + [pl.col("pace").mean().alias("team_pace")]
        )
    )
    lf_team_actual = lf_team_actual.with_columns([
        safe_ratio("team_fieldGoalsMade", "team_fieldGoalsAttempted", "team_fg_pct"),
        (
            pl.when(pl.col("team_fieldGoalsAttempted") > 0)
            .then((pl.col("team_fieldGoalsMade") + 0.5 * pl.col("team_threePointersMade")) /
                  pl.col("team_fieldGoalsAttempted"))
            .otherwise(None)
        ).alias("team_efg_pct"),
        safe_ratio("team_assists", "team_fieldGoalsMade", "team_assist_to_fgm"),
        safe_ratio("team_threePointersMade", "team_threePointersAttempted", "team_3p_pct"),
    ])

    # DEFENSIVE PROFILE: assists and shooting ALLOWED by the defense.
    # For each team-game row, take the OPPONENT's actual team offense.
    opp_cols = [
        "team_assists", "team_fieldGoalsMade", "team_fieldGoalsAttempted",
        "team_threePointersMade", "team_threePointersAttempted", "team_points",
        "team_efg_pct", "team_assist_to_fgm", "team_pace",
    ]
    lf_allowed = lf_team_actual.select(
        ["gameId", pl.col("playerteamId").alias("opponentteamId")]
        + [pl.col(c).alias("allowed_" + c.removeprefix("team_")) for c in opp_cols]
    )
    lf_team_actual = lf_team_actual.join(
        lf_allowed, on=["gameId", "opponentteamId"], how="left"
    )

    TEAM_METRICS = [
        "team_assists", "team_turnovers", "team_fieldGoalsMade",
        "team_fieldGoalsAttempted", "team_threePointersMade",
        "team_threePointersAttempted", "team_points", "team_pace",
        "team_fg_pct", "team_efg_pct", "team_assist_to_fgm", "team_3p_pct",
        "allowed_assists", "allowed_fieldGoalsMade", "allowed_fieldGoalsAttempted",
        "allowed_threePointersMade", "allowed_threePointersAttempted",
        "allowed_points", "allowed_efg_pct", "allowed_assist_to_fgm",
        "allowed_pace",
    ]

    # Strictly lagged history for every completed team game.
    lf_team_history = history(
        lf_team_actual, TEAM_METRICS, "playerteamId",
        ["playerteamId", "gameDateTimeEst", "gameId"]
    )
    TEAM_HIST = [f"{c}_{suffix}" for c in TEAM_METRICS
                 for suffix in ("lag1", "ewma", "avg5", "avg10")]

    # For future scheduled games, construct inclusive snapshots from the
    # most recent completed team game, with a STRICT earlier-date join.
    lf_snapshots = lf_team_actual.sort(
        ["playerteamId", "gameDateTimeEst", "gameId"]
    )
    lf_snapshots = lf_snapshots.with_columns([
        pl.col(c).ewm_mean(alpha=ALPHA, adjust=False, ignore_nulls=True)
        .over("playerteamId").alias(f"{c}_ewma") for c in TEAM_METRICS
    ])
    for w in WINDOWS:
        lf_snapshots = lf_snapshots.with_columns([
            pl.col(c).rolling_mean(window_size=w, min_samples=1)
            .over("playerteamId").alias(f"{c}_avg{w}") for c in TEAM_METRICS
        ])
    lf_snapshots = lf_snapshots.with_columns([
        pl.col(c).alias(f"{c}_lag1") for c in TEAM_METRICS
    ]).select(["playerteamId", "gameDateTimeEst"] + TEAM_HIST)

    lf_schedule = lf_player_stats.select([
        "gameId", "gameDateTimeEst", "playerteamId", "opponentteamId"
    ]).unique()
    lf_asof = (
        lf_schedule.with_columns(
            (pl.col("gameDateTimeEst") - pl.duration(days=1)).alias("_asof_lookup_date")
        ).sort(["_asof_lookup_date", "playerteamId"])
        .join_asof(
            lf_snapshots.sort(["gameDateTimeEst", "playerteamId"]),
            left_on="_asof_lookup_date", right_on="gameDateTimeEst",
            by="playerteamId", strategy="backward",
        )
        .drop("_asof_lookup_date")
    )
    # Exact lagged features for completed games, as-of for scheduled games.
    lf_team_feature = lf_asof.join(
        lf_team_history.select(["gameId", "playerteamId"] + TEAM_HIST),
        on=["gameId", "playerteamId"], how="left", suffix="_exact"
    )
    lf_team_feature = lf_team_feature.with_columns([
        pl.coalesce(pl.col(f"{c}_exact"), pl.col(c)).alias(c)
        for c in TEAM_HIST
    ]).drop([f"{c}_exact" for c in TEAM_HIST])

    # Own-team offense and upcoming opponent's defense are separate joins.
    lf_own = lf_team_feature.select(["gameId", "playerteamId"] + TEAM_HIST)
    DEFENSE_HIST = [c for c in TEAM_HIST if c.startswith("allowed_")]
    lf_opp = lf_team_feature.select(
        ["gameId", pl.col("playerteamId").alias("opponentteamId")]
        + [pl.col(c).alias(f"opponent_{c}") for c in DEFENSE_HIST]
    )
    lf_player_stats = (
        lf_player_stats
        .join(lf_own, on=["gameId", "playerteamId"], how="left")
        .join(lf_opp, on=["gameId", "opponentteamId"], how="left")
    )

    # PLAYER FORM. Source assistPercentage and assistToTurnoverRatio are
    # historical player-game advanced statistics. Guard null/invalid ratios.
    lf_player_stats = lf_player_stats.with_columns([
        (
            pl.when(pl.col("numMinutes") > 0)
            .then(pl.col("assists") * 36 / pl.col("numMinutes"))
            .otherwise(None)
        ).alias("assists_per36"),
    ])

    PLAYER_METRICS = [
        "assists", "assistToTurnoverRatio", "assistPercentage",
        "numMinutes", "assists_per36", "turnovers", "usagePercentage",
        "pace", "points", "fieldGoalsAttempted", "fieldGoalsMade",
    ]
    lf_player_stats = history(
        lf_player_stats, PLAYER_METRICS, "personId",
        ["personId", "gameDateTimeEst", "gameId"]
    )
    lf_player_stats = lf_player_stats.with_columns([
        (pl.col("assists_avg5") - pl.col("assists_avg10"))
        .alias("assists_trend_5v10"),
        (pl.col("assistPercentage_avg5") - pl.col("assistPercentage_avg10"))
        .alias("assist_pct_trend_5v10"),
        (pl.col("numMinutes_avg5") - pl.col("numMinutes_avg10"))
        .alias("minutes_trend_5v10"),
    ])
    print("Assist player, team eFG and opponent defense features defined.")


    TARGET = "assists"
    PLAYER_FEATURES = [
        f"{c}_{suffix}" for c in PLAYER_METRICS
        for suffix in ("lag1", "ewma", "avg5", "avg10")
    ] + ["assists_trend_5v10", "assist_pct_trend_5v10", "minutes_trend_5v10"]

    # Current team's historical eFG, shooting and assists.
    TEAM_FEATURES = [c for c in TEAM_HIST if c.startswith("team_")]
    # Opponent's OWN prior assists allowed, shooting allowed and pace.
    OPPONENT_FEATURES = [f"opponent_{c}" for c in DEFENSE_HIST]
    ASSISTS_FEATURES = (
        PLAYER_FEATURES + TEAM_FEATURES + OPPONENT_FEATURES
        + ["is_starter", "pregame_available"]
    )

    ID_COLS = [
        "personId", "gameId", "gameDateTimeEst", "playerteamId",
        "opponentteamId", "home"
    ]
    lf_player_stats_assists = (
        lf_player_stats
        .filter((pl.col(TARGET).is_not_null() & (pl.col("numMinutes") > 0))
                | pl.col("is_prediction_row").fill_null(False))
        .select(ID_COLS + [TARGET] + ASSISTS_FEATURES)
        .sort(["gameDateTimeEst", "gameId", "personId"])
    )
    print(f"Target: {TARGET}; {len(ASSISTS_FEATURES)} pregame features")

    return lf_player_stats_assists, ASSISTS_FEATURES



def export_latest_box_scores(
    raw: pl.LazyFrame,
    output_dir: Path,
    lineup_date: dt.date,
):
    """
    Export the latest completed NBA player box scores on or before lineup_date.

    Output columns:
        Player name, gameDate, gameid, playerid,
        Points, Rebounds, Assists

    Player names are constructed from firstName + lastName.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. VALIDATE REQUIRED COLUMNS
    # ============================================================

    schema = raw.collect_schema()
    columns = schema.names()

    required = [
        "personId",
        "gameId",
        "gameDateTimeEst",
        "firstName",
        "lastName",
        "points",
        "reboundsTotal",
        "assists",
    ]

    missing = [c for c in required if c not in columns]

    if missing:
        raise ValueError(
            f"Cannot export box scores: missing columns {missing}"
        )

    # ============================================================
    # 2. CONVERT GAME DATETIME TO DATE
    # ============================================================

    date_type = schema["gameDateTimeEst"]

    if date_type == pl.Date:
        date_expr = pl.col("gameDateTimeEst")

    elif date_type == pl.String:
        date_expr = (
            pl.col("gameDateTimeEst")
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False)
        )

    else:
        date_expr = (
            pl.col("gameDateTimeEst")
            .dt.date()
        )

    # ============================================================
    # 3. CREATE PLAYER NAME AND FILTER COMPLETED GAMES
    # ============================================================

    df = (
        raw
        .with_columns([
            date_expr.alias("_game_date"),

            pl.concat_str(
                [
                    pl.col("firstName")
                    .cast(pl.String)
                    .fill_null("")
                    .str.strip_chars(),

                    pl.col("lastName")
                    .cast(pl.String)
                    .fill_null("")
                    .str.strip_chars(),
                ],
                separator=" "
            )
            .str.strip_chars()
            .alias("_player_name"),
        ])

        # Only games on or before the requested lineup date
        .filter(
            pl.col("_game_date") <= lineup_date
        )

        # Exclude games without completed box scores
        .filter(
            pl.col("points").is_not_null()
            & pl.col("reboundsTotal").is_not_null()
            & pl.col("assists").is_not_null()
        )

        # Require a valid player identifier and name
        .filter(
            pl.col("personId").is_not_null()
            & (pl.col("_player_name") != "")
        )
    )

    # ============================================================
    # 4. IDENTIFY LATEST AVAILABLE COMPLETED GAME DATE
    # ============================================================

    latest = (
        df
        .select(
            pl.col("_game_date").max()
        )
        .collect()
        .item()
    )

    if latest is None:
        print(
            f"No completed player box scores found "
            f"on or before {lineup_date}."
        )
        return None

    # ============================================================
    # 5. SELECT REQUIRED OUTPUT COLUMNS
    # ============================================================

    result = (
        df
        .filter(
            pl.col("_game_date") == latest
        )
        .select([
            pl.col("_player_name").alias("Player name"),
            pl.col("_game_date").alias("gameDate"),
            pl.col("gameId").alias("gameid"),
            pl.col("personId").alias("playerid"),
            pl.col("points").alias("Points"),
            pl.col("reboundsTotal").alias("Rebounds"),
            pl.col("assists").alias("Assists"),
        ])

        # One row per player per game
        .unique(
            subset=["gameid", "playerid"],
            keep="last",
            maintain_order=True,
        )

        .sort([
            "gameDate",
            "gameid",
            "playerid",
        ])

        .collect()
    )

    # ============================================================
    # 6. EXPORT CSV
    # ============================================================

    path = output_dir / "nba_latest_player_box_scores.csv"

    result.write_csv(path)

    print(
        f"Exported {result.height} completed player box scores "
        f"for {latest}: {path}"
    )

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", help="Path to PlayerStatisticsExtended.csv")
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--output-dir", default="processed_nba_props")
    parser.add_argument("--props", nargs="+", choices=["points", "rebounds", "assists"],
                        default=["points", "rebounds", "assists"])
    parser.add_argument("--lineup-date", help="NBA calendar date YYYY-MM-DD; default tomorrow in New York")
    parser.add_argument("--skip-lineups", action="store_true", help="Historical features only")
    parser.add_argument("--injury-report-et", help="Report timestamp in US Eastern time YYYY-MM-DDTHH:MM")
    args = parser.parse_args()

    api_date = (dt.date.fromisoformat(args.lineup_date) if args.lineup_date
                else dt.datetime.now(ZoneInfo("America/New_York")).date() + dt.timedelta(days=1))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_player_data(args.players, args.kaggle)
    export_latest_box_scores(raw, output, api_date)
    historical = shared_clean(raw)
    historical = historical.with_columns([
        pl.lit(False).alias("is_prediction_row"),
        pl.col("pregame_available").cast(pl.Int8),
        pl.col("is_starter").cast(pl.Int8),
    ])

    lineup = None if args.skip_lineups else fetch_daily_lineups(api_date)
    if lineup is not None:
        # Assert valid API output before incorporating into feature calculations.
        assert lineup.height > 0 and lineup["gameId"].null_count() == 0
        if args.injury_report_et:
            report_time = dt.datetime.fromisoformat(args.injury_report_et)
            if report_time.tzinfo is not None:
                report_time = report_time.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
            if report_time.date() != api_date:
                raise ValueError('Injury report date must match --lineup-date')
            lineup = append_injuries(lineup, raw, report_time, output)
        lineup.write_csv(output / f"nba_lineups_{api_date:%Y%m%d}.csv")
        # Prediction indicator on synthetic records only.
        # We attach after building the schema-compatible rows.
        source = append_lineup_rows(historical, lineup)
    else:
        source = historical

    builders = {"points": build_points, "rebounds": build_rebounds, "assists": build_assists}
    for prop in args.props:
        print(f"\nBuilding {prop} features ...", flush=True)
        frame, features = builders[prop](source.clone())
        # Strictly older than the API date = training. Same-day or newer = new.
        # This avoids accidentally treating games on the API date as training.
        old = frame.filter(pl.col("gameDateTimeEst") < api_date)
        new = frame.filter(pl.col("gameDateTimeEst") >= api_date)
        old_path = output / f"nba_{prop}_train.parquet"
        new_path = output / f"nba_{prop}_new.parquet"
        old.sink_parquet(old_path)
        new.sink_parquet(new_path)
        if prop == "points":
            check_cols = ["opp_points_allowed_avg5", "opp_points_allowed_avg10",
                          "opp_expected_rotation_drtg",
                          "opp_expected_active_rotation_drtg",
                          "opp_defensive_minutes_out"]
            counts = new.select([
                pl.len().alias("rows"),
                *[pl.col(c).is_not_null().sum().alias(c) for c in check_cols]
            ]).collect().to_dicts()[0]
            print(f"POINTS NEW defensive feature non-null counts: {counts}", flush=True)
        print(f"Saved {old_path} and {new_path} | {len(features)} features", flush=True)
        (output / f"nba_{prop}_features.txt").write_text("\n".join(features) + "\n")
    print("Preprocessing complete. No model training performed.")


if __name__ == "__main__":
    main()
