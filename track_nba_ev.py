"""Persistent NBA EV bet tracker.

Prepare date-specific actuals: python track_nba_ev.py prepare --source-actuals PATH --lineup-date YYYY-MM-DD
Capture BEFORE games: python track_nba_ev.py capture
Settle AFTER updating box scores: python track_nba_ev.py settle --actuals data/PlayerStatisticsExtended.csv
Or: python track_nba_ev.py update --actuals data/PlayerStatisticsExtended.csv

Capture is idempotent; settled records are never overwritten. No bets are assumed
placed: profits represent hypothetical one-unit stakes at recorded quoted odds.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np

BLENDED_ODDS_MAX = 2.20  # Strict upper bound; 2.20 is excluded.
BLENDED_EV_MIN = 0.05  # Strict lower bound; 0.05 is excluded.
BLENDED_EV_COL = "EV of blended vs sportsbook"

STAT_COLS={'points':['points','Points','numPoints','PTS'], 'rebounds':['reboundsTotal','rebounds','Rebounds','numRebounds','REB'], 'assists':['assists','Assists','numAssists','AST']}

def norm(x):
    if pd.isna(x): return ''
    s=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower()
    s=re.sub(r'[^a-z0-9 ]',' ',s)
    return re.sub(r'\s+(jr|sr|ii|iii|iv)$','', ' '.join(s.split()))

def key(row):
    raw='|'.join(str(row.get(k,'')) for k in ('event_name','player','prop market','line','sportsbookname','sportsbook odds'))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

def load(path):
    if not path.exists(): raise FileNotFoundError(f'Missing file: {path}')
    return pd.read_csv(path, low_memory=False, dtype={c: 'string' for c in ('game_id', 'gameId', 'gameid', 'game_id')})

def clean_game_ids(values):
    """Normalise IDs as nullable strings; avoid float64 assignment failures."""
    return (values.astype('string').str.strip()
            .str.replace(r'\.0$', '', regex=True)
            .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA}))


def ensure_ledger_game_ids(ledger):
    if 'game_id' not in ledger.columns:
        ledger['game_id'] = pd.Series(pd.NA, index=ledger.index, dtype='string')
    else:
        ledger['game_id'] = clean_game_ids(ledger['game_id'])
    return ledger


# Canonical NBA matchup identifiers, independent of sportsbook wording/order.
TEAM_ALIASES = {
    'atl': ['atlanta hawks','hawks'], 'bos': ['boston celtics','celtics'],
    'bkn': ['brooklyn nets','nets'], 'cha': ['charlotte hornets','hornets'],
    'chi': ['chicago bulls','bulls'], 'cle': ['cleveland cavaliers','cavaliers','cavs'],
    'dal': ['dallas mavericks','mavericks','mavs'], 'den': ['denver nuggets','nuggets'],
    'det': ['detroit pistons','pistons'], 'gsw': ['golden state warriors','warriors'],
    'hou': ['houston rockets','rockets'], 'ind': ['indiana pacers','pacers'],
    'lac': ['los angeles clippers','la clippers','l a clippers','clippers'],
    'lal': ['los angeles lakers','la lakers','l a lakers','lakers'],
    'mem': ['memphis grizzlies','grizzlies'], 'mia': ['miami heat','heat'],
    'mil': ['milwaukee bucks','bucks'], 'min': ['minnesota timberwolves','timberwolves','wolves'],
    'nop': ['new orleans pelicans','pelicans'], 'nyk': ['new york knicks','knicks'],
    'okc': ['oklahoma city thunder','thunder'], 'orl': ['orlando magic','magic'],
    'phi': ['philadelphia 76ers','philadelphia sixers','76ers','sixers'],
    'phx': ['phoenix suns','suns'], 'por': ['portland trail blazers','portland trailblazers','trail blazers','trailblazers','blazers'],
    'sac': ['sacramento kings','kings'], 'sas': ['san antonio spurs','spurs'],
    'tor': ['toronto raptors','raptors'], 'uta': ['utah jazz','jazz'],
    'was': ['washington wizards','wizards'],
}

def matchup_key(value):
    """Return a sorted pair of NBA team codes; None if not exactly two teams."""
    if pd.isna(value):
        return None
    raw = str(value).lower().replace('@', ' vs ').replace(' v ', ' vs ')
    raw = re.sub(r'\b(?:at|versus|against|vs\.?|v)\b', ' vs ', raw)
    raw = re.sub(r'[^a-z0-9 ]+', ' ', raw)
    raw = ' '.join(raw.split())
    found = set()
    # City-only abbreviations commonly emitted by bookmakers; omit ambiguous LA/NY.
    city_aliases = {
        'atl':'atlanta', 'bos':'boston', 'bkn':'brooklyn', 'cha':'charlotte',
        'chi':'chicago', 'cle':'cleveland', 'dal':'dallas', 'den':'denver',
        'det':'detroit', 'gsw':'golden state', 'hou':'houston', 'ind':'indiana',
        'mem':'memphis', 'mia':'miami', 'mil':'milwaukee', 'min':'minnesota',
        'nop':'new orleans', 'okc':'oklahoma city', 'orl':'orlando',
        'phi':'philadelphia', 'phx':'phoenix', 'por':'portland',
        'sac':'sacramento', 'sas':'san antonio', 'tor':'toronto',
        'uta':'utah', 'was':'washington',
    }
    for code, city in city_aliases.items():
        if re.search(r'(?<![a-z0-9])' + re.escape(city) + r'(?![a-z0-9])', raw):
            found.add(code)
    for code, names in TEAM_ALIASES.items():
        for alias in [*names, code]:
            if re.search(r'(?<![a-z0-9])' + re.escape(alias) + r'(?![a-z0-9])', raw):
                found.add(code)
                break
    return '|'.join(sorted(found)) if len(found) == 2 else None

def nonempty(value):
    return pd.notna(value) and str(value).strip().lower() not in ('', 'nan', 'none', '<na>')

def enrich_ledger_from_books(ledger, books_path):
    """Fill missing game metadata only from uniquely matched sportsbook records.

    Match by bookmaker/player/stat/line/odds and canonical team pair.
    A fallback without matchup is permitted only if the candidate rows identify
    one unique event and one unique date. Existing metadata is never overwritten.
    """
    if not books_path.exists():
        print(f'No sportsbook odds file for game enrichment: {books_path}')
        return ledger
    books = load(books_path).copy()
    required = {'bookmaker', 'player_name', 'stat', 'odds'}
    if not required.issubset(books.columns):
        print(f'Skipping sportsbook enrichment; missing {sorted(required - set(books.columns))}')
        return ledger
    ledger = ensure_ledger_game_ids(ledger)
    if 'game_date' not in ledger:
        ledger['game_date'] = pd.Series(pd.NA, index=ledger.index, dtype='string')
    else:
        ledger['game_date'] = ledger['game_date'].astype('string')
    event_col = next((c for c in ('event_name', 'match_name', 'event_key') if c in books), None)
    books['_match'] = books[event_col].map(matchup_key) if event_col else None
    books['_player'] = books['player_name'].map(norm)
    books['_stat'] = books['stat'].map(norm)
    books['_book'] = books['bookmaker'].map(norm)
    books['_odds'] = pd.to_numeric(books['odds'], errors='coerce')
    raw_line = pd.to_numeric(books.get('line', pd.Series(np.nan, index=books.index)), errors='coerce')
    equivalent = pd.to_numeric(books.get('equivalent_over_line', pd.Series(np.nan, index=books.index)), errors='coerce')
    alt = books.get('market_category', pd.Series('', index=books.index)).astype(str).str.lower().isin(['alternative', 'alt', 'ladder'])
    books['_line'] = equivalent.fillna(raw_line - np.where(alt, 0.5, 0))
    if 'start_time' in books:
        starts = pd.to_datetime(books['start_time'], utc=True, errors='coerce', format='mixed')
        # Game date in the NBA's Eastern timezone, not UTC or Australian time.
        books['_date'] = starts.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    else:
        books['_date'] = pd.Series(pd.NA, index=books.index, dtype='string')
    id_col = next((c for c in ('gameId', 'gameid', 'game_id') if c in books), None)
    books['_game'] = clean_game_ids(books[id_col]) if id_col else pd.Series(pd.NA, index=books.index, dtype='string')
    filled_dates = filled_ids = 0
    for i, r in ledger.iterrows():
        if str(r.get('status')) in ('Win', 'Loss', 'Push', 'Void'):
            continue
        if nonempty(r.get('game_date')) and nonempty(r.get('game_id')):
            continue
        stat = norm(str(r.get('prop market', '')).split('-')[0])
        line = pd.to_numeric(r.get('line'), errors='coerce')
        odds = pd.to_numeric(r.get('sportsbook odds'), errors='coerce')
        if not np.isfinite(line) or not np.isfinite(odds):
            continue
        hits = books.loc[
            (books['_player'] == norm(r.get('player'))) &
            (books['_book'] == norm(r.get('sportsbookname'))) &
            (books['_stat'] == stat) &
            np.isclose(books['_line'], line, atol=.001, rtol=0) &
            np.isclose(books['_odds'], odds, atol=.001, rtol=0)
        ]
        expected_match = matchup_key(r.get('event_name'))
        if expected_match is not None:
            hits = hits[hits['_match'] == expected_match]
        else:
            # Without identifiable teams, only use an exact normalized event.
            expected_event = norm(r.get('event_name'))
            if event_col is None or not expected_event:
                continue
            hits = hits[hits[event_col].map(norm) == expected_event]
        if hits.empty:
            continue
        # Never infer an event from one of several distinct matchups.
        matches = hits['_match'].dropna().unique()
        if len(matches) > 1:
            continue
        dates = hits['_date'].dropna().unique()
        ids = hits['_game'].dropna().unique()
        if not nonempty(r.get('game_date')) and len(dates) == 1:
            ledger.at[i, 'game_date'] = str(dates[0])
            filled_dates += 1
        if not nonempty(r.get('game_id')) and len(ids) == 1:
            ledger.at[i, 'game_id'] = str(ids[0])
            filled_ids += 1
    print(f'Sportsbook enrichment: filled {filled_dates} missing dates, {filled_ids} missing game IDs')
    return ledger

def capture(args):
    best=load(args.best)
    required=['event_name','player','prop market','line','sportsbookname','sportsbook odds']
    missing=set(required)-set(best.columns)
    if missing: raise ValueError(f'Best bets missing: {sorted(missing)}')
    best=best.copy()
    if 'blended odds' not in best.columns:
        raise ValueError("Best bets missing required 'blended odds' column")
    if BLENDED_EV_COL not in best.columns:
        raise ValueError(f"Best bets missing required '{BLENDED_EV_COL}' column")
    blended = pd.to_numeric(best['blended odds'], errors='coerce')
    blended_ev = pd.to_numeric(best[BLENDED_EV_COL], errors='coerce')
    before = len(best)
    best = best.loc[
        (blended < BLENDED_ODDS_MAX) & (blended_ev > BLENDED_EV_MIN)
    ].copy()
    print(f'Blended odds < {BLENDED_ODDS_MAX:.2f} and blended EV > {BLENDED_EV_MIN:.0%}: '
          f'{len(best)} of {before} candidate bets retained')
    best['play_id']=best.apply(key,axis=1)
    best['captured_at_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    best['status']='Pending'
    best['actual_value']=np.nan
    best['units_profit']=np.nan
    best['game_id']=pd.Series(pd.NA, index=best.index, dtype='string')
    best['game_date']=''
    # Resolve different sportsbook event names through canonical NBA team pairs.
    best = enrich_ledger_from_books(best, args.books)
    args.ledger.parent.mkdir(parents=True,exist_ok=True)
    if args.ledger.exists():
        old=ensure_ledger_game_ids(load(args.ledger))
        # Preserve all historical selections and original snapshots.
        fresh=best[~best['play_id'].isin(old['play_id'])]
        ledger=pd.concat([old,fresh],ignore_index=True)
    else: fresh=best; ledger=best
    ledger=ensure_ledger_game_ids(ledger)
    ledger.to_csv(args.ledger,index=False)
    print(f'Captured {len(fresh)} new plays; ledger contains {len(ledger)} plays: {args.ledger}')

def prepare_actuals(source, destination, lineup_date):
    """Save game-day rows from the raw Kaggle box-score dataset.

    Filter by the calendar date written in gameDateTimeEst (Eastern game date),
    without converting to UTC, which could shift games to the next day.
    """
    if source is None:
        raise ValueError('Supply --source-actuals with the path to the raw PlayerStatisticsExtended CSV/parquet')
    if lineup_date is None:
        raise ValueError('Supply --lineup-date YYYY-MM-DD')
    day = pd.Timestamp(lineup_date).date()
    if source.resolve() == destination.resolve():
        raise ValueError('--source-actuals and --actuals must be different files (avoid overwriting the raw dataset)')
    df = load(source) if source.suffix.lower() == '.csv' else pd.read_parquet(source)
    date_col = next((c for c in ('gameDateTimeEst', 'gameDateTime', 'gameDate', 'game_date') if c in df.columns), None)
    if date_col is None:
        raise ValueError('Raw actuals missing gameDateTimeEst/gameDateTime/gameDate/game_date column')
    # A timezone-aware string is deliberately interpreted using its written
    # calendar date rather than converted to UTC.
    date_text = df[date_col].astype('string').str.extract(r'^(\d{4}-\d{2}-\d{2})', expand=False)
    parsed = pd.to_datetime(date_text, errors='coerce').dt.date
    if parsed.isna().all():
        parsed = pd.to_datetime(df[date_col], errors='coerce', format='mixed', utc=True).dt.date
    selected = df.loc[parsed == day].copy()
    if selected.empty:
        raise ValueError(f'No box-score rows for {day} in {source} (column {date_col}); output not overwritten')
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)
    print(f'Saved {len(selected)} player-game rows for {day} to {destination} (filtered by {date_col})')
    return selected



def actuals_table(path):
    from pathlib import Path

    path = Path(path)

    df = (
        load(path)
        if path.suffix.lower() == ".csv"
        else pd.read_parquet(path)
    )

    df = df.copy()

    # ============================================================
    # 1. PLAYER NAME
    # ============================================================

    name = next(
        (
            x for x in [
                "playerName",
                "player_name",
                "personName",
                "name",
                "fullName",
                "Player Name",
                "Player name",
            ]
            if x in df.columns
        ),
        None,
    )

    if name is None and {"firstName", "lastName"}.issubset(df.columns):
        df["_name"] = (
            df["firstName"].fillna("").astype(str)
            + " "
            + df["lastName"].fillna("").astype(str)
        ).str.strip()

        name = "_name"

    if name is None:
        raise ValueError(
            "Actuals require 'Player name', 'playerName', "
            "or firstName + lastName"
        )

    df["_player"] = df[name].map(norm)

    # ============================================================
    # 2. GAME DATE
    # ============================================================

    date = next(
        (
            x for x in [
                "gameDateTimeEst",
                "gameDate",
                "game_date",
                "start_time",
            ]
            if x in df.columns
        ),
        None,
    )

    if date is None:
        raise ValueError(
            "Actuals require gameDateTimeEst or gameDate"
        )

    df["_date"] = (
        df[date]
        .astype("string")
        .str.extract(
            r"^(\d{4}-\d{2}-\d{2})",
            expand=False
        )
    )

    if df["_date"].isna().all():
        df["_date"] = (
            pd.to_datetime(
                df[date],
                errors="coerce",
                format="mixed",
                utc=True
            )
            .dt.date
            .astype("string")
        )

    # ============================================================
    # 3. GAME ID
    # ============================================================

    game_col = next(
        (
            x for x in ["gameId", "gameid", "game_id"]
            if x in df.columns
        ),
        None,
    )

    df["_game"] = (
        clean_game_ids(df[game_col])
        if game_col is not None
        else ""
    )

    # ============================================================
    # 4. POINTS, REBOUNDS AND ASSISTS
    # ============================================================

    stat_aliases = {
        "points": [
            "points", "Points", "pts", "PTS"
        ],
        "rebounds": [
            "reboundsTotal", "rebounds",
            "Rebounds", "reb", "REB"
        ],
        "assists": [
            "assists", "Assists", "ast", "AST"
        ],
    }

    # Preserve any additional aliases from your existing script.
    for stat, aliases in STAT_COLS.items():
        stat_aliases.setdefault(stat, [])

        stat_aliases[stat] = list(
            dict.fromkeys(stat_aliases[stat] + list(aliases))
        )

    for stat, aliases in stat_aliases.items():

        col = next(
            (x for x in aliases if x in df.columns),
            None,
        )

        df["_" + stat] = (
            pd.to_numeric(df[col], errors="coerce")
            if col is not None
            else np.nan
        )

    return df


def settle(args):
    if not args.ledger.exists(): raise FileNotFoundError('No ledger yet; run capture first')
    ledger=ensure_ledger_game_ids(load(args.ledger).copy())
    # Repair existing pending records, not just newly captured bets.
    ledger=enrich_ledger_from_books(ledger, args.books)
    actual=actuals_table(args.actuals)
    settled=0; ambiguous=0
    for i,r in ledger.iterrows():
        if str(r['status']) in ('Win','Loss','Push','Void'): continue
        stat=norm(str(r['prop market']).split('-')[0]); side=norm(str(r['prop market']).split('-')[-1]);
        if stat not in STAT_COLS or side not in ('over','under'): continue
        if not nonempty(r.get('game_date')) and not nonempty(r.get('game_id')): continue
        matches=actual[actual['_player']==norm(r['player'])]
        gid=r.get('game_id')
        if nonempty(gid):
            matches=matches[matches['_game']==str(gid)]
        else:
            date=str(r.get('game_date',''))
            if not date or date=='nan': continue
            # NBA Eastern date can be the previous day relative to UTC start time.
            d=pd.Timestamp(date).date()
            dates={str(d),str(d-pd.Timedelta(days=1))}
            matches=matches[matches['_date'].isin(dates)]
        matches=matches[matches['_'+stat].notna()]
        # Only settle if precisely one distinct game; never pick a random box score.
        unique=matches.drop_duplicates(['_game','_date','_player','_'+stat])
        if len(unique)!=1:
            if len(unique)>1: ambiguous+=1
            continue
        val=float(unique.iloc[0]['_'+stat]); line=float(r['line']); odds=float(r['sportsbook odds'])
        result='Push' if np.isclose(val,line) else ('Win' if (val>line if side=='over' else val<line) else 'Loss')
        ledger.at[i,'actual_value']=val
        ledger.at[i,'status']=result
        ledger.at[i,'units_profit']=odds-1 if result=='Win' else (0 if result=='Push' else -1)
        if not nonempty(gid): ledger.at[i,'game_id']=unique.iloc[0]['_game']
        settled+=1
    ledger.to_csv(args.ledger,index=False)
    # Restrict reporting to blended odds < 2.20 and blended EV > 5%.
    # Keep the full historical ledger unchanged, including excluded and pending plays.
    if 'blended odds' not in ledger.columns:
        raise ValueError("Ledger missing required 'blended odds' column")
    if BLENDED_EV_COL not in ledger.columns:
        raise ValueError(f"Ledger missing required '{BLENDED_EV_COL}' column")
    blended_odds = pd.to_numeric(ledger['blended odds'], errors='coerce')
    blended_ev = pd.to_numeric(ledger[BLENDED_EV_COL], errors='coerce')
    qualified = ledger.loc[
        (blended_odds < BLENDED_ODDS_MAX) & (blended_ev > BLENDED_EV_MIN)
    ].copy()
    completed = qualified[qualified['status'].isin(['Win', 'Loss', 'Push'])].copy()
    if len(completed):
        completed['units_profit'] = pd.to_numeric(completed['units_profit'], errors='coerce')
        completed['ev_model'] = pd.to_numeric(completed.get('EV of model vs sportsbook'), errors='coerce')
        completed['ev_blended'] = pd.to_numeric(completed.get('EV of blended vs sportsbook'), errors='coerce')
        summary = completed.groupby('prop market', dropna=False).agg(
            plays=('play_id', 'size'),
            wins=('status', lambda x: (x == 'Win').sum()),
            losses=('status', lambda x: (x == 'Loss').sum()),
            pushes=('status', lambda x: (x == 'Push').sum()),
            profit_units=('units_profit', 'sum'),
            expected_model_units=('ev_model', 'sum'),
            expected_blended_units=('ev_blended', 'sum'),
        ).reset_index()
        summary['roi'] = summary['profit_units'] / summary['plays']
        summary.to_csv(args.ledger.parent / 'nba_ev_performance_summary.csv', index=False)

        # Sort first, then cumsum: running profit follows chronological capture order.
        completed['_capture_sort'] = pd.to_datetime(
            completed['captured_at_utc'], utc=True, errors='coerce'
        )
        completed = completed.sort_values(
            ['_capture_sort', 'play_id'], kind='stable', na_position='last'
        ).drop(columns=['_capture_sort'])
        completed['running_profit_units'] = completed['units_profit'].cumsum()
        completed.to_csv(args.ledger.parent / 'nba_ev_settled_plays.csv', index=False)
        print(
            f"Settled {settled} new plays across full ledger; "
            f"{len(completed)} qualifying completed plays (blended odds < {BLENDED_ODDS_MAX:.2f}, blended EV > {BLENDED_EV_MIN:.0%}); "
            f"profit {completed['units_profit'].sum():+.2f} units; "
            f"ROI {completed['units_profit'].mean():+.1%}"
        )
    else:
        # Prevent stale reports from showing results outside the filter.
        completed['running_profit_units'] = pd.Series(dtype='float64')
        completed.to_csv(args.ledger.parent / 'nba_ev_settled_plays.csv', index=False)
        pd.DataFrame(columns=[
            'prop market', 'plays', 'wins', 'losses', 'pushes', 'profit_units',
            'expected_model_units', 'expected_blended_units', 'roi'
        ]).to_csv(args.ledger.parent / 'nba_ev_performance_summary.csv', index=False)
        print(f'No completed plays with blended odds < {BLENDED_ODDS_MAX:.2f} and blended EV > {BLENDED_EV_MIN:.0%}')
    print(f'Unresolved/ambiguous: {(ledger.status=="Pending").sum()} / {ambiguous}. Ledger: {args.ledger}')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action',choices=['capture','prepare','settle','update'])
    p.add_argument('--best',type=Path,default=Path('ev_results/nba_best_blended_bets.csv'))
    p.add_argument('--books',type=Path,default=Path('sportsbook_odds/nba_all_sportsbook_odds.csv'))
    p.add_argument('--actuals',type=Path,default=Path('processed_nba_props/nba_latest_player_box_scores.csv'))
    p.add_argument('--ledger',type=Path,default=Path('ev_results/nba_ev_play_ledger.csv'))
    p.add_argument('--source-actuals',type=Path,help='Full raw Kaggle player-statistics CSV/parquet')
    p.add_argument('--lineup-date',help='Game date YYYY-MM-DD to extract from gameDateTimeEst')
    args=p.parse_args()
    if args.action == 'prepare' or (args.action in ('settle','update') and args.source_actuals is not None):
        prepare_actuals(args.source_actuals, args.actuals, args.lineup_date)
    if args.action in ('capture','update'): capture(args)
    if args.action in ('settle','update'): settle(args)
if __name__=='__main__':main()
