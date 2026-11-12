"""Local NBA pipeline controller: preprocessing, training, predictions and EV.

Examples:
  python automate_nba.py daily --kaggle
  python automate_nba.py daily --players data/PlayerStatisticsExtended.csv
  python automate_nba.py preprocess --players data/PlayerStatisticsExtended.csv --lineup-date 2026-10-23
  python automate_nba.py train --players data/PlayerStatisticsExtended.csv
  python automate_nba.py odds

'daily' = preprocess -> settle prior captured bets -> predict -> scrape -> EV -> capture.
'full'  = preprocess -> settle prior captured bets -> retrain -> predict -> scrape -> EV -> capture.
'odds'  = scrape -> EV -> capture, using previously generated predictions.
Settlement uses freshly exported latest box scores, NEVER current sportsbook odds as actuals.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PROPS = ('points', 'rebounds', 'assists')


def run(script: str, *arguments: str) -> None:
    path = ROOT / script
    if not path.is_file():
        raise FileNotFoundError(f'Missing script: {path}')
    cmd = [sys.executable, str(path), *arguments]
    logging.info('RUN %s', subprocess.list2cmdline(cmd))
    # Capture child output in the controller log and display it live.
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          encoding='utf-8', errors='replace', bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            logging.info('[%s] %s', script, line.rstrip())
        result = process.wait()
    if result:
        raise RuntimeError(f'{script} failed (exit code {result})')


def require_fresh(path: str, started: float) -> None:
    file = ROOT / path
    if not file.is_file() or file.stat().st_size == 0 or file.stat().st_mtime < started - 2:
        raise RuntimeError(f'Missing, empty or stale output: {file}')


def preprocess(args: argparse.Namespace) -> None:
    if not args.kaggle and not args.players:
        raise ValueError('Preprocessing needs --kaggle or --players PATH.')

    flags = ['--output-dir', 'processed_nba_props']
    if args.kaggle:
        flags.append('--kaggle')
    else:
        flags.extend(['--players', str(Path(args.players).expanduser().resolve())])

    if args.lineup_date:
        flags.extend(['--lineup-date', args.lineup_date])

    if args.skip_lineups:
        flags.append('--skip-lineups')
    else:
        lineup_date = (
            datetime.strptime(args.lineup_date, '%Y-%m-%d').date()
            if args.lineup_date
            else datetime.now(ZoneInfo('America/New_York')).date() + timedelta(days=1)
        )
        # Current Eastern timestamp (automatically EDT/EST), not a future
        # 17:00 timestamp on the prediction date.
        injury_report_et = (
            args.injury_report_et
            or f'{lineup_date:%Y-%m-%d}T17:30'
            ## datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%dT%H:%M')
        )
        logging.info('Prediction lineup date (ET): %s', lineup_date)
        flags.extend(['--injury-report-et', injury_report_et])
        logging.info('NBA injury report timestamp (ET): %s', injury_report_et)

    started = datetime.now().timestamp()
    run('preprocess_nba_props_fuzzy_injuries.py', *flags)
    for prop in PROPS:
        require_fresh(f'processed_nba_props/nba_{prop}_train.parquet', started)
        require_fresh(f'processed_nba_props/nba_{prop}_new.parquet', started)
        require_fresh(f'processed_nba_props/nba_{prop}_features.txt', started)
    if args.skip_lineups:
        logging.warning('Lineups skipped: *_new.parquet may contain no upcoming prediction rows.')



def settle_previous_plays(preprocess_started: float) -> None:
    """Settle existing snapshots after preprocessing refreshes completed box scores.

    Never capture before settlement: new sportsbook prices represent upcoming
    games and must not be paired with yesterday's box scores.
    """
    ledger = ROOT / 'ev_results/nba_ev_play_ledger.csv'
    actuals = ROOT / 'processed_nba_props/nba_latest_player_box_scores.csv'
    if not ledger.is_file():
        logging.info('No existing EV ledger; skipping settlement (first run).')
        return
    if not actuals.is_file() or actuals.stat().st_size == 0:
        logging.warning('No latest completed box scores; skipping settlement: %s', actuals)
        return
    if actuals.stat().st_mtime < preprocess_started - 2:
        logging.warning('Latest box scores were not refreshed by this preprocessing run; '
                        'skipping settlement to avoid using stale actuals: %s', actuals)
        return
    run('track_nba_ev.py', 'settle', '--actuals', str(actuals))


def train() -> None:
    started = datetime.now().timestamp()
    run('retrain_nba_models.py')
    for prop in PROPS:
        require_fresh(f'models/nba_props_full/{prop}/{prop}_mean_lightgbm.joblib', started)
        require_fresh(f'models/nba_props_full/{prop}/model_config.json', started)


def predictions() -> None:
    started = datetime.now().timestamp()
    run('predict_nba_props.py')
    require_fresh('predictions/nba_all_line_odds.csv', started)


def odds(args: argparse.Namespace) -> None:
    started = datetime.now().timestamp()
    run('scrape_nba_sportsbooks.py', '--books', 'all', '--days-ahead', str(args.days_ahead))
    require_fresh('sportsbook_odds/nba_all_sportsbook_odds.csv', started)
    started = datetime.now().timestamp()
    flags = ['--model-weight', str(args.model_weight), '--min-other-books',
             str(args.min_other_books), '--min-ev', str(args.min_ev)]
    if args.require_agreement:
        flags.append('--require-agreement')
    run('compare_nba_ev.py', *flags)
    require_fresh('ev_results/nba_all_ev_comparison.csv', started)
    require_fresh('ev_results/nba_best_blended_bets.csv', started)
    run('track_nba_ev.py', 'capture')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', choices=['preprocess', 'train', 'predictions', 'odds', 'daily', 'full'])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--players', help='Local historical PlayerStatisticsExtended.csv')
    source.add_argument('--kaggle', action='store_true', help='Download historical source through kagglehub')
    parser.add_argument('--lineup-date', help='NBA calendar date YYYY-MM-DD; default is tomorrow in New York (preprocessor default)')
    parser.add_argument('--injury-report-et', default=None, help='Injury report ET YYYY-MM-DDTHH:MM; default current Eastern date/time (EDT/EST)')
    parser.add_argument('--skip-lineups', action='store_true', help='Skip live lineups (usually only useful for historical training)')
    parser.add_argument('--days-ahead', type=int, default=14)
    parser.add_argument('--model-weight', type=float, default=0.5)
    parser.add_argument('--min-other-books', type=int, default=2)
    parser.add_argument('--min-ev', type=float, default=0)
    parser.add_argument('--require-agreement', action='store_true')
    args = parser.parse_args()
    if args.job in ('preprocess', 'train', 'daily', 'full') and not (args.kaggle or args.players):
        parser.error(f'{args.job} requires --players PATH or --kaggle')
    if args.job in ('daily', 'full') and args.skip_lineups:
        parser.error('Cannot use --skip-lineups with daily/full because predictions need lineup rows')
    if not 0 <= args.model_weight <= 1:
        parser.error('--model-weight must be between 0 and 1')

    (ROOT / 'logs').mkdir(exist_ok=True)
    log = ROOT / 'logs' / f'nba_{args.job}_{datetime.now():%Y%m%d_%H%M%S}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    try:
        if args.job in ('preprocess', 'train', 'daily', 'full'):
            preprocessing_started = datetime.now().timestamp()
            preprocess(args)
            settle_previous_plays(preprocessing_started)
        if args.job in ('train', 'full'):
            train()
        if args.job in ('predictions', 'daily', 'full'):
            predictions()
        if args.job in ('odds', 'daily', 'full'):
            odds(args)
        logging.info('Completed %s successfully', args.job)
    except Exception:
        logging.exception('Pipeline failed; subsequent stages were not run')
        sys.exit(1)


if __name__ == '__main__':
    main()
