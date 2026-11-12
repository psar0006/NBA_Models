"""Compare NBA NB2 model fair odds with sportsbook prices and a simple odds blend.

Inputs: sportsbook_odds/nba_all_sportsbook_odds.csv and predictions/nba_all_line_odds.csv
Run: python compare_nba_ev.py --min-other-books 2 --min-ev 0

EV = quoted_decimal_odds * estimated_probability - 1.
The simple blended odds are the arithmetic mean of the model fair odds and
each distinct sportsbook price for the same event/player/stat/side/line.
The quoted sportsbook is INCLUDED in that blend, as requested.
Legacy MPTO / LOO metrics remain internal and are not exported.
"""
from __future__ import annotations
import argparse
import logging
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

LOG = logging.getLogger('nba_ev')
STATS = {'pts':'points','point':'points','points':'points','reb':'rebounds','rebs':'rebounds','rebound':'rebounds','rebounds':'rebounds','ast':'assists','assist':'assists','assists':'assists'}
TEAMS = [
('ATL','atlanta hawks'),('BOS','boston celtics'),('BKN','brooklyn nets'),('CHA','charlotte hornets'),('CHI','chicago bulls'),('CLE','cleveland cavaliers'),('DAL','dallas mavericks'),('DEN','denver nuggets'),('DET','detroit pistons'),('GSW','golden state warriors'),('HOU','houston rockets'),('IND','indiana pacers'),('LAC','los angeles clippers'),('LAL','los angeles lakers'),('MEM','memphis grizzlies'),('MIA','miami heat'),('MIL','milwaukee bucks'),('MIN','minnesota timberwolves'),('NOP','new orleans pelicans'),('NYK','new york knicks'),('OKC','oklahoma city thunder'),('ORL','orlando magic'),('PHI','philadelphia 76ers'),('PHX','phoenix suns'),('POR','portland trail blazers'),('SAC','sacramento kings'),('SAS','san antonio spurs'),('TOR','toronto raptors'),('UTA','utah jazz'),('WAS','washington wizards')]
ALIASES = {}
for abbr, name in TEAMS:
    ALIASES[abbr.lower()] = name
    ALIASES[name] = name
ALIASES.update({'la clippers':'los angeles clippers','la lakers':'los angeles lakers','ny knicks':'new york knicks','golden state':'golden state warriors','oklahoma city':'oklahoma city thunder','brooklyn':'brooklyn nets','new jersey nets':'brooklyn nets','san antonio':'san antonio spurs','new orleans':'new orleans pelicans','phoenix':'phoenix suns','portland':'portland trail blazers','philadelphia':'philadelphia 76ers','sixers':'philadelphia 76ers','utah':'utah jazz','washington':'washington wizards','los angeles clippers':'los angeles clippers'})
SUFFIX = re.compile(r'\s+(?:jr|sr|ii|iii|iv)$')


def norm(s):
    if pd.isna(s): return ''
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()
    s=re.sub(r'[^a-z0-9 ]+',' ',s)
    return ' '.join(s.split())


def player_key(s):
    return SUFFIX.sub('',norm(s))


def event_key(s):
    if pd.isna(s): return None
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()
    bits=re.split(r'\s+(?:vs?\.?|versus|@|at)\s+|\s+[-–—]\s+',s,maxsplit=1)
    if len(bits)!=2: return None
    a,b=[ALIASES.get(norm(v),norm(v)) for v in bits]
    if not a or not b or a==b: return None
    return ' | '.join(sorted((a,b)))


def coalesce_cols(df, names, default=np.nan):
    result=pd.Series(np.nan,index=df.index,dtype='object')
    for col in names:
        if col in df: result=result.combine_first(df[col])
    return result.fillna(default) if not pd.isna(default) else result


def read_csv(path):
    if not path.exists(): raise FileNotFoundError(f'Missing {path}. Run the sportsbook/prediction scripts first.')
    return pd.read_csv(path,low_memory=False)


def prepare_books(df):
    needed={'bookmaker','player_name','stat','odds'}
    if not needed.issubset(df.columns): raise ValueError(f'Bookmaker file missing {needed-set(df.columns)}')
    x=df.copy()
    x['bookmaker']=x['bookmaker'].astype('string').str.strip()
    x['player_name']=x['player_name'].astype('string').str.replace(r'\s+To\s+Get\s*$','',regex=True,flags=re.I).str.strip()
    x['player_key']=x['player_name'].map(player_key)
    x['stat']=x['stat'].astype('string').str.lower().str.strip().map(STATS)
    x['bet_type']=coalesce_cols(x,['bet_type'],'').astype('string').str.lower().str.strip().replace({'higher':'over','lower':'under','yes':'over','no':'under'})
    x['market_category']=coalesce_cols(x,['market_category'],'standard').astype('string').str.lower().str.strip()
    x.loc[x['bet_type'].isin(['','<NA>']) & x['market_category'].isin(['alternative','alt','ladder']),'bet_type']='over'
    x['raw_line']=pd.to_numeric(coalesce_cols(x,['line','raw_line']),errors='coerce')
    x['line']=pd.to_numeric(coalesce_cols(x,['equivalent_over_line']),errors='coerce')
    missing=x['line'].isna()
    x.loc[missing,'line']=x.loc[missing,'raw_line']-np.where(x.loc[missing,'market_category'].isin(['alternative','alt','ladder']) & x.loc[missing,'bet_type'].eq('over'),.5,0)
    x['line']=x['line'].round(3)
    x['odds']=pd.to_numeric(x['odds'],errors='coerce')
    x['event_name']=coalesce_cols(x,['event_name','match_name']).astype('string')
    x['event_key']=x['event_name'].map(event_key)
    x['start_time']=coalesce_cols(x,['start_time','advertised_start']).astype('string')
    x['start_utc']=pd.to_datetime(x['start_time'],utc=True,errors='coerce',format='mixed')
    valid=(x['bookmaker'].notna() & x['player_key'].ne('') & x['stat'].notna() & x['bet_type'].isin(['over','under']) & x['line'].ge(0) & np.isfinite(x['line']) & x['odds'].gt(1) & np.isfinite(x['odds']))
    rejected=x.loc[~valid].copy()
    x=x.loc[valid].copy()
    # One bookmaker is one vote per market, irrespective of duplicate API selections.
    keys=['bookmaker','event_key','player_key','stat','bet_type','line']
    # Unknown event names are not merged across games; use original event label as scoped key.
    x['book_event_key']=x['event_key'].fillna('raw:'+x['event_name'].fillna('').map(norm))
    keys[1]='book_event_key'
    x=x.sort_values('odds',ascending=False).drop_duplicates(keys,keep='first')
    return x.reset_index(drop=True),rejected


def prepare_model(df):
    x=df.copy()
    if 'player_name' not in x and 'playerName' in x: x['player_name']=x['playerName']
    needed={'player_name','stat','handicap','over_probability','under_probability'}
    if not needed.issubset(x.columns): raise ValueError(f'Model file missing {needed-set(x.columns)}')
    x['player_key']=x['player_name'].map(player_key)
    x['stat']=x['stat'].astype('string').str.lower().str.strip().map(STATS)
    x['line']=pd.to_numeric(x['handicap'],errors='coerce').round(3)
    x['over_probability']=pd.to_numeric(x['over_probability'],errors='coerce')
    x['under_probability']=pd.to_numeric(x['under_probability'],errors='coerce')
    x['predicted_mean']=pd.to_numeric(coalesce_cols(x,['predicted_mean']),errors='coerce')
    x['gameId']=coalesce_cols(x,['gameId']).astype('string')
    # gameDateTimeEst is usually US Eastern local time, not UTC.
    x['model_date']=pd.to_datetime(coalesce_cols(x,['gameDateTimeEst']),errors='coerce').dt.date
    x=x[x['player_key'].ne('') & x['stat'].notna() & x['line'].notna()].copy()
    x=x[x['over_probability'].between(0,1) & x['under_probability'].between(0,1)]
    x=x.drop_duplicates(['gameId','player_key','stat','line'],keep='first')
    a=x.assign(bet_type='over',model_probability=x['over_probability'])
    b=x.assign(bet_type='under',model_probability=x['under_probability'])
    return pd.concat([a,b],ignore_index=True)[['player_key','player_name','stat','line','bet_type','model_probability','predicted_mean','gameId','model_date']]


def join_model(books,models):
    """Only attach a model when player+market maps to exactly one game.

    Date filter: sportsbook UTC date may be a day later than NBA Eastern game date.
    Never match by player alone if multiple game candidates remain.
    """
    x=books.copy().reset_index(drop=True)
    x['_rowid']=np.arange(len(x))
    merged=x.merge(models,on=['player_key','stat','bet_type','line'],how='left',suffixes=('','_model'))
    book_day=merged['start_utc'].dt.date
    # allow 0/1 day offset (UTC vs Eastern); avoid distant matchup misjoins
    offset=(pd.to_datetime(book_day)-pd.to_datetime(merged['model_date'])).dt.days
    admissible=merged['model_date'].isna() | book_day.isna() | offset.isin([0,1])
    merged=merged[admissible].copy()
    # Reject ambiguity, including when a book's date is missing and player has 2 games.
    counts=merged.groupby('_rowid',dropna=False)['gameId'].transform(lambda s:s.dropna().nunique())
    merged=merged[(counts<=1)].copy()
    merged=merged.sort_values('_rowid').drop_duplicates('_rowid',keep='first')
    return x.merge(merged[['_rowid','model_probability','predicted_mean','gameId','model_date']],on='_rowid',how='left').drop(columns='_rowid')


def calc_ev(x,model_weight,min_other_books,margin):
    """Compute bookmaker MPTO consensus and model+market blend (LOO)."""
    x=x.copy()
    x['implied_probability']=1/x['odds']
    # Original R notebook's assumed two-outcome margin correction.
    for m in (8,15,25):
        p=x['implied_probability']-m/200
        x[f'fair_prob_{m}']=p.where(p.gt(0)&p.lt(1))
    group=['book_event_key','player_key','stat','bet_type','line']
    g=x.groupby(group,dropna=False,sort=False)
    x['n_bookmakers']=g['bookmaker'].transform('nunique')
    x['avg_odds']=g['odds'].transform('mean')
    x['median_odds']=g['odds'].transform('median')
    x['min_odds']=g['odds'].transform('min')
    x['max_odds']=g['odds'].transform('max')
    x['loo_n_bookmakers']=x['n_bookmakers']-1
    x['loo_avg_odds']=(g['odds'].transform('sum')-x['odds'])/x['loo_n_bookmakers'].replace(0,np.nan)
    for m in (8,15,25):
        col=f'fair_prob_{m}'
        count=g[col].transform('count')
        total=g[col].transform('sum')
        is_valid=x[col].notna()
        loo_n=count-is_valid.astype(int)
        x[f'loo_n_valid_{m}']=loo_n
        x[f'consensus_prob_{m}']=total/count.replace(0,np.nan)
        x[f'loo_prob_{m}']=(total-x[col].fillna(0))/loo_n.replace(0,np.nan)
        x[f'consensus_odds_{m}']=1/x[f'consensus_prob_{m}']
        x[f'loo_odds_{m}']=1/x[f'loo_prob_{m}']
        x[f'ev_full_{m}']=x['odds']*x[f'consensus_prob_{m}']-1
        x[f'ev_loo_{m}']=x['odds']*x[f'loo_prob_{m}']-1
    x['model_fair_odds']=1/x['model_probability']
    x['ev_model']=x['odds']*x['model_probability']-1
    loo=x[f'loo_prob_{margin}']
    model=x['model_probability']
    # Blend only when BOTH components exist and min other books met.
    both=model.notna() & loo.notna() & x[f'loo_n_valid_{margin}'].ge(min_other_books)
    x['blended_probability']=np.where(both,model_weight*model+(1-model_weight)*loo,np.nan)
    x['blended_fair_odds']=1/x['blended_probability']
    x['ev_blended']=x['odds']*x['blended_probability']-1
    x['positive_ev_model']=x['ev_model']>0
    x['positive_ev_loo']=x[f'ev_loo_{margin}']>0
    x['positive_ev_blended']=x['ev_blended']>0
    x['model_market_agree']=x['positive_ev_model'] & x['positive_ev_loo']
    # Requested odds-level blend: one model fair price plus one price per
    # bookmaker, equally weighted. The candidate bookmaker is included.
    # The group was already deduplicated to one selection per bookmaker.
    valid_model=np.isfinite(x['model_fair_odds']) & x['model_fair_odds'].gt(1)
    x['blended_fair_odds']=(g['odds'].transform('sum')+x['model_fair_odds'])/(x['n_bookmakers']+1)
    x.loc[~valid_model,'blended_fair_odds']=np.nan
    x['blended_probability']=1/x['blended_fair_odds']
    x['ev_blended']=x['odds']/x['blended_fair_odds']-1
    x['positive_ev_blended']=x['ev_blended']>0
    return x


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--books-file',type=Path,default=Path('sportsbook_odds/nba_all_sportsbook_odds.csv'))
    ap.add_argument('--model-file',type=Path,default=Path('predictions/nba_all_line_odds.csv'))
    ap.add_argument('--output-dir',type=Path,default=Path('ev_results'))
    ap.add_argument('--model-weight',type=float,default=.5,help='Legacy option (simple odds blend always uses equal weights)')
    ap.add_argument('--margin',type=int,choices=[8,15,25],default=8,help='Assumed market margin for MPTO consensus')
    ap.add_argument('--min-other-books',type=int,default=2)
    ap.add_argument('--min-ev',type=float,default=0,help='Minimum EV as decimal, e.g. .05 for 5%%')
    ap.add_argument('--require-agreement',action='store_true',help='Require positive model EV as well as positive blended EV')
    args=ap.parse_args()
    if not 0<=args.model_weight<=1: ap.error('--model-weight must be in [0,1]')
    if args.min_other_books<0: ap.error('--min-other-books must be >= 0')
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    books,rejected=prepare_books(read_csv(args.books_file))
    models=prepare_model(read_csv(args.model_file))
    LOG.info('Valid bookmaker selections: %s; rejected: %s; model lines: %s',len(books),len(rejected),len(models))
    if books.empty: raise SystemExit('No valid bookmaker selections to compare.')
    combined=join_model(books,models)
    LOG.info('Model matched: %s / %s',combined.model_probability.notna().sum(),len(combined))
    scored=calc_ev(combined,args.model_weight,args.min_other_books,args.margin)
    # Highest blended EV for this game, player, prop, direction and exact line.
    # Tied highest values are all marked True; missing EV is never highest.
    market_keys=['book_event_key','player_key','stat','bet_type','line']
    max_ev=scored.groupby(market_keys,dropna=False)['ev_blended'].transform('max')
    scored['highest EV for market and line']=(
        scored['ev_blended'].notna() &
        np.isclose(scored['ev_blended'],max_ev,rtol=1e-10,atol=1e-12)
    )
    # The exported view is intentionally limited to the eleven requested fields.
    # Prop market includes direction so Over/Under cannot be confused.
    def presentation(frame):
        view=frame.copy()
        view['player']=view['player_name']
        view['prop market']=view['stat'].str.title()+' - '+view['bet_type'].str.title()
        view['predicted value from model']=view['predicted_mean']
        view['model odds']=view['model_fair_odds']
        view['blended odds']=view['blended_fair_odds']
        view['sportsbookname']=view['bookmaker']
        view['sportsbook odds']=view['odds']
        view['EV of model vs sportsbook']=view['ev_model']
        view['EV of blended vs sportsbook']=view['ev_blended']
        view['highest EV for market and line']=view['highest EV for market and line']
        columns=['event_name','player','prop market','line',
                 'predicted value from model','model odds','blended odds',
                 'sportsbookname','sportsbook odds',
                 'EV of model vs sportsbook','EV of blended vs sportsbook',
                 'highest EV for market and line']
        return view[columns]

    model_positive=scored[scored['ev_model']>args.min_ev].sort_values('ev_model',ascending=False)
    # min-other-books now only affects the blended positive shortlist, not
    # the full comparison or the blend's mathematical definition.
    blend_mask=scored['ev_blended'].gt(args.min_ev) & scored['n_bookmakers'].ge(args.min_other_books+1)
    if args.require_agreement:
        blend_mask &= scored['ev_model'].gt(args.min_ev)
    blend_positive=scored[blend_mask].sort_values('ev_blended',ascending=False)
    best=blend_positive.sort_values(['ev_blended','odds'],ascending=False).drop_duplicates(
        ['book_event_key','player_key','stat','bet_type','line'],keep='first')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    exports=[
        ('nba_all_ev_comparison.csv',scored),
        ('nba_positive_ev_model.csv',model_positive),
        ('nba_positive_ev_blended.csv',blend_positive),
        ('nba_best_blended_bets.csv',best),
    ]
    for filename,frame in exports:
        output=presentation(frame)
        output.to_csv(args.output_dir/filename,index=False)
        LOG.info('%s: %s rows',filename,len(output))
    if len(best):
        print('\nTOP BLENDED POSITIVE EV CANDIDATES\n')
        print(presentation(best).head(20).to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    else:
        print('\nNo qualifying positive blended EV selections. Check model matches and bookmaker coverage.')

if __name__=='__main__': main()
