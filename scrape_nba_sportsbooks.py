"""NBA player props: Python ports of six user-supplied R sportsbook scrapers.

Run: python scrape_nba_sportsbooks.py --books all --days-ahead 14 --output-dir sportsbook_odds
Dependencies: requests pandas curl_cffi (TAB only)

Markets are one row per offered selection. `line` is the literal sportsbook
line (20 for 20+, 19.5 for O/U 19.5); `equivalent_over_line` is 19.5
for 20+. Does not attempt to circumvent access restrictions.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import re
import random
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger('nba_sportsbooks')
BOOKS = ('sportsbet', 'tabtouch', 'tab', 'betr', 'ladbrokes', 'pointsbet')
COLS = ['bookmaker','event_id','event_name','start_time','player_name','stat','market_category','bet_type','line','equivalent_over_line','odds','market_id','market_name','selection_id','selection_name','scraped_at']
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36'


def val(x, default=None):
    if x is None or isinstance(x,(dict,list)): return default
    s=str(x).strip()
    return s if s and s.lower() not in ('none','nan','null') else default

def num(x):
    try:
        z=float(x)
        return z if math.isfinite(z) else None
    except (TypeError,ValueError): return None

def objs(x):
    if isinstance(x,list): return [z for z in x if isinstance(z,dict)]
    if isinstance(x,dict): return [z for z in x.values() if isinstance(z,dict)]
    return []

def clean_name(x):
    return re.sub(r'\s+',' ',re.sub(r'\s*\([^()]*\)\s*$','',val(x,'') or '')).strip()

def parse_date(s):
    if not s: return None
    try: return datetime.fromisoformat(str(s).replace('Z','+00:00')).astimezone(timezone.utc)
    except (ValueError,TypeError): return None

def upcoming(s,days):
    d=parse_date(s)
    return d is None or datetime.now(timezone.utc) <= d <= datetime.now(timezone.utc)+timedelta(days=days)

def row(book,event_id,event_name,start,player,stat,category,side,line,odds,market_id=None,market_name=None,selection_id=None,selection_name=None,**extra):
    if not player or stat not in ('points','rebounds','assists') or side not in ('over','under') or category not in ('standard','alternative'): return None
    line,odds=num(line),num(odds)
    if line is None or line<0 or odds is None or odds<=1: return None
    result=dict(bookmaker=book,event_id=val(event_id),event_name=val(event_name),start_time=val(start),player_name=clean_name(player),stat=stat,market_category=category,bet_type=side,line=line,equivalent_over_line=line-0.5 if category=='alternative' and side=='over' else line,odds=odds,market_id=val(market_id),market_name=val(market_name),selection_id=val(selection_id),selection_name=val(selection_name),scraped_at=datetime.now(timezone.utc).isoformat())
    if not result['player_name']: return None
    result.update(extra)
    return result

class Client:
    def __init__(self,delay=.5,timeout=30):
        self.delay=delay; self.timeout=timeout
        self.s=requests.Session()
        self.s.headers.update({'User-Agent':UA,'Accept':'application/json, text/plain, */*','Accept-Language':'en-AU,en;q=0.9'})
        retries=Retry(total=2,backoff_factor=1,status_forcelist=[408,429,500,502,503,504],allowed_methods=['GET'],respect_retry_after_header=True)
        self.s.mount('https://',HTTPAdapter(max_retries=retries))
    def get(self,url,params=None,headers=None):
        LOG.debug('GET %s %s',url,params)
        response=self.s.get(url,params=params,headers=headers,timeout=self.timeout)
        response.raise_for_status()
        result=response.json()
        if isinstance(result,dict) and result.get('errors'): raise ValueError(f'API errors: {result["errors"]}')
        return result
    def wait(self):
        if self.delay>0: time.sleep(self.delay)

def sportsbet(c,days):
    base='https://www.sportsbet.com.au/apigw/sportsbook-sports/Sportsbook/Sports'
    headers={'apptoken':'cxp-desktop-web','channel':'cxp','Referer':'https://www.sportsbet.com.au/'}
    data=c.get(base+'/Competitions/6927',headers=headers)
    events=objs(data.get('events')) if isinstance(data,dict) else []
    out=[]
    for e in events:
        eid=val(e.get('id')); name=val(e.get('name')); start=val(e.get('startTime') or e.get('startDate') or e.get('eventStartTime'))
        if not eid or not upcoming(start,days): continue
        try:
            payload=c.get(f'{base}/Events/{eid}/Markets',headers=headers)
            markets=objs(payload.get('markets')) if isinstance(payload,dict) else objs(payload)
            for m in markets:
                mn=val(m.get('name'),'')
                standard=re.fullmatch(r'(.+?)\s+-\s+(Points|Rebounds|Assists)',mn,re.I)
                altp=re.fullmatch(r'To Score (\d+)\+ Points',mn,re.I)
                altra=re.fullmatch(r'To Record (\d+)\+ (Rebounds|Assists)',mn,re.I)
                if not (standard or altp or altra): continue
                if val(m.get('statusCode')) not in (None,'A'): continue
                for s in objs(m.get('selections')):
                    if val(s.get('statusCode')) not in (None,'A'): continue
                    sn=val(s.get('name'),'')
                    if standard:
                        match=re.search(r'\b(Over|Under)\s*$',sn,re.I)
                        if not match: continue
                        stat=standard.group(2).lower(); player=standard.group(1); side=match.group(1).lower(); category='standard'; line=num(s.get('unformattedHandicap'))
                    else:
                        stat='points' if altp else altra.group(2).lower(); player=sn; side='over'; category='alternative'; line=float((altp or altra).group(1))
                    price=s.get('price') or {}
                    odds=price.get('winPrice') if isinstance(price,dict) else None
                    r=row('Sportsbet',eid,name,start,player,stat,category,side,line,odds,m.get('id'),mn,s.get('id'),sn)
                    if r: out.append(r)
        except (requests.RequestException,ValueError,TypeError) as ex: LOG.warning('Sportsbet event %s: %s',eid,ex)
        c.wait()
    return out

def tabtouch(c,days):
    base='https://oc-offering-api.kambicdn.com/offering/v2018/rwwawa'
    headers={'Origin':'https://www.tabtouch.com.au','Referer':'https://www.tabtouch.com.au/'}
    feed=c.get(base+'/listView/basketball/nba/all/all/matches',params={'lang':'en_AU','market':'AU','useCombined':'true','useCombinedLive':'true','includeParticipants':'true','category':'5905'},headers=headers)
    criteria={'2100058934':('points','Points Scored By The Player'),'2100058936':('rebounds','Rebounds By The Player'),'2100058937':('assists','Assists By The Player')}
    out=[]
    for entry in objs(feed.get('events')):
        e=entry.get('event') or {}
        if str(e.get('group','')).upper()!='NBA' or str(e.get('sport','')).upper()!='BASKETBALL' or 'MATCH' not in e.get('tags',[]): continue
        eid=val(e.get('id')); start=val(e.get('start'))
        if not eid or not upcoming(start,days): continue
        try:
            payload=c.get(base+'/betoffer/event/'+eid,params={'lang':'en_AU','market':'AU','includeParticipants':'true'},headers=headers)
            for offer in objs(payload.get('betOffers')):
                criterion=offer.get('criterion') or {}; cid=val(criterion.get('id'))
                if cid not in criteria: continue
                stat,expected=criteria[cid]; label=val(criterion.get('label'),'')
                m=re.fullmatch(r'(\d+)\+ '+re.escape(expected)+r' - Including Overtime',label,re.I)
                if not m or val((offer.get('betOfferType') or {}).get('id'))!='127' or criterion.get('lifetime')!='FULL_TIME_OVERTIME': continue
                threshold=float(m.group(1))
                for o in objs(offer.get('outcomes')):
                    if o.get('type')!='OT_YES' or o.get('status')!='OPEN': continue
                    line=num(o.get('line')); odds=num(o.get('odds'))
                    if line is None or line/1000!=threshold: continue
                    r=row('TabTouch',eid,e.get('name'),start,o.get('participant'),stat,'alternative','over',threshold,odds/1000 if odds is not None else None,offer.get('id'),label,o.get('id'),o.get('label'),player_participant_id=val(o.get('participantId')))
                    if r: out.append(r)
        except (requests.RequestException,ValueError,TypeError) as ex: LOG.warning('TabTouch event %s: %s',eid,ex)
        c.wait()
    return out

def walk_dicts(x):
    if isinstance(x,dict):
        yield x
        for v in x.values(): yield from walk_dicts(v)
    elif isinstance(x,list):
        for v in x: yield from walk_dicts(v)

def tab(c, days):
    """TAB NBA individual P/R/A ladders, adapted from the standalone TAB scraper.

    Returns the shared sportsbook row schema; curl_cffi is used only for TAB.
    """
    from curl_cffi import requests as cffi_requests

    base = 'https://api.beta.tab.com.au/v1/tab-info-service'
    competition = '/sports/Basketball/competitions/NBA'
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-AU,en;q=0.9',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'origin': 'https://www.tab.com.au',
        'referer': 'https://www.tab.com.au/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        # curl_cffi supplies browser-consistent User-Agent/client hints.
    }
    cookie = os.getenv('TAB_COOKIE', '').strip()
    if cookie:
        headers['cookie'] = cookie

    def make_session():
        return cffi_requests.Session(impersonate='chrome')

    session = make_session()

    def get_json(path, query=None, max_attempts=4):
        nonlocal session
        url = base + path
        params = {'jurisdiction': 'NSW'}
        params.update(query or {})
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                session.close()
                session = make_session()
            try:
                LOG.debug('TAB GET %s | %s | attempt %s/%s',
                          url, params, attempt, max_attempts)
                response = session.get(url, params=params, headers=headers,
                                       timeout=c.timeout, allow_redirects=True)
                LOG.debug('TAB HTTP %s | bytes=%s',
                          response.status_code, len(response.content))
                if response.ok:
                    if not response.content:
                        raise ValueError('TAB returned empty JSON')
                    return response.json()
                preview = response.text[:350]
                # An edge-access denial is not a transient server error.
                if response.status_code == 403 or 'errors.edgesuite.net' in preview:
                    raise PermissionError(f'TAB access denied (HTTP {response.status_code}): {preview}')
                if response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f'TAB HTTP {response.status_code}: {preview}')
                LOG.warning('TAB retryable HTTP %s [%s/%s]: %s',
                            response.status_code, attempt, max_attempts, preview)
            except PermissionError:
                raise
            except Exception as exc:
                LOG.warning('TAB transport/JSON error [%s/%s]: %s',
                            attempt, max_attempts, exc)
                if attempt == max_attempts:
                    raise RuntimeError(f'TAB request failed after {max_attempts} attempts') from exc
            if attempt == max_attempts:
                raise RuntimeError(f'TAB request failed after {max_attempts} attempts: {url}')
            time.sleep(2 ** (attempt - 1))

    matches = {}
    output = []
    seen = set()
    market_pattern = re.compile(
        r'^([0-9]+(?:\.[0-9]+)?)\s*\+\s*(Points|Rebounds|Assists)$', re.I
    )
    match_pattern = re.compile(r'\s+(?:v|vs\.?|@)\s+', re.I)

    try:
        payload = get_json(competition, {'numTopMarkets': 5})
        for obj in walk_dicts(payload):
            name = val(obj.get('name') or obj.get('matchName'))
            if not name or not match_pattern.search(name):
                continue
            start = val(obj.get('startTime') or obj.get('advertisedStartTime'))
            if upcoming(start, days):
                matches.setdefault(name, start)
        LOG.info('TAB upcoming NBA matches: %d', len(matches))

        for index, (name, start) in enumerate(matches.items(), 1):
            LOG.info('TAB match [%d/%d]: %s', index, len(matches), name)
            try:
                data = get_json(
                    competition + '/matches/' + quote(name, safe=''),
                    {'includeH2hMulti': 'false'},
                )
                markets = objs(data.get('markets'))
                eligible = 0
                for market in markets:
                    label = re.sub(r'\s+', ' ', val(market.get('betOption'), '')).strip()
                    found = market_pattern.fullmatch(label)
                    if not found:
                        continue
                    eligible += 1
                    line = float(found.group(1))
                    stat = found.group(2).lower()
                    for prop in objs(market.get('propositions')):
                        if (prop.get('isOpen') is not True or
                                str(prop.get('bettingStatus', '')).lower() != 'open'):
                            continue
                        selection_name = val(prop.get('name'))
                        market_id = val(market.get('id'))
                        selection_id = val(prop.get('id'))
                        key = (val(data.get('id')), market_id, selection_id)
                        if key in seen:
                            continue
                        record = row(
                            'TAB', data.get('id'), data.get('name') or name,
                            data.get('startTime') or start, selection_name, stat,
                            'alternative', 'over', line, prop.get('returnWin'),
                            market_id, label, selection_id, selection_name,
                        )
                        if record:
                            seen.add(key)
                            output.append(record)
                LOG.debug('TAB %s: %d markets, %d eligible P/R/A ladders',
                          name, len(markets), eligible)
            except PermissionError:
                raise  # Fail fast on access restrictions rather than silently skipping games.
            except Exception as exc:
                LOG.warning('TAB match %s: %s', name, exc)
            c.wait()
        LOG.info('TAB parsed %d selections', len(output))
        return output
    finally:
        session.close()

def betr(c,days):
    base='https://web20-api.bluebet.com.au'
    payload=c.get(base+'/MasterCategory',params={'EventTypeId':107,'CategoryId':39251,'WithLevelledMarkets':'true'})
    events={}
    def walk(x,category=None):
        if isinstance(x,dict):
            category=x.get('CategoryId',category)
            if x.get('MasterEventId') is not None and str(category)=='39251':
                eid=val(x.get('MasterEventId'))
                if eid: events[eid]=x
            for v in x.values(): walk(v,category)
        elif isinstance(x,list):
            for v in x: walk(v,category)
    walk(payload)
    groups={'assists':'C18946','rebounds':'C18947','points':'C16147'}
    out=[]
    for eid,event in events.items():
        start=val(event.get('MinAdvertisedStartTime') or event.get('AdvertisedStartTime'))
        if not upcoming(start,days): continue
        for stat,group in groups.items():
            try:
                data=c.get(base+'/MasterEventGrid',params={'MasterEventId':eid,'GroupTypeCode':group})
                master=data.get('MasterEvent') or event
                if val(master.get('CategoryId')) not in (None,'39251'): continue
                for ev in objs(data.get('Events')):
                    player=clean_name(ev.get('DisplayEventName') or ev.get('EventName'))
                    player=re.sub(r'^(?:player\s+)?(?:points|rebounds|assists)(?:\s+(?:over|under))?\s*[-:]\s*','',player,flags=re.I)
                    for o in objs(ev.get('Outcomes')):
                        sn=val(o.get('OutcomeName'),'')
                        pattern={'points':'points?|pts','rebounds':'rebounds?|rebs','assists':'assists?|asts'}[stat]
                        plus=re.search(r'(\d+(?:\.\d+)?)\s*\+\s*(?:'+pattern+r')?\s*$',sn,re.I)
                        ou=re.search(r'\b(Over|Under)\b\s*(\d+(?:\.\d+)?)?',sn,re.I)
                        if plus: category='alternative'; side='over'; line=plus.group(1)
                        elif ou:
                            category='standard'; side=ou.group(1).lower(); line=ou.group(2)
                            if line is None:
                                line=next((num(v) for v in [o.get('Handicap'),o.get('Line'),o.get('Points'),ev.get('Handicap'),ev.get('Line'),ev.get('Points')] if num(v) is not None),None)
                        else: continue
                        odds=o.get('Price') or o.get('DecimalPrice') or o.get('Odds')
                        r=row('Betr',eid,master.get('MasterEventName'),start,player,stat,category,side,line,odds,ev.get('EventId'),ev.get('GroupName'),o.get('OutcomeId'),sn,request_group_code=group)
                        if r: out.append(r)
            except (requests.RequestException,ValueError,TypeError) as ex: LOG.warning('Betr %s %s: %s',eid,stat,ex)
            c.wait()
    return out

def ladbrokes(c,days):
    base='https://api.ladbrokes.com.au'
    hash_='2fd0c21755f04cc22957f8549a2906a06d972d708107be7392e715257f8090e2'
    headers={'Referer':'https://www.ladbrokes.com.au/','graphql-client-build':'ad9ddba7628a517168a04230dbdbaa0c71dc53dd','graphql-client-name':'sportsbook','graphql-client-version':'release/cup-2026.4','Accept':'application/graphql-response+json, application/json'}
    variables={'category':'BASKETBALL','regionSlug':'usa','competitionSlug':'nba','statuses':['OPEN','LIVE'],'excludeCategoryIds':[],'includeLeagues':False,'includeUpcomingEvents':True,'upcomingEventsGroupBy':'UNSPECIFIED','includeFutures':True,'futuresGroupBy':'LEAGUE'}
    payload=c.get(base+'/gql/router',params={'extensions':json.dumps({'persistedQuery':{'version':1,'sha256Hash':hash_}},separators=(',',':')),'operationName':'SportingCompetitionScreen','variables':json.dumps(variables,separators=(',',':'))},headers=headers)
    events=objs((((payload.get('data') or {}).get('upcomingEvents') or {}).get('events') or {}).get('nodes'))
    out=[]; device=str(uuid.uuid4())
    # Ladbrokes EventCard can return HTTP 500 when requested in quick succession.
    # Pace requests *before* sending them, including the first event.
    for e in events:
        eid=val(e.get('id'))
        if not eid: continue
        eid=eid.removeprefix('SportingEvent:'); start=val(e.get('advertisedStart'))
        if not upcoming(start,days): continue
        c.wait()
        try:
            # Mirror the working R httr request: explicit headers, stable device ID,
            # and manually retry the EventCard response instead of urllib3 status retries.
            event_headers={
                'Accept':'*/*',
                'accept-language':'en-GB,en-US;q=0.9,en;q=0.8',
                'content-type':'application/json',
                'device-id':device,
                'sec-fetch-dest':'empty',
                'sec-fetch-mode':'cors',
                'sec-fetch-site':'same-site',
                'Referer':'https://www.ladbrokes.com.au/',
                'Origin':'https://www.ladbrokes.com.au',
                'User-Agent':UA,
            }
            data=None
            for attempt in range(1,4):
                try:
                    # Requests directly, bypassing the shared automatic 500 retries.
                    response=c.s.get(base+'/v2/sport/EventCard',
                                     params={'id':eid},headers=event_headers,
                                     timeout=c.timeout)
                    if response.ok and response.content:
                        data=response.json()
                        break
                    status=response.status_code
                    if status not in (408,429,500,502,503,504):
                        response.raise_for_status()
                        raise ValueError(f'HTTP {status}: {response.text[:350]}')
                    LOG.warning('Ladbrokes event %s attempt %s/3 HTTP %s',
                                eid,attempt,status)
                except (requests.RequestException,ValueError) as exc:
                    if attempt==3:
                        raise
                    LOG.warning('Ladbrokes event %s attempt %s/3: %s',
                                eid,attempt,exc)
                if attempt<3:
                    time.sleep(1.5*2**(attempt-1)+random.uniform(0,.5))
            if data is None:
                raise ValueError(f'EventCard returned no valid JSON after 3 attempts: {eid}')
            prices={}
            for key,p in (data.get('prices') or {}).items():
                odds_obj=p.get('odds') or {} if isinstance(p,dict) else {}
                odds=None
                if isinstance(odds_obj,dict):
                    n=num(odds_obj.get('numerator')); d=num(odds_obj.get('denominator'))
                    if n is not None and d and d>0: odds=1+n/d
                    else: odds=next((num(odds_obj.get(k)) for k in ('decimal','decimal_odds','price') if num(odds_obj.get(k)) is not None),None)
                if odds is None and isinstance(p,dict): odds=next((num(p.get(k)) for k in ('decimal','decimal_odds','price') if num(p.get(k)) is not None),None)
                prices.setdefault(str(key).split(':')[0],set()).add(odds)
            resolved={k:next(iter(v)) for k,v in prices.items() if len(v)==1}
            markets={}
            for key,m in (data.get('markets') or {}).items():
                if m.get('visible') is False: continue
                markets[val(m.get('id'),str(key))]=m
            for key,entrant in (data.get('entrants') or {}).items():
                if entrant.get('visible') is False: continue
                sid=val(entrant.get('id'),str(key)); mid=val(entrant.get('market_id'))
                m=markets.get(mid)
                if not m: continue
                mn=val(m.get('name'),''); sn=val(entrant.get('name'),'')
                std=re.fullmatch(r'Player (Points|Rebounds|Assists) O/U\s*-\s*(.+)',mn,re.I)
                alt=re.fullmatch(r'To (?:Score|Have)\s+(\d+(?:\.\d+)?)\+\s+(Points|Rebounds|Assists)',mn,re.I)
                if std:
                    side=sn.strip().lower()
                    if side not in ('over','under'): continue
                    stat=std.group(1).lower(); player=std.group(2); line=m.get('handicap'); category='standard'
                elif alt:
                    side='over'; stat=alt.group(2).lower(); player=sn; line=alt.group(1); category='alternative'
                else: continue
                r=row('Ladbrokes',eid,e.get('name'),start,player,stat,category,side,line,resolved.get(sid),mid,mn,sid,sn)
                if r: out.append(r)
        except (requests.RequestException,ValueError,TypeError) as ex: LOG.warning('Ladbrokes event %s: %s (skipping event)',eid,ex)
    return out

def pointsbet(c,days):
    base='https://api.au.pointsbet.com/api/mes/v3/events'
    headers={'Origin':'https://pointsbet.com.au','Referer':'https://pointsbet.com.au/'}
    out=[]
    for page in range(1,4):
        payload=c.get(base+'/featured/competition/7176',params={'page':page},headers=headers)
        events=objs(payload.get('events'))
        if not events: break
        for e in events:
            eid=val(e.get('key') or e.get('eventKey'))
            if not eid: continue
            try:
                data=c.get(base+'/'+eid,headers=headers)
                start=val(data.get('startsAt') or data.get('advertisedStartTime'))
                if not upcoming(start,days): continue
                for mk in ('fixedOddsMarkets','specialFixedOddsMarkets','markets'):
                    for m in objs(data.get(mk)):
                        mn=val(m.get('name'),''); context=val(m.get('eventClass'),mn)
                        if re.search(r'Points\s*\+\s*Rebounds|Points\s*\+\s*Assists|Rebounds\s*\+\s*Assists|\bPRA\b|Double\s*Double|Triple\s*Double|Team Total|\b(?:[1-4](?:st|nd|rd|th) Quarter|First Half|Second Half)\b',mn,re.I): continue
                        kinds=[s for s,p in [('points',r'\bpoints?\b|\bpts\b|\bto score\b|\bscorer\b'),('rebounds',r'\brebounds?\b|\brebs\b|\bto grab\b'),('assists',r'\bassists?\b|\basts\b|\bto record\b')] if re.search(p,context,re.I)]
                        if len(kinds)!=1: continue
                        for o in objs(m.get('outcomes')):
                            if o.get('isHidden') is True or o.get('isOpenForBetting') is False: continue
                            sn=val(o.get('name'),'')
                            player=val(o.get('playerName') or m.get('playerName'))
                            if not player:
                                prefix=re.match(r'^(.+?)\s+-\s+',mn)
                                if prefix: player=prefix.group(1)
                            if not player:
                                guess=re.sub(r'\s+(?:Over|Under)\s*\d*(?:\.\d+)?\s*$|\s+\d+(?:\.\d+)?\s*\+.*$','',sn,flags=re.I)
                                if guess!=sn: player=guess
                            if not player or re.match(r'^(?:Player|Total|Team|NBA|Over|Under|Yes|No)$',player,re.I): continue
                            plus=re.search(r'(\d+(?:\.\d+)?)\s*\+\s*(?:Points?|Pts|Rebounds?|Rebs|Assists?|Asts)?\s*$',sn,re.I) or re.search(r'(\d+(?:\.\d+)?)\s*\+\s*(?:Points?|Pts|Rebounds?|Rebs|Assists?|Asts)?\b',mn,re.I)
                            side_m=re.search(r'\b(Over|Under|Higher|Lower)\b',sn,re.I)
                            side=({'higher':'over','lower':'under'}.get(side_m.group(1).lower(),side_m.group(1).lower()) if side_m else val(o.get('resultType'),'').lower())
                            if plus: category='alternative'; line=plus.group(1); side=side if side in ('over','under') else 'over'
                            else:
                                category='standard'; line=o.get('points')
                                if line is None:
                                    lm=re.search(r'\b(?:Over|Under)\s+(\d+(?:\.\d+)?)',sn,re.I)
                                    line=lm.group(1) if lm else None
                            price=o.get('price'); odds=price.get('decimal') or price.get('winPrice') if isinstance(price,dict) else price
                            if odds is None: odds=o.get('decimalOdds')
                            r=row('PointsBet',eid,data.get('name') or data.get('eventName'),start,player,kinds[0],category,side,line,odds,m.get('key'),mn,o.get('key'),sn)
                            if r: out.append(r)
            except (requests.RequestException,ValueError,TypeError) as ex: LOG.warning('PointsBet event %s: %s',eid,ex)
            c.wait()
    return out

SCRAPERS={'sportsbet':sportsbet,'tabtouch':tabtouch,'tab':tab,'betr':betr,'ladbrokes':ladbrokes,'pointsbet':pointsbet}

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--books',nargs='+',choices=['all',*BOOKS],default=['all'])
    ap.add_argument('--days-ahead',type=int,default=14)
    ap.add_argument('--delay',type=float,default=.5)
    ap.add_argument('--ladbrokes-delay',type=float,default=3.0,help='Seconds between Ladbrokes EventCard requests (default: 3)')
    ap.add_argument('--timeout',type=int,default=30)
    ap.add_argument('--output-dir',type=Path,default=Path('sportsbook_odds'))
    ap.add_argument('--debug',action='store_true')
    args=ap.parse_args()
    if args.delay < 0 or args.ladbrokes_delay < 0: ap.error('Delays must be nonnegative')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,format='%(levelname)s %(message)s')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selected=BOOKS if 'all' in args.books else tuple(dict.fromkeys(args.books))
    combined=[]
    for book in selected:
        LOG.info('Scraping %s',book)
        try: records=SCRAPERS[book](Client(args.ladbrokes_delay if book=='ladbrokes' else args.delay,args.timeout),args.days_ahead)
        except Exception as exc:
            LOG.error('%s discovery failed: %s',book,exc)
            records=[]
        df=pd.DataFrame(records)
        if df.empty: df=pd.DataFrame(columns=COLS)
        else:
            df=df.drop_duplicates(subset=['bookmaker','event_id','market_id','selection_id','player_name','stat','line','bet_type'])
            df=df.reindex(columns=COLS+[x for x in df.columns if x not in COLS])
        df.to_csv(args.output_dir/f'{book}_nba_props.csv',index=False)
        LOG.info('%s: %s selections',book,len(df))
        combined.extend(df.to_dict('records'))
    all_df=pd.DataFrame(combined)
    if all_df.empty: all_df=pd.DataFrame(columns=COLS)
    all_df.to_csv(args.output_dir/'nba_all_sportsbook_odds.csv',index=False)
    LOG.info('Total: %s selections -> %s',len(all_df),args.output_dir.resolve())

if __name__=='__main__': main()
