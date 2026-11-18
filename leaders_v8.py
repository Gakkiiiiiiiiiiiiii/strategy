"""Frozen retrospective test of countertrend leaders before a market rebound.

No search or parameter selection. Industry membership is a dated, prior-year
snapshot; unknown membership/status is never silently treated as a pass.
"""
import argparse
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dryup_v5 as d
import confirm_v6 as c6
from review_v2 import stats

OUT=d.bt.ROOT/'results/research_v8'
HORIZONS=d.HORIZONS
SPEC=dict(asof=20261116,search=False,market='frozen surviving-universe equal-weight daily returns',
    confirmation='prior-day trailing20 market return <0; today market >=1%; stock >6% and >market+2pp; positive body, wick<=body/3, volume>prior5 and prior10; price>=3, amount20>=30m',
    impulse='same stock candle/liquidity rules; market daily<=0 and trailing20 including impulse<0; most recent such day 5-30 market sessions before confirmation',
    intervening='market index at confirmation-1 below impulse close; no selection using post-confirmation prices',
    pullback='prior close 5-20% below highest close from impulse through confirmation-1; every intervening close >=pre-impulse close; last3 volume <=70% mean from impulse through known peak',
    resistant='entire waiting-phase maximum close-to-close drawdown <5%; every intervening close >=pre-impulse close; recovered-after-deep-dip kept separate',
    entry='confirmation+1 market session open; upper-limit open skipped; no 9.5% cap; historical ST excluded',
    exit='5/10/20/30/60/120 sessions; entry is day1; hard15% stop default; T+1 and locked lower-limit delayed; same-entry no-stop control',
    repeated='first qualifying confirmation per stock/impulse for each setup group; same-stock nonoverlap separately by horizon, using common stop/no-stop exit',
    industry='prior-calendar-year-end dated CSRC industry snapshot; blank/future/unknown excluded; focal stock removed from peer daily returns and breadth',
    industry_confirmation='at least20 other members; sector prior5 return beats market prior5, sector today beats market, >=60% other members up',
    controls='same signal date, board and signal-time amount20 quartile; no eligible prior countertrend impulse; compare mean fixed-horizon event returns',
    split='entry<=2024 and entire stop/no-stop exit<=2024 training-description; signals>=2025 retrospective validation; cross-boundary excluded',
    cost=.0035,limitations=['Previously inspected history, no pristine holdout or proven absorption/causality.','Current TDX price universe omits some delisted stocks.','Annual lagged classifications can be stale; CSRC industry is not a theme/concept board.','Event statistics do not model portfolio capital or imply CAGR.'])


def dump(name,value):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def build():
    dump('spec.json',SPEC)
    e=pd.read_pickle(d.r.OUT/'events.pkl')
    panel=pd.read_pickle(d.r.OUT/'panel.pkl');vols=d.volume_data(panel)
    market=pd.read_csv(d.r.OUT/'market.csv');level=(1+market.daily).cumprod().to_numpy()
    previous=market.set_index('date').market20.shift()
    liquid=(e.price>=3)&(e.amount20>=3e7)
    anchor=liquid&(e.market20<0)&(e.daily<=0)
    candidates=liquid&(e.signal_date.map(previous)<0)&(e.daily>=.01)&(e.signal_return>e.daily+.02)
    rows=[]
    for code,g in e[candidates].groupby('code'):
        prior=e[anchor&e.code.eq(code)].sort_values('signal_index')
        dates,raw,close,factor,pre,mi,*_=panel[code];volume=vols[code][:,0]
        for t in g.itertuples():
            rec=t._asdict();rec.pop('Index');rec.update(event_id=int(t.Index),impulse_date=0,impulse_index=-1,
                leader=False,pullback=False,resistant=False,recovered=False,deep_impulse=False,peak_index=-1,peak_date=0,
                phase_market=np.nan,phase_stock=np.nan,phase_excess=np.nan,draw_from_peak=np.nan,dry_ratio=np.nan,
                impulse_return=np.nan,impulse_market=np.nan,impulse_volume5=np.nan,impulse_volume10=np.nan,phase_max_drawdown=np.nan)
            choices=prior[(prior.signal_index<=t.signal_index-5)&(prior.signal_index>=t.signal_index-30)]
            if len(choices):
                a=choices.iloc[-1];i=int(t.stock_index);s=int(a.stock_index);sm=int(a.signal_index)
                # The anchor choice is independent of subsequent price/volume shape.
                if level[t.signal_index-1]<level[sm]:
                    peak=s+int(np.argmax(close[s:i]));drop=1-close[i-1]/close[peak]
                    defend=np.min(close[s:i])>=close[s-1]
                    dry=float(volume[i-3:i].mean()/volume[s:peak+1].mean())
                    maxdd=float(np.max(1-close[s:i]/np.maximum.accumulate(close[s:i])))
                    pr=float(close[i-1]/close[s]-1);mr=float(level[t.signal_index-1]/level[sm]-1)
                    rec.update(leader=True,impulse_date=int(a.signal_date),impulse_index=s,peak_index=peak,peak_date=int(dates[peak]),
                        phase_market=mr,phase_stock=pr,phase_excess=pr-mr,draw_from_peak=float(drop),dry_ratio=dry,
                        pullback=bool(.05<=drop<=.20 and defend and dry<=.70),resistant=bool(maxdd<.05 and defend),
                        recovered=bool(maxdd>=.05 and drop<.05 and defend),phase_max_drawdown=maxdd,
                        deep_impulse=bool(a.drawdown>=.30 and a.floor_days<=8 and a.prior_from_floor<=.08),
                        impulse_return=float(a.signal_return),impulse_market=float(a.daily),
                        impulse_volume5=float(a.volume5),impulse_volume10=float(a.volume10))
            rows.append(rec)
    x=pd.DataFrame(rows).sort_values(['signal_date','code']).reset_index(drop=True)
    x['amount_bucket']=x.groupby(['signal_date','board']).amount20.transform(lambda z:np.minimum(3,np.floor(z.rank(method='first',pct=True)*3.999))).astype(int)
    x.to_pickle(OUT/'events.pkl')
    print('BUILT',len(x),'leaders',int(x.leader.sum()),'pullback',int(x.pullback.sum()),'resistant',int(x.resistant.sum()),flush=True)


def industries():
    import update_market as u
    socket.setdefaulttimeout(25)
    directory=OUT/'industry_cache';directory.mkdir(exist_ok=True)
    assert u.bs.login().error_code=='0'
    receipt=[]
    try:
        for year in range(2022,2027):
            date=f'{year-1}-12-31';path=directory/f'{year}.csv'
            if not path.exists():
                frame=u.frame(u.bs.query_stock_industry(date=date))
                assert len(frame)>1000
                assert pd.to_datetime(frame.updateDate).max()<=pd.Timestamp(date)
                u.save(frame,path)
            frame=pd.read_csv(path)
            receipt.append(dict(signal_year=year,requested_asof=date,max_update=str(frame.updateDate.max()),rows=len(frame),classified=int(frame.industry.notna().sum())))
            print('INDUSTRY',receipt[-1],flush=True)
    finally:u.bs.logout()
    dump('industry_sources.json',receipt)


def sector_features():
    e=pd.read_pickle(OUT/'events.pkl');panel=pd.read_pickle(d.r.OUT/'panel.pkl')
    market=pd.read_csv(d.r.OUT/'market.csv');cal=market.date.to_numpy();market_level=(1+market.daily).cumprod().to_numpy()
    codes=list(panel);mapping={code:i for i,code in enumerate(codes)}
    ret=np.full((len(codes),len(cal)),np.nan)
    for code,p in panel.items():
        dates,raw,close,factor,pre,mi,*_=p
        valid=np.arange(len(dates))>=60
        ret[mapping[code],mi[valid]]=raw[valid,3]/pre[valid]-1
    output=[]
    for year,g in e.groupby(e.signal_date//10000):
        path=OUT/'industry_cache'/f'{year}.csv'
        if not path.exists():continue
        classification=pd.read_csv(path).dropna(subset=['industry']).drop_duplicates('code',keep='last').set_index('code')
        aggregates={}
        for sector,members in classification.groupby('industry'):
            ids=[mapping[k] for k in members.index if k in mapping]
            if not ids:continue
            a=ret[ids];aggregates[sector]=(np.nansum(a,axis=0),np.isfinite(a).sum(0),(a>0).sum(0))
        for x in g.itertuples():
            rec=dict(event_id=x.event_id,industry='',industry_date='',sector_confirm=False,sector_prior_lead=False,
                     sector_today=np.nan,sector_prior5=np.nan,sector_excess5=np.nan,sector_breadth=np.nan,sector_members=0,
                     sector_impulse_excess=np.nan,sector_market_contribution=np.nan)
            if x.code in classification.index:
                row=classification.loc[x.code];sector=row.industry;sums,counts,up=aggregates[sector];own=ret[mapping[x.code]]
                n=counts-np.isfinite(own);s=sums-np.nan_to_num(own);peers=np.divide(s,n,out=np.full(len(cal),np.nan),where=n>0)
                breadth=np.divide(up-(own>0),n,out=np.full(len(cal),np.nan),where=n>0)
                t=int(x.signal_index);five=float(np.prod(1+peers[t-5:t])-1);mfive=float(market_level[t-1]/market_level[t-6]-1)
                if n[t]>=20 and np.min(n[t-5:t])>=20:
                    rec.update(industry=sector,industry_date=str(row.updateDate),sector_today=float(peers[t]),sector_prior5=five,
                        sector_excess5=five-mfive,sector_breadth=float(breadth[t]),sector_members=int(n[t]),
                        sector_confirm=bool(five>mfive and peers[t]>x.daily and breadth[t]>=.60),
                        sector_market_contribution=float(sums[t]/np.isfinite(ret[:,t]).sum()))
                    if x.leader:
                        a=int(np.searchsorted(cal,x.impulse_date));excess=float(peers[a]-market.daily.iloc[a])
                        rec.update(sector_impulse_excess=excess,sector_prior_lead=bool(n[a]>=20 and excess>0 and breadth[a]>=.50))
            output.append(rec)
    features=pd.DataFrame(output).set_index('event_id')
    e=e.drop(columns=e.columns.intersection(features.columns)).join(features,on='event_id');e.to_pickle(OUT/'events.pkl')
    print('SECTOR',e.industry.fillna('').ne('').sum(),'confirmed',int(e.sector_confirm.fillna(False).sum()),flush=True)


def status_maps():
    flags=pd.read_pickle(c6.OUT/'status_maps.pkl')
    for p in (OUT/'status_cache').glob('*.csv'):
        x=pd.read_csv(p);x['date']=x.date.str.replace('-','').astype(int)
        flags.setdefault(p.stem,{}).update({int(a.date):int(a.isST) for a in x.dropna(subset=['isST']).itertuples()})
    return flags


def sector_outcomes():
    """Ex-post reference returns of a static equal-weight industry basket.

    Known prior-year membership, focal stock excluded, peers must have an
    actual entry-day open. Suspended peers carry their last observed close.
    No upper-limit/stop execution simulation: this is contextual attribution.
    """
    e=pd.read_pickle(OUT/'events.pkl');panel=pd.read_pickle(d.r.OUT/'panel.pkl')
    cal=pd.read_csv(d.r.OUT/'market.csv').date.to_numpy();codes=list(panel);lookup={c:i for i,c in enumerate(codes)}
    opens=np.full((len(codes),len(cal)),np.nan);closes=np.full_like(opens,np.nan)
    for code,p in panel.items():
        dates,raw,close,factor,pre,mi,*_=p;k=lookup[code]
        opens[k,mi]=raw[:,0]*factor
        at=np.searchsorted(mi,np.arange(len(cal)),side='right')-1;valid=at>=0
        closes[k,valid]=close[at[valid]]
    maps={}
    for year in range(2022,2027):
        mapping=pd.read_csv(OUT/'industry_cache'/f'{year}.csv').dropna(subset=['industry'])
        for industry,g in mapping.groupby('industry'):
            maps[year,industry]=np.array([lookup[c] for c in g.code if c in lookup],dtype=int)
    output=[]
    for x in e.itertuples():
        rec={'event_id':x.event_id};ids=maps.get((x.signal_date//10000,x.industry),np.array([],dtype=int))
        ids=ids[ids!=lookup[x.code]];j=x.signal_index+1
        for h in HORIZONS:
            rec[f'sector_return_{h}']=np.nan;rec[f'sector_n_{h}']=0
            end=j+h-1
            if end>=len(cal):continue
            members=ids[np.isfinite(opens[ids,j])&np.isfinite(closes[ids,end])]
            if len(members)<20:continue
            gross=np.mean(closes[members,end]/opens[members,j])-1
            rec[f'sector_return_{h}']=float((1+gross)*(1-d.r.COST/2)/(1+d.r.COST/2)-1)
            rec[f'sector_n_{h}']=len(members)
        output.append(rec)
    pd.DataFrame(output).to_pickle(OUT/'sector_outcomes.pkl')
    print('INDUSTRY_REFERENCE',len(output),flush=True)


def statuses(limit=0,shard=0,shards=1):
    import update_market as u
    socket.setdefaulttimeout(20)
    directory=OUT/'status_cache';directory.mkdir(exist_ok=True)
    e=pd.read_pickle(OUT/'events.pkl');flags=status_maps();requests=[]
    # Prioritize leader stocks; controls with unknown dated flags remain excluded.
    for code,g in e.groupby('code'):
        if int(code[3:])%shards!=shard:continue
        required=set(g.signal_date)|set(g.entry_date)|set(g.loc[g.leader,'impulse_date'])
        missing=required-set(flags.get(code,{}))
        missing.discard(0)
        if missing:requests.append((not g.leader.any(),code,min(missing)))
    requests.sort();print('STATUS_NEEDS',len(requests),flush=True)
    if limit:requests=requests[:limit]
    assert u.bs.login().error_code=='0'
    try:
        for n,(_,code,start) in enumerate(requests):
            path=directory/f'{code}.csv';date=str(pd.Timestamp(str(start)).date())
            for attempt in range(3):
                try:
                    x=u.frame(u.bs.query_history_k_data_plus(code,'date,isST,tradestatus',date,'2026-11-16',frequency='d',adjustflag='3'))
                    assert len(x)>0
                    if path.exists():x=pd.concat([pd.read_csv(path),x]).drop_duplicates('date',keep='last').sort_values('date')
                    u.save(x,path);break
                except Exception:
                    if attempt==2:raise
                    u.session.default_socket.close();assert u.bs.login().error_code=='0'
            if n%100==0:print('STATUS',n,code,flush=True)
    finally:u.bs.logout()


def outcomes():
    e=pd.read_pickle(OUT/'events.pkl');panel=pd.read_pickle(d.r.OUT/'panel.pkl');flags=status_maps()
    market=pd.read_csv(d.r.OUT/'market.csv');cal=market.date.to_numpy()
    bm=pd.read_csv(d.OUT/'sh.000300.csv');bm['date']=bm.date.str.replace('-','').astype(int);bm=bm.set_index('date')
    rows=[];start=time.time()
    for n,x in enumerate(e.itertuples()):
        dates,raw,close,factor,pre,mi,*_=panel[x.code];i=x.stock_index;j=i+1;st=flags.get(x.code,{})
        rec=x._asdict();rec.pop('Index');rec['entry_status']='no_next_bar'
        if j>=len(raw) or mi[j]!=mi[i]+1:rows.append(rec);continue
        checks=[int(dates[i]),int(dates[j])]+([int(x.impulse_date)] if x.leader else [])
        rec.update(entry_date=int(dates[j]),buy_price=float(raw[j,0]/100))
        if any(k not in st for k in checks):rec['entry_status']='unknown_st';rows.append(rec);continue
        if any(st[k] for k in checks):rec['entry_status']='st_excluded';rows.append(rec);continue
        rate=20 if x.code[3:6] in {'300','301','688'} else 10
        upper=max(math.floor((pre[j]*(100+rate)+50)/100),pre[j]+1)
        if raw[j,0]>=upper:rec['entry_status']='upper_open';rows.append(rec);continue
        rec['entry_status']='filled'
        for h in HORIZONS:
            target=int(mi[j]+h-1)
            if target>=len(cal):continue
            a,sa=d.executable_exit(raw,factor,mi,pre,st,x.code,j,target,cal)
            b,sb=c6.hard_exit(raw,factor,mi,pre,dates,st,x.code,j,target,cal)
            if a is None or b is None:continue
            za,adj,px,delay=a
            net=lambda p:float(p/(raw[j,0]*factor[j])*(1-d.r.COST/2)/(1+d.r.COST/2)-1)
            benchmark=float(bm.loc[int(cal[target]),'close']/bm.loc[int(dates[j]),'open']*(1-d.r.COST/2)/(1+d.r.COST/2)-1)
            rec.update({f'net_{h}':net(b['adjusted']),f'nostop_{h}':net(adj),f'end_{h}':int(max(dates[za],dates[b['z']])),
                f'stop_exit_{h}':int(dates[b['z']]),f'nostop_exit_{h}':int(dates[za]),f'stop_price_{h}':b['price'],
                f'reason_{h}':b['reason'],f'benchmark_{h}':benchmark,f'excess_{h}':net(b['adjusted'])-benchmark})
        rows.append(rec)
        if n%5000==0:print('OUTCOME',n,round(time.time()-start,1),flush=True)
    z=pd.DataFrame(rows);z.to_pickle(OUT/'outcomes.pkl');print('FILLS',z.entry_status.value_counts().to_dict(),flush=True)


def nonoverlap(frame,h):
    selected=[];last={}
    for x in frame.sort_values(['signal_date','code']).itertuples():
        if x.entry_date<=last.get(x.code,0):continue
        selected.append(x.Index);last[x.code]=getattr(x,f'end_{h}')
    return frame.loc[selected].copy()


def interval(frame,column,iterations=5000):
    block=(frame.signal_date//10000)*2+((frame.signal_date//100)%100>=7).astype(int)
    g=frame.assign(block=block).groupby('block')[column].agg(['sum','count'])
    if len(g)<2:return None
    rng=np.random.default_rng(8826);a=rng.integers(0,len(g),size=(iterations,len(g)))
    values=g['sum'].to_numpy()[a].sum(1)/g['count'].to_numpy()[a].sum(1)
    return dict(half_year_blocks=len(g),ci95=np.quantile(values,[.025,.975]).tolist(),note='Entry half-year blocks; few blocks, retrospective and not adjusted for prior model discovery.')


def analyze():
    e=pd.read_pickle(OUT/'outcomes.pkl').join(pd.read_pickle(OUT/'sector_outcomes.pkl').set_index('event_id'),on='event_id')
    panel=pd.read_pickle(d.r.OUT/'panel.pkl')
    e['impulse_gain_to_confirm']=np.nan;e['waiting_days']=np.nan
    for x in e[e.leader].itertuples():
        dates,raw,close,factor,pre,mi,*_=panel[x.code]
        e.loc[x.Index,'impulse_gain_to_confirm']=close[x.stock_index]/close[x.impulse_index-1]-1
        e.loc[x.Index,'waiting_days']=mi[x.stock_index]-mi[x.impulse_index]
    summary=[];matched=[];annual=[];sector_conditioned=[];review=[]
    groups={'today_only':np.ones(len(e),bool),'nonleader':~e.leader,'prior_leader':e.leader,
        'pullback':e.pullback,'resistant':e.resistant,'recovered':e.recovered,'both':e.pullback|e.resistant,
        'both_sector':(e.pullback|e.resistant)&e.sector_confirm.fillna(False),
        'both_sector_early':(e.pullback|e.resistant)&e.sector_confirm.fillna(False)&e.sector_prior_lead.fillna(False),
        'both_deep':(e.pullback|e.resistant)&e.deep_impulse}
    for h in HORIZONS:
        valid=e.entry_status.eq('filled')&e[f'net_{h}'].notna()
        allcontrols=e[valid&~e.leader]
        # Means conditional on the same signal day, board and known liquidity.
        key=['signal_date','board','amount_bucket']
        controls=allcontrols.groupby(key)[f'net_{h}'].agg(['mean','count'])
        for name,mask in groups.items():
            g=e[mask].copy()
            if name not in ('today_only','nonleader'):
                g=g.sort_values('signal_date').drop_duplicates(['code','impulse_date'])
            g=g[valid.loc[g.index]]
            g=nonoverlap(g,h)
            if name in ('pullback','resistant'):
                for confirmed,q in g[g.industry.fillna('').ne('')].groupby('sector_confirm'):
                    p=q.join(controls,on=key);p=p[p['count']>=3]
                    sector_conditioned.append(dict(group=name,horizon=h,sector_confirm=bool(confirmed),**stats(q[f'net_{h}']),
                        same_day_control_delta=float((p[f'net_{h}']-p['mean']).mean()) if len(p) else None,
                        entry_dates=int(q.signal_date.nunique()),mean_industry_excess=float((q[f'net_{h}']-q[f'sector_return_{h}']).mean()),
                        interval=interval(q,f'net_{h}'),without_top2_mean=float(q[f'net_{h}'].sort_values().iloc[:-2].mean()) if len(q)>2 else None,
                        note='Same original first-signal cohort; no later re-entry chosen to satisfy sector confirmation.'))
            for split,q in [('all',g),('through2024',g[g[f'end_{h}']<=20241231]),('from2025',g[g.signal_date>=20250101])]:
                if len(q)==0:continue
                summary.append(dict(group=name,horizon=h,split=split,**stats(q[f'net_{h}']),nostop_mean=float(q[f'nostop_{h}'].mean()),
                    mean_excess=float(q[f'excess_{h}'].mean()),entry_dates=int(q.signal_date.nunique()),
                    median_gain_before_entry=float(q.impulse_gain_to_confirm.median()) if q.impulse_gain_to_confirm.notna().any() else None,
                    median_waiting_days=float(q.waiting_days.median()) if q.waiting_days.notna().any() else None,
                    industry_comparable=int(q[f'sector_return_{h}'].notna().sum()),
                    industry_mean=float(q[f'sector_return_{h}'].mean()) if q[f'sector_return_{h}'].notna().any() else None,
                    mean_industry_excess=float((q[f'net_{h}']-q[f'sector_return_{h}']).mean()) if q[f'sector_return_{h}'].notna().any() else None,
                    stop_fraction=float(q[f'reason_{h}'].str.startswith('stop').mean()),interval=interval(q,f'net_{h}')))
                if name not in ('today_only','nonleader'):
                    p=q.join(controls,on=key);p=p[p['count']>=3].copy();p['delta']=p[f'net_{h}']-p['mean']
                    matched.append(dict(group=name,horizon=h,split=split,n=len(p),dates=int(p.signal_date.nunique()),
                        mean_delta=float(p.delta.mean()) if len(p) else None,interval=interval(p,'delta')))
            for year,q in g.groupby(g.entry_date//10000):
                annual.append(dict(group=name,horizon=h,year=int(year),**stats(q[f'net_{h}']),dates=int(q.signal_date.nunique())))
            if name in ('pullback','resistant','both_sector','both_sector_early'):g.to_csv(OUT/f'{name}_{h}_trades.csv',index=False,encoding='utf-8-sig')
            if name in ('pullback','resistant') and h in (20,30):
                selected=['event_id','code','name','signal_date','impulse_date','entry_date','buy_price','industry','industry_date','sector_confirm',
                    'phase_market','phase_stock','phase_excess','draw_from_peak','phase_max_drawdown','dry_ratio','impulse_gain_to_confirm','waiting_days',
                    f'net_{h}',f'nostop_{h}',f'stop_exit_{h}',f'stop_price_{h}',f'reason_{h}',f'benchmark_{h}',f'sector_return_{h}']
                q=g[selected].copy();q['group']=name;q['horizon']=h
                q=q.rename(columns={f'{k}_{h}':k for k in ('net','nostop','stop_exit','stop_price','reason','benchmark','sector_return')});review.append(q)
    report=dict(spec=SPEC,coverage=dict(candidates=len(e),entry_status=e.entry_status.value_counts().to_dict(),
        industry_known=int(e.industry.fillna('').ne('').sum()),signal_dates=int(e.signal_date.nunique())),summary=summary,matched=matched,annual=annual,sector_conditioned=sector_conditioned)
    dump('report.json',report)
    pd.DataFrame(summary).drop(columns='interval').to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(matched).drop(columns='interval').to_csv(OUT/'matched_summary.csv',index=False,encoding='utf-8-sig')
    pd.concat(review,ignore_index=True).to_csv(OUT/'signals_review.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(summary).query("horizon in [20,30] and split=='all'")[['group','horizon','n','mean','win','payoff','nostop_mean','mean_excess']].to_string(index=False),flush=True)
    print('MATCHED',pd.DataFrame(matched).query("horizon in [20,30] and split=='all'")[['group','horizon','n','mean_delta']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['build','industries','sectors','sector_outcomes','statuses','outcomes','analyze']);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.task=='statuses':statuses(a.limit,a.shard,a.shards)
    else:globals()[{'sectors':'sector_features'}.get(a.task,a.task)]()
