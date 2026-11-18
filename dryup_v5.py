"""Low-base volume rally and causal volume-dry pullback event study.

Frozen descriptive thresholds, not a return-maximizing parameter search.
The 2026-11-16 data snapshot has already been inspected in prior work.
"""
import argparse
import json
import math
import socket
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import backtest as bt
import research_v2 as r
from review_v2 import stats

OUT=bt.ROOT/'results/research_v5'
HORIZONS=(5,10,20,30,60,120)
PROFILES={
    'ordinary':dict(label='普通缩量',dry=.70,relative=.80,minimum=5),
    'extreme':dict(label='极致缩量',dry=.40,relative=.50,minimum=10),
    'ultra':dict(label='更严缩量',dry=.25,relative=.35,minimum=20),
}
SPEC=dict(asof=20261116,data_start=20210802,holding_days=list(HORIZONS),cost=r.COST,
    low='floor close <= 1.15 * prior 500-session minimum close; at least 250 prior bars',
    decline='prior 250-session peak to floor >=30%',
    rally='last 60-session peak after a 5-60 bar rally of >=15% from prior 90-bar low',
    volume='rally average > both pre-floor 5 and 10-bar means',
    pullback='5-30 bars, price decline 5%-30%, retrace 20%-85%, no close below floor',
    wash_volume='last 5 pullback bars mean <=70% of rally mean; whole-pullback mean retained for diagnosis',
    extreme='3-bar mean <=40% of rally mean AND <=50% of preceding 20-bar mean; current volume <= previous 10-bar minimum',
    entry='signal confirmed after close; next market session open; upper-limit opens skipped',
    exit='entry is day1; scheduled close; locked lower-limit/suspension delays to first sellable open',
    dedup='first qualifying signal per stock/floor/peak; then exclude overlapping same-stock trades per horizon',
    stop='none; fixed horizon isolates the new entry pattern',
    warning='retrospective, surviving TDX universe; daily execution estimate; total-return adjustment approximation',
    profiles=PROFILES,parameter_optimization=False)


@njit(cache=True)
def scan(close,volume,amount,raw_close):
    """Every feature at i reads index <= i only; no centered minima."""
    result=np.empty((len(close),19));n=0
    for i in range(280,len(close)):
        if raw_close[i]<300 or np.mean(amount[i-19:i+1])<3e7:continue
        p=i-60+np.argmax(close[i-60:i]);wash=i-p
        if wash<5 or wash>30:continue
        f=max(0,p-90)+np.argmin(close[max(0,p-90):p-2])
        if f<250 or p-f<5 or p-f>60:continue
        low=np.min(close[max(0,f-499):f+1]);near=close[f]/low
        if near>1.15:continue
        high=np.max(close[f-250:f]);decline=1-close[f]/high
        rally=close[p]/close[f]-1;drop=1-close[i]/close[p]
        retrace=(close[p]-close[i])/(close[p]-close[f])
        if decline<.30 or rally<.15 or drop<.05 or drop>.30 or retrace<.20 or retrace>.85:continue
        if np.min(close[p+1:i+1])<close[f]:continue
        rv=np.mean(volume[f+1:p+1]);base=max(np.mean(volume[f-4:f+1]),np.mean(volume[f-9:f+1]))
        if rv<=base:continue
        wash_ratio=np.mean(volume[p+1:i+1])/rv
        tail_ratio=np.mean(volume[i-4:i+1])/rv
        if tail_ratio>.70:continue
        v3=np.mean(volume[i-2:i+1]);v20=np.mean(volume[i-20:i])
        result[n]=np.array([i,f,p,decline,rally,drop,retrace,near,rv/base,wash_ratio,v3/rv,v3/v20,
            1.0*(volume[i]<=np.min(volume[i-5:i])),1.0*(volume[i]<=np.min(volume[i-10:i])),
            1.0*(volume[i]<=np.min(volume[i-20:i])),rv,v3,base,tail_ratio])
        n+=1
    return result[:n]


def volume_data(panel):
    cache=OUT/'volume_cache.pkl'
    if cache.exists():return pd.read_pickle(cache)
    _,updates=bt.updated_bars(20261104,20261116,{})
    result={}
    with ExitStack() as stack:
        _,read=bt.market_files(Path('D:/tdx/vipdoc'),stack)
        for code,p in panel.items():
            b=np.frombuffer(read(code.replace('.','')),bt.BAR_DTYPE)
            m={int(x['date']):(float(x['volume']),float(x['amount'])) for x in b}
            if code in updates:
                for x in updates[code][0]:m[int(x['date'])]=(float(x['volume']),float(x['amount']))
            result[code]=np.array([m[int(d)] for d in p[0]])
    pd.to_pickle(result,cache)
    return result


def build():
    OUT.mkdir(exist_ok=True)
    (OUT/'spec.json').write_text(json.dumps(SPEC,ensure_ascii=False,indent=2),encoding='utf-8')
    p=pd.read_pickle(r.OUT/'panel.pkl');vols=volume_data(p);names=bt.current_names(Path('D:/tdx'))
    rows=[];started=time.time()
    for n,(code,pack) in enumerate(p.items()):
        dates,raw,c,factor,pre,mi,*_=pack;vol,amt=vols[code].T
        found=scan(c,vol,amt,raw[:,3])
        for x in found:
            i,f,pk=map(int,x[:3])
            if dates[i]<20220101:continue
            rec=dict(code=code,name=names.get(code,''),signal_date=int(dates[i]),stock_index=i,
                     signal_index=int(mi[i]),floor_date=int(dates[f]),peak_date=int(dates[pk]),
                     floor_index=f,peak_index=pk,price=float(raw[i,3]/100),
                     signal_return=float(raw[i,3]/pre[i]-1),amount20=float(amt[i-19:i+1].mean()),
                     floor_price=float(c[f]/factor[i]/100),peak_price=float(c[pk]/factor[i]/100),
                     wash_days=i-pk,rally_days=pk-f)
            for key,val in zip(('decline','rally','pullback','retrace','near_low','rally_volume_ratio','wash_volume_ratio',
                'dry_ratio','dry_vs20','minimum5','minimum10','minimum20','rally_volume','dry_volume','base_volume','tail_volume_ratio'),x[3:]):rec[key]=float(val)
            rows.append(rec)
        if n%1000==0:print(n,'stocks',len(rows),'structural days',round(time.time()-started,1),'sec',flush=True)
    e=pd.DataFrame(rows).sort_values(['code','signal_date']).reset_index(drop=True)
    for key,q in PROFILES.items():e[key]=(e.dry_ratio<=q['dry'])&(e.dry_vs20<=q['relative'])&(e['minimum'+str(q['minimum'])]>0)
    e['event_id']=e.index
    e.to_pickle(OUT/'events.pkl')
    e.loc[e.code.eq('sz.300369')].to_csv(OUT/'nsfocus_candidates.csv',index=False,encoding='utf-8-sig')
    print('built',len(e),'events;', {k:int(e[k].sum()) for k in PROFILES},flush=True)


def inputs():
    import update_market as u
    socket.setdefaulttimeout(30);OUT.mkdir(exist_ok=True)
    assert u.bs.login().error_code=='0'
    try:
        for code,start in [('sh.000300','2021-08-01'),('sz.300369','2014-01-01')]:
            q=u.frame(u.bs.query_history_k_data_plus(code,'date,open,high,low,close,volume,preclose',start,
                '2026-11-16',frequency='d',adjustflag='3'))
            assert len(q)>0
            u.save(q,OUT/(code+'.csv'));print(code,len(q),flush=True)
    finally:u.bs.logout()


def statuses(shard=0,shards=1):
    """Fetch dated ST flags for all profile candidates, never use today's name."""
    import update_market as u
    socket.setdefaulttimeout(30)
    directory=OUT/'status_cache';directory.mkdir(exist_ok=True)
    e=pd.read_pickle(OUT/'events.pkl');e=e[e[list(PROFILES)].any(axis=1)]
    print('codes requiring historical status',e.code.nunique(),flush=True)
    assert u.bs.login().error_code=='0'
    try:
        for n,(code,g) in enumerate(e.groupby('code')):
            if n%shards!=shard:continue
            path=directory/(code+'.csv')
            start=str(pd.Timestamp(str(g.signal_date.min())).date())
            if path.exists() and pd.read_csv(path,usecols=['date']).date.min()<=start:continue
            for attempt in range(3):
                try:
                    frame=u.frame(u.bs.query_history_k_data_plus(code,'date,isST,tradestatus',start,'2026-11-16',frequency='d',adjustflag='3'))
                    if frame.empty:raise ValueError('Empty historical flags: '+code)
                    u.save(frame,path);break
                except Exception:
                    if attempt==2:raise
                    u.session.default_socket.close();assert u.bs.login().error_code=='0'
            if n%100==0:print('status',n,code,flush=True)
    finally:u.bs.logout()


def status_maps():
    result={}
    for file in (OUT/'status_cache').glob('*.csv'):
        d=pd.read_csv(file);d['date']=d.date.str.replace('-','').astype(int)
        result[file.stem]={int(x.date):int(x.isST) for x in d.dropna(subset=['isST']).itertuples()}
    return result


def executable_exit(raw,factor,mi,pre,st,code,j,target,cal):
    if target>=len(cal):return None,'immature'
    z=int(np.searchsorted(mi,target));delayed=0
    while z<len(raw):
        growth=code[3:6] in {'300','301','688'}
        rate=20 if growth else 5 if st.get(int(cal[mi[z]]))==1 and cal[mi[z]]<bt.ST_LIMIT_CHANGE else 10
        limit=min(math.floor((pre[z]*(100-rate)+50)/100),pre[z]-1)
        locked=raw[z,1]==raw[z,2] and raw[z,2]<=limit
        if not locked:
            delayed=int(mi[z]-target);px=raw[z,3] if mi[z]==target else raw[z,0]
            return (z,float(px*factor[z]),float(px/100),delayed),'complete'
        z+=1
    return None,'unresolved'


def outcomes():
    e=pd.read_pickle(OUT/'events.pkl');p=pd.read_pickle(r.OUT/'panel.pkl');flags=status_maps()
    cal=pd.read_csv(r.OUT/'market.csv').date.to_numpy();index=pd.read_csv(OUT/'sh.000300.csv')
    index['date']=index.date.str.replace('-','').astype(int);benchmark=index.set_index('date')
    rows=[]
    for code,g in e[e[list(PROFILES)].any(axis=1)].groupby('code'):
        dates,raw,c,factor,pre,mi,*_=p[code];st=flags.get(code,{})
        for x in g.itertuples():
            i=x.stock_index;j=i+1;rec=x._asdict();rec.pop('Index',None)
            rec.update(entry_date=0,entry_status='no_next_bar',buy_price=np.nan,entry_gap=np.nan)
            for h in HORIZONS:rec.update({f'net_{h}':np.nan,f'exit_date_{h}':0,f'exit_price_{h}':np.nan,f'status_{h}':'no_entry',f'excess_{h}':np.nan})
            if j>=len(raw) or mi[j]!=mi[i]+1:rows.append(rec);continue
            rec.update(entry_date=int(dates[j]),buy_price=float(raw[j,0]/100),entry_gap=float(raw[j,0]/pre[j]-1))
            if int(dates[i]) not in st or int(dates[j]) not in st:
                rec['entry_status']='unknown_st';rows.append(rec);continue
            if st[int(dates[i])]==1 or st[int(dates[j])]==1:
                rec['entry_status']='st_excluded';rows.append(rec);continue
            rate=20 if code[3:6] in {'300','301','688'} else 10
            upper=max(math.floor((pre[j]*(100+rate)+50)/100),pre[j]+1)
            rec['entry_status']='upper_open' if raw[j,0]>=upper else 'filled'
            for h in HORIZONS:
                target=int(mi[j]+h-1);exit,why=executable_exit(raw,factor,mi,pre,st,code,j,target,cal)
                rec[f'status_{h}']=why
                if exit is None:continue
                z,adj,px,delay=exit;gross=adj/(raw[j,0]*factor[j])-1
                net=(1+gross)*(1-r.COST/2)/(1+r.COST/2)-1
                ix_exit=float(benchmark.loc[int(dates[z]),'close' if delay==0 else 'open'])
                ix_entry=float(benchmark.loc[int(dates[j]),'open']);bench=ix_exit/ix_entry-1
                rec.update({f'net_{h}':net,f'gross_{h}':gross,f'exit_date_{h}':int(dates[z]),f'exit_price_{h}':px,
                    f'delay_{h}':delay,f'benchmark_{h}':bench,f'excess_{h}':net-bench,
                    f'mae_{h}':float(np.min(raw[j:z+1,2]*factor[j:z+1])/(raw[j,0]*factor[j])-1)})
            rows.append(rec)
    pd.DataFrame(rows).sort_values(['signal_date','code']).to_pickle(OUT/'outcomes.pkl')
    print('outcomes',len(rows),flush=True)


def first_signals(e,key):
    # Once per visible wave. Filtering uses no subsequent execution/outcome.
    return e[e[key]].sort_values(['code','signal_date']).drop_duplicates(['code','floor_date','peak_date'])


def nonoverlap(e,h):
    last={};rows=[]
    for x in e.sort_values(['signal_date','code']).itertuples():
        if x.signal_date<=last.get(x.code,0):continue
        rows.append(x.event_id);last[x.code]=getattr(x,f'exit_date_{h}')
    return e[e.event_id.isin(rows)].copy()


def interval(g,h):
    if len(g)<2:return None
    # Six-month blocks reflect correlated market cohorts; approximate only.
    t=pd.DataFrame({'block':g.entry_date//10000*2+((g.entry_date//100%100)>6).astype(int),'net':g[f'net_{h}']})
    b=t.groupby('block').net.agg(['sum','count']);rng=np.random.default_rng(5152026)
    if len(b)<3:return None
    ix=rng.integers(0,len(b),size=(3000,len(b)))
    v=b['sum'].to_numpy()[ix].sum(axis=1)/b['count'].to_numpy()[ix].sum(axis=1)
    return dict(blocks=len(b),ci95=np.quantile(v,[.025,.975]).tolist())


def describe(g,h):
    values=np.sort(g[f'net_{h}'].to_numpy());n=len(values)
    s=stats(values);s.update(horizon=h,stocks=int(g.code.nunique()),entry_dates=int(g.entry_date.nunique()),
        date_equal_mean=float(g.groupby('entry_date')[f'net_{h}'].mean().mean()) if len(g) else None,
        mean_excess=float(g[f'excess_{h}'].mean()) if len(g) else None,
        excess_win=float((g[f'excess_{h}']>0).mean()) if len(g) else None,
        tail_p10=float(g[f'net_{h}'].quantile(.1)) if len(g) else None,
        median_mae=float(g[f'mae_{h}'].median()) if len(g) else None,
        block_interval=interval(g,h),delayed_exits=int((g.get(f'delay_{h}',pd.Series(dtype=float))>0).sum()),
        excluding_top1_mean=float(values[:-max(1,math.ceil(n*.01))].mean()) if n>1 else None,
        excluding_top5_mean=float(values[:-max(1,math.ceil(n*.05))].mean()) if n>1 else None)
    return s


def analyze():
    e=pd.read_pickle(OUT/'outcomes.pkl');report=dict(spec=SPEC,profiles={},case={},audit={})
    table=[]
    for key,q in PROFILES.items():
        signals=first_signals(e,key);filled=signals[signals.entry_status.eq('filled')]
        section=dict(label=q['label'],signals=len(signals),entry_status=signals.entry_status.value_counts().to_dict(),
            horizons=[],annual=[],common120=[],upper_open=[],groups=[])
        common=nonoverlap(filled[filled.status_120.eq('complete')],120)
        for h in HORIZONS:
            mature=filled[filled[f'status_{h}'].eq('complete')];chosen=nonoverlap(mature,h)
            s=describe(chosen,h);s.update(immature=int(filled[f'status_{h}'].eq('immature').sum()),
                unresolved=int(filled[f'status_{h}'].eq('unresolved').sum()),overlap_removed=len(mature)-len(chosen))
            section['horizons'].append(s);table.append(dict(profile=key,**{k:v for k,v in s.items() if not isinstance(v,dict)}))
            section['common120'].append(describe(common,h))
            for year,g in chosen.groupby(chosen.entry_date//10000):section['annual'].append(dict(year=int(year),**describe(g,h)))
            upper=signals[signals.entry_status.eq('upper_open')&signals[f'status_{h}'].eq('complete')]
            section['upper_open'].append(dict(horizon=h,**stats(upper[f'net_{h}'])))
            for field,bins,labels in [('rally_volume_ratio',[1,1.5,2,3,np.inf],['1-1.5','1.5-2','2-3','3+']),
                                      ('rally',[.15,.3,.5,np.inf],['15-30%','30-50%','50%+'])]:
                for label,g in chosen.groupby(pd.cut(chosen[field],bins,labels=labels,include_lowest=True),observed=True):
                    section['groups'].append(dict(field=field,bucket=str(label),**describe(g,h)))
            chosen.to_csv(OUT/f'{key}_{h}_trades.csv',index=False,encoding='utf-8-sig')
        signals.to_csv(OUT/f'{key}_signals.csv',index=False,encoding='utf-8-sig')
        report['profiles'][key]=section
    case=e[(e.code=='sz.300369')&(e.signal_date.between(20260818,20260828))].copy()
    case.to_csv(OUT/'nsfocus_case.csv',index=False,encoding='utf-8-sig')
    report['case']=json.loads(case.to_json(orient='records'))
    history=pd.read_csv(OUT/'sz.300369.csv')
    factor=np.cumprod(np.r_[1.,history.close.to_numpy()[:-1]/history.preclose.to_numpy()[1:]])
    history['adjusted_close']=history.close*factor/factor[-1]
    history['adjusted_low']=history.low*factor/factor[-1]
    past=history[history.date<='2026-06-29']
    low=past.loc[past.adjusted_low.idxmin()];close=past.loc[past.adjusted_close.idxmin()]
    report['case_history']=dict(start=history.date.iloc[0],through=history.date.iloc[-1],
        prior_low_date=low.date,prior_low=float(low.adjusted_low),prior_close_low_date=close.date,
        prior_close_low=float(close.adjusted_close),note='BaoStock preclose-linked adjusted history; signal floor intraday low 5.15, close 5.25.')
    # Keep all eligible signal rows for transparent latest/immature display.
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(table).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(table)[['profile','horizon','n','mean','median','win','payoff','mean_excess']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['inputs','build','statuses','outcomes','analyze'])
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.step=='statuses':statuses(args.shard,args.shards)
    else:globals()[args.step]()
