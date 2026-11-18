"""Frozen dry-up setups, first volume-positive confirmation, and hard 15% stop.

No return-based parameter selection. A 20-market-session confirmation expiry
is an explicit operational assumption. All horizon cohorts must mature even
when a stop has already closed a trade, to avoid asymmetric right censoring.
"""
import json
import math
import numpy as np
import pandas as pd
import dryup_v5 as d

OUT=d.bt.ROOT/'results/research_v6'
HORIZONS=d.HORIZONS
WAIT=20
STOP=.15
VARIANTS={'direct':'缩量后直接买入','confirm':'放量阳线确认','stop':'放量确认 + 15%止损'}
SPEC=dict(d.SPEC,confirmation_wait=WAIT,
    confirmation='first later bullish body (close > open), volume > preceding 5 AND 10 means; within 20 market sessions; cancel after a close below original floor',
    entry='confirmation at close; next market-session open; ST on setup/confirmation/entry excluded; upper-limit open skipped',
    stop='buy adjusted price * 0.85; at/below rounded-down tick triggers full exit; T+1 entry-day breach latched to next sellable open; gap/locked-down exits may lose more than 15%',
    maturity='all horizon cohorts require full calendar maturity even if stopped early; cash earns zero after exit',
    comparison='matched stop/no-stop comparison uses the same mature nonoverlapping confirmation entries; benchmark ends at horizon, not early stop',
    parameter_optimization=False)


def confirmation(raw,close,volume,mi,i,f,wait=WAIT):
    for z in range(i+1,len(raw)):
        if mi[z]-mi[i]>wait:return None,'expired'
        if close[z]<close[f]:return None,'floor_broken'
        if raw[z,3]>raw[z,0] and volume[z]>np.mean(volume[z-5:z]) and volume[z]>np.mean(volume[z-10:z]):
            return z,'confirmed'
    return None,'pending' if mi[-1]-mi[i]<wait else 'expired'


def lower_locked(raw,pre,dates,st,code,z):
    rate=20 if code[3:6] in {'300','301','688'} else 5 if st.get(int(dates[z]))==1 and dates[z]<d.bt.ST_LIMIT_CHANGE else 10
    down=min(math.floor((pre[z]*(100-rate)+50)/100),pre[z]-1)
    return raw[z,1]==raw[z,2] and raw[z,2]<=down


def hard_exit(raw,factor,mi,pre,dates,st,code,j,target,cal,stop=STOP):
    # Closed losers must not enter an otherwise immature cohort early.
    if target>=len(cal):return None,'immature'
    ref=raw[j,0]*factor[j]*(1-stop)
    tick=lambda z:math.floor(ref/factor[z]+1e-7)
    pending=raw[j,2]<=tick(j)
    trigger=int(dates[j]) if pending else 0
    blocked=0
    for z in range(j+1,len(raw)):
        o,hi,low,c=raw[z];level=tick(z);age=int(mi[z]-mi[j]+1)
        breach=low<=level
        if lower_locked(raw,pre,dates,st,code,z):
            if breach and not trigger:trigger=int(dates[z])
            pending|=breach
            blocked+=int(pending or mi[z]>=target)
            continue
        if pending:price=o;reason='stop_queued'
        elif o<=level:price=o;reason='stop_gap';trigger=int(dates[z])
        elif mi[z]>target:price=o;reason='expiry_delayed'
        elif breach:price=level;reason='stop_intraday';trigger=int(dates[z])
        elif mi[z]==target:price=c;reason='expiry'
        else:continue
        return dict(z=z,adjusted=float(price*factor[z]),price=float(price/100),reason=reason,
                    trigger_date=trigger,blocked=blocked,entry_breach=bool(raw[j,2]<=tick(j)),
                    age=age,stop_ref=float(ref/factor[j]/100)), 'complete'
    return None,'unresolved'


def build():
    OUT.mkdir(exist_ok=True)
    panel=pd.read_pickle(d.r.OUT/'panel.pkl');vols=d.volume_data(panel)
    cache=OUT/'status_maps.pkl'
    flags=pd.read_pickle(cache) if cache.exists() else d.status_maps()
    if not cache.exists():pd.to_pickle(flags,cache)
    cal=pd.read_csv(d.r.OUT/'market.csv').date.to_numpy()
    bm=pd.read_csv(d.OUT/'sh.000300.csv');bm['date']=bm.date.str.replace('-','').astype(int);bm=bm.set_index('date')
    setups=d.first_signals(pd.read_pickle(d.OUT/'events.pkl'),'extreme')
    rows=[];tracking=[]
    for x in setups.itertuples():
        dates,raw,close,factor,pre,mi,*_=panel[x.code];st=flags.get(x.code,{})
        z,why=confirmation(raw,close,vols[x.code][:,0],mi,x.stock_index,x.floor_index)
        tracking.append(dict(event_id=x.event_id,code=x.code,name=x.name,dry_date=x.signal_date,
                             confirmation_date=int(dates[z]) if z is not None else 0,status=why))
        if z is None:continue
        rec=x._asdict();rec.pop('Index',None)
        rec.update(dry_date=x.signal_date,dry_stock_index=x.stock_index,signal_date=int(dates[z]),stock_index=z,
            signal_index=int(mi[z]),signal_return=float(raw[z,3]/pre[z]-1),price=float(raw[z,3]/100),
            floor_price=float(close[x.floor_index]/factor[z]/100),peak_price=float(close[x.peak_index]/factor[z]/100),
            confirm_wait=int(mi[z]-mi[x.stock_index]),volume5=float(vols[x.code][z,0]/vols[x.code][z-5:z,0].mean()),
            volume10=float(vols[x.code][z,0]/vols[x.code][z-10:z,0].mean()),
            entry_status='no_next_bar',entry_date=0,buy_price=np.nan,stop_price=np.nan,stop_signal_price=np.nan)
        j=z+1
        if j>=len(raw) or mi[j]!=mi[z]+1:
            for variant in ('confirm','stop'):rows.append(dict(rec,variant=variant))
            continue
        rec.update(entry_date=int(dates[j]),buy_price=float(raw[j,0]/100),
            stop_price=float(math.floor(raw[j,0]*(1-STOP)+1e-7)/100),
            stop_signal_price=float(raw[j,0]*factor[j]*(1-STOP)/factor[z]/100))
        if any(int(k) not in st for k in (x.signal_date,dates[z],dates[j])):rec['entry_status']='unknown_st'
        elif any(st[int(k)] for k in (x.signal_date,dates[z],dates[j])):rec['entry_status']='st_excluded'
        else:
            rate=20 if x.code[3:6] in {'300','301','688'} else 10
            upper=max(math.floor((pre[j]*(100+rate)+50)/100),pre[j]+1)
            rec['entry_status']='upper_open' if raw[j,0]>=upper else 'filled'
        records={key:dict(rec,variant=key) for key in ('confirm','stop')}
        for h in HORIZONS:
            target=int(mi[j]+h-1)
            base,status=d.executable_exit(raw,factor,mi,pre,st,x.code,j,target,cal)
            stopped,stop_status=hard_exit(raw,factor,mi,pre,dates,st,x.code,j,target,cal)
            # Same planned horizon for the benchmark, including cash after stops.
            benchmark=float(bm.loc[int(cal[target]),'close']/bm.loc[int(dates[j]),'open']-1) if target<len(cal) else np.nan
            for key,outcome,state in [('confirm',base,status),('stop',stopped,stop_status)]:
                r=records[key];r.update({f'net_{h}':np.nan,f'exit_date_{h}':0,f'exit_price_{h}':np.nan,
                    f'status_{h}':state,f'benchmark_{h}':benchmark,f'reason_{h}':'',f'excess_{h}':np.nan,
                    f'mae_{h}':np.nan,f'delay_{h}':0,f'stop_trigger_{h}':0})
                if outcome is None:continue
                if key=='confirm':k,adj,px,delay=outcome;reason='expiry_delayed' if delay else 'expiry';trigger=0
                else:
                    k=outcome['z'];adj=outcome['adjusted'];px=outcome['price'];reason=outcome['reason'];trigger=outcome['trigger_date']
                    delay=max(0,int(mi[k]-target));r[f'blocked_{h}']=outcome['blocked'];r['entry_breach']=outcome['entry_breach']
                gross=adj/(raw[j,0]*factor[j])-1;net=(1+gross)*(1-d.r.COST/2)/(1+d.r.COST/2)-1
                # On an intraday stop bar the later daily low is not held exposure.
                lows=raw[j:k+1,2]*factor[j:k+1] if reason=='expiry' else raw[j:k,2]*factor[j:k]
                mae=min(float(lows.min()) if len(lows) else adj,adj)/(raw[j,0]*factor[j])-1
                r.update({f'net_{h}':net,f'gross_{h}':gross,f'exit_date_{h}':int(dates[k]),f'exit_price_{h}':px,
                    f'sale_signal_price_{h}':float(adj/factor[z]/100),f'reason_{h}':reason,f'excess_{h}':net-benchmark,
                    f'mae_{h}':mae,f'delay_{h}':delay,f'stop_trigger_{h}':trigger})
        rows.extend(records.values())
    confirmed=pd.DataFrame(rows)
    for h in HORIZONS:
        for field,value in [('status','no_entry'),('net',np.nan),('exit_date',0),('benchmark',np.nan),('excess',np.nan),('mae',np.nan),('reason',''),('delay',0)]:
            col=f'{field}_{h}';confirmed[col]=confirmed[col].fillna(value)
    confirmed.to_pickle(OUT/'outcomes.pkl')
    pd.DataFrame(tracking).to_csv(OUT/'setups.csv',index=False,encoding='utf-8-sig')
    (OUT/'spec.json').write_text(json.dumps(SPEC,ensure_ascii=False,indent=2),encoding='utf-8')
    print('confirmation status',pd.DataFrame(tracking).status.value_counts().to_dict(),flush=True)
    analyze()


def describe(g,h):
    s=d.describe(g,h)
    stopped=g.get(f'reason_{h}',pd.Series('',index=g.index)).str.startswith('stop_')
    s.update(stopped=int(stopped.sum()),stop_rate=float(stopped.mean()) if len(g) else None)
    return s


def analyze():
    confirmed=pd.read_pickle(OUT/'outcomes.pkl')
    baseline=d.first_signals(pd.read_pickle(d.OUT/'outcomes.pkl'),'extreme').copy();baseline['variant']='direct';baseline['dry_date']=baseline.signal_date
    allrows=pd.concat([baseline,confirmed],ignore_index=True)
    for h in HORIZONS:allrows[f'reason_{h}']=allrows[f'reason_{h}'].fillna('expiry')
    report=dict(spec=SPEC,profiles={},matched=[],case=[],audit={})
    report['setup_status']=pd.read_csv(OUT/'setups.csv').status.value_counts().to_dict()
    table=[]
    for key,label in VARIANTS.items():
        signals=allrows[allrows.variant.eq(key)].copy();filled=signals[signals.entry_status.eq('filled')]
        section=dict(label=label,signals=len(signals),entry_status=signals.entry_status.value_counts().to_dict(),horizons=[],common120=[],annual=[],groups=[])
        common=d.nonoverlap(filled[filled.status_120.eq('complete')],120)
        for h in HORIZONS:
            mature=filled[filled[f'status_{h}'].eq('complete')];chosen=d.nonoverlap(mature,h)
            s=describe(chosen,h);s.update(immature=int(filled[f'status_{h}'].eq('immature').sum()),unresolved=int(filled[f'status_{h}'].eq('unresolved').sum()),overlap_removed=len(mature)-len(chosen))
            section['horizons'].append(s);table.append(dict(variant=key,**{k:v for k,v in s.items() if not isinstance(v,dict)}))
            section['common120'].append(describe(common,h))
            for year,g in chosen.groupby(chosen.entry_date//10000):section['annual'].append(dict(year=int(year),**describe(g,h)))
            for bucket,g in chosen.groupby(pd.cut(chosen.rally_volume_ratio,[1,1.5,2,3,np.inf],labels=['1-1.5','1.5-2','2-3','3+']),observed=True):
                section['groups'].append(dict(field='rally_volume_ratio',bucket=str(bucket),**describe(g,h)))
            chosen.to_csv(OUT/f'{key}_{h}_trades.csv',index=False,encoding='utf-8-sig')
        signals.to_csv(OUT/f'{key}_signals.csv',index=False,encoding='utf-8-sig');report['profiles'][key]=section
    for h in HORIZONS:
        a=pd.read_csv(OUT/f'confirm_{h}_trades.csv');b=confirmed[(confirmed.variant=='stop')&confirmed.event_id.isin(a.event_id)&confirmed[f'status_{h}'].eq('complete')]
        a=a[a.event_id.isin(b.event_id)]
        report['matched'].append(dict(horizon=h,confirm=describe(a,h),stop=describe(b,h),mean_delta=float(b[f'net_{h}'].mean()-a[f'net_{h}'].mean()) if len(a) else None))
    case=allrows[(allrows.code=='sz.300369')&allrows.dry_date.eq(20260825)]
    report['case']=json.loads(case.to_json(orient='records'))
    case.to_csv(OUT/'nsfocus_case.csv',index=False,encoding='utf-8-sig')
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(table).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(table)[['variant','horizon','n','mean','win','payoff','stopped']].to_string(index=False),flush=True)


if __name__=='__main__':build()
