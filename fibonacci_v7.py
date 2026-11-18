"""Causal Fibonacci ladder controls, cash-inclusive returns, temporal selection.

Known v5 dry-up setups are the anchors. No order exists before the next day.
For each complete ladder use both OHLC and OLHC daily paths; retain the lower
terminal return. This brackets two path assumptions, not every possible path.
"""
import itertools
import json
import math
import numpy as np
import pandas as pd
from numba import njit
import confirm_v6 as c

OUT=c.d.bt.ROOT/'results/research_v7'
HORIZONS=c.HORIZONS
MODES={'retracement':'1.382 向下回撤','extension':'1.382 向上扩展'}
SPEC=dict(asof=20261116,wait=20,stop=.15,cost=c.d.r.COST,
    anchor='v5 first extreme dry-up setup; L is adjusted intraday low on the known floor date; H is highest adjusted intraday high from floor through the already known closing-peak date; anchors frozen thereafter',
    levels='H-r*(H-L) for r=.382,.5,.618; fourth H-1.382*(H-L) or L+1.382*(H-L)',
    orders='valid from next market session for 20 sessions, cancel unfilled at stop/exit; no retrospective fills; limit gap fills at open, extension gap fills at open only after a retracement leg filled; upper-limit opens skip buys; historical ST days skip buys',
    sizing='fixed budget fractions including fees, unfilled cash earns zero; stop uses weighted average filled cost, marks adjusted for corporate actions',
    risk='stop is 15% below weighted filled price, assessed before adding through an existing stop; cancel all orders on breach; sell eligible shares, T+1 shares queue until next sellable open; do not cancel at old floor so deep 1.382 remains testable',
    horizon='5/10/20/30/60/120 market sessions from first fill, entry day=1; no adds on expiry day',
    cohort='same mature setup cohort per horizon, full 20-session confirmation window plus next-open entry and horizon must fit snapshot; setup windows separated per stock',
    path='minimum terminal return of two global daily paths O-H-L-C and O-L-H-C; not a proof of fill or exhaustive worst case',
    selection='25 percentage point weight grid + equal thirds and 20/30/50; training exits through 2024-12-31; validation setups from 2025-01-01; best staged has >=2 nonzero legs, >=100 filled training setups, >=50% fill rate; choose highest mean total-budget net return only in training',
    caveat='historical data already explored; temporal holdout is not pristine unseen validation; survivor universe and multiple comparisons remain')


@njit(cache=True)
def simulate(raw,factor,mi,st,upper,locked,i,levels,weights,extension,h,path,cost=.0035):
    cash=1.;units=0.;basis=0.;spent=0.;first=-1;last=-1;mask=0;liquidating=0;stop_hit=0;worst=0.
    log=np.zeros((8,6));nlog=0;expiry=-1
    for z in range(i+1,len(raw)):
        if first<0 and mi[z]>mi[i]+20:break
        old=units;o,hi,lo,cl=raw[z];fac=factor[z]
        # Sell queued shares / gap breaches before any new averaging-down order.
        stop_tick=math.floor(basis/units*.85/fac+1e-7) if units>1e-12 else -1.
        if units>1e-12 and (liquidating or o<=stop_tick or mi[z]>expiry):
            if not liquidating:liquidating=1 if o<=stop_tick else 2
            if liquidating==1:stop_hit=1
            if not locked[z]:
                cash+=units*o*fac*(1-cost/2);log[nlog]=np.array([z,o,units,-1,liquidating,stop_tick]);nlog+=1
                units=0.;basis=0.;last=z
                return np.array([cash-1,first,last,mask,spent,stop_hit,min(worst,cash-1),0,path]),log[:nlog]
        canbuy=not liquidating and mi[z]<=mi[i]+20 and (first<0 or mi[z]<expiry) and st[z]==0 and not upper[z]
        if canbuy:
            for leg in range(4):
                if weights[leg]<=0 or mask&(1<<leg):continue
                isbreak=extension and leg==3
                if isbreak and not (mask&7):continue
                level=math.ceil(levels[leg]/fac-1e-7) if isbreak else math.floor(levels[leg]/fac+1e-7)
                if level<=0:continue
                if (isbreak and o>=level) or (not isbreak and o<=level):
                    q=weights[leg]/(o*fac*(1+cost/2));cash-=weights[leg];spent+=weights[leg]
                    units+=q;basis+=q*o*fac;mask|=1<<leg
                    if first<0:first=z;expiry=mi[z]+h-1
                    log[nlog]=np.array([z,o,q,1,leg,basis/units*.85/fac]);nlog+=1
        points=np.array([o,hi,lo,cl]) if path==0 else np.array([o,lo,hi,cl])
        for part in range(3):
            current=points[part];end=points[part+1];down=end<current
            for iteration in range(8):
                action=-1;hit=end
                if not liquidating and units>1e-12 and down:
                    stop_tick=math.floor(basis/units*.85/fac+1e-7)
                    if end<=stop_tick<=current:action=4;hit=stop_tick
                if canbuy and not liquidating:
                    for leg in range(4):
                        if weights[leg]<=0 or mask&(1<<leg):continue
                        isbreak=extension and leg==3
                        if isbreak and not (mask&7):continue
                        level=math.ceil(levels[leg]/fac-1e-7) if isbreak else math.floor(levels[leg]/fac+1e-7)
                        if level<=0:continue
                        crosses=(down and not isbreak and end<=level<=current) or (not down and isbreak and current<=level<=end)
                        # At equal price a stop takes priority over a new buy.
                        if crosses and (action<0 or (down and level>hit) or (not down and level<hit)):
                            action=leg;hit=level
                if action<0:break
                worst=min(worst,cash+units*hit*fac-1)
                if action==4:
                    liquidating=1;stop_hit=1;canbuy=False
                    if not locked[z] and old>1e-12:
                        q=min(old,units);cash+=q*hit*fac*(1-cost/2);basis*=max(0.,units-q)/units;units-=q;old-=q;last=z
                        log[nlog]=np.array([z,hit,q,-1,1,hit]);nlog+=1
                    if units<1e-12:return np.array([cash-1,first,last,mask,spent,stop_hit,min(worst,cash-1),0,path]),log[:nlog]
                else:
                    q=weights[action]/(hit*fac*(1+cost/2));cash-=weights[action];spent+=weights[action]
                    units+=q;basis+=q*hit*fac;mask|=1<<action
                    if first<0:first=z;expiry=mi[z]+h-1
                    log[nlog]=np.array([z,hit,q,1,action,basis/units*.85/fac]);nlog+=1
                current=hit
            worst=min(worst,cash+units*end*fac-1)
        if units>1e-12 and mi[z]>=expiry:
            if not liquidating:liquidating=2
            if not locked[z] and old>1e-12:
                q=min(old,units);cash+=q*cl*fac*(1-cost/2);basis*=max(0.,units-q)/units;units-=q;old-=q;last=z
                log[nlog]=np.array([z,cl,q,-1,liquidating,0.]);nlog+=1
                if units<1e-12:return np.array([cash-1,first,last,mask,spent,stop_hit,min(worst,cash-1),0,path]),log[:nlog]
    return np.array([np.nan if units>1e-12 else cash-1,first,last,mask,spent,stop_hit,worst,1.0*(units>1e-12),path]),log[:nlog]


def weight_grid():
    weights=[tuple(x/4 for x in v) for v in itertools.product(range(5),repeat=4) if sum(v)==4]
    return np.array(weights+[(1/3,1/3,1/3,0),(.2,.3,.5,0)])


def build():
    OUT.mkdir(exist_ok=True)
    panel=pd.read_pickle(c.d.r.OUT/'panel.pkl');flags=pd.read_pickle(c.OUT/'status_maps.pkl')
    cal=pd.read_csv(c.d.r.OUT/'market.csv').date.to_numpy();setups=c.d.first_signals(pd.read_pickle(c.d.OUT/'events.pkl'),'extreme')
    weights=weight_grid();results=[];inputs={};meta=[]
    for x in setups.itertuples():
        pack=panel[x.code];dates,raw,close,factor,pre,mi,*_=pack;st=flags.get(x.code,{})
        if st.get(int(x.signal_date),1)!=0:continue
        # Missing historical status never grants entry permission.
        statuses=np.array([st.get(int(day),1) for day in dates],np.int8)
        if x.code not in inputs:
            rate=20 if x.code[3:6] in {'300','301','688'} else 10
            upper=raw[:,0]>=np.maximum(np.floor((pre*(100+rate)+50)/100),pre+1)
            locked=np.array([c.lower_locked(raw,pre,dates,st,x.code,j) for j in range(len(raw))])
            inputs[x.code]=(raw,factor,mi,statuses,upper,locked)
        f=float(raw[x.floor_index,2]*factor[x.floor_index])
        hp=int(x.floor_index+np.argmax(raw[x.floor_index:x.peak_index+1,1]*factor[x.floor_index:x.peak_index+1]))
        p=float(raw[hp,1]*factor[hp]);span=p-f
        for mode in MODES:
            levels=np.array([p-.382*span,p-.5*span,p-.618*span,p-1.382*span if mode=='retracement' else f+1.382*span])
            for h in HORIZONS:
                # Every configuration receives the same fully observable setup.
                end=int(x.signal_index+20+h)
                if end>=len(cal):continue
                for wi,w in enumerate(weights):
                    a,la=simulate(*inputs[x.code],int(x.stock_index),levels,w,mode=='extension',h,0)
                    b,lb=simulate(*inputs[x.code],int(x.stock_index),levels,w,mode=='extension',h,1)
                    choice=1 if not np.isfinite(a[0]) or (np.isfinite(b[0]) and b[0]<a[0]) else 0
                    v=b if choice else a
                    # If either path remains unresolved, do not select the other as known.
                    if not np.isfinite(a[0]) or not np.isfinite(b[0]):v[0]=np.nan
                    first=int(v[1]);last=int(v[2]);results.append(dict(event_id=x.event_id,code=x.code,name=x.name,
                        setup_date=int(x.signal_date),setup_index=int(x.signal_index),observation_end=int(cal[end]),mode=mode,horizon=h,weight_id=wi,
                        net=float(v[0]),entry_date=int(dates[first]) if first>=0 else 0,exit_date=int(dates[last]) if last>=0 else 0,
                        mask=int(v[3]),spent=float(v[4]),stopped=int(v[5]),mae=float(v[6]),unresolved=int(v[7]),path=choice,
                        path_gap=float(abs(a[0]-b[0])) if np.isfinite(a[0]) and np.isfinite(b[0]) else np.nan))
        meta.append(dict(x._asdict(),fib_low=f,fib_high=p,fib_high_index=hp,fib_high_date=int(dates[hp])))
        if len(meta)%200==0:print('fib setups',len(meta),'rows',len(results),flush=True)
    pd.DataFrame(results).to_pickle(OUT/'grid.pkl');pd.DataFrame(meta).to_pickle(OUT/'setups.pkl')
    np.save(OUT/'weights.npy',weights)
    (OUT/'spec.json').write_text(json.dumps(SPEC,ensure_ascii=False,indent=2),encoding='utf-8')
    analyze()


def metrics(g):
    a=g.net.to_numpy();s=c.d.stats(a);filled=g[g.spent>1e-9]
    s.update(filled=len(filled),fill_rate=len(filled)/len(g) if len(g) else None,
        filled_win=float((filled.net>0).mean()) if len(filled) else None,
        filled_payoff=c.d.stats(filled.net)['payoff'],avg_spent=float(g.spent.mean()) if len(g) else None,
        stopped=int(g.stopped.sum()),median_mae=float(g.mae.median()) if len(g) else None,
        path_ambiguous=int((g.path_gap>1e-8).sum()),mean_path_gap=float(g.path_gap.mean()) if len(g) else None,
        fourth_fills=int(((g['mask'].astype(int)&8)>0).sum()),
        multi_fills=int(g['mask'].map(lambda n:int(n).bit_count()>=2).sum()),stocks=int(g.code.nunique()))
    return s


def month_interval(g,column='net'):
    if len(g)<2:return None
    grouped=g.assign(month=g.setup_date//100).groupby('month')[column].agg(['sum','count'])
    if len(grouped)<3:return None
    rng=np.random.default_rng(71826);ix=rng.integers(0,len(grouped),(2000,len(grouped)))
    samples=grouped['sum'].to_numpy()[ix].sum(axis=1)/grouped['count'].to_numpy()[ix].sum(axis=1)
    return dict(months=len(grouped),ci95=np.quantile(samples,[.025,.975]).tolist())


def analyze():
    full=pd.read_pickle(OUT/'grid.pkl');weights=np.load(OUT/'weights.npy');report=dict(spec=SPEC,weights=weights.tolist(),horizons=[],selected=[])
    reports=[];kept=[]
    confirms=pd.read_pickle(c.OUT/'outcomes.pkl')
    direct=c.d.first_signals(pd.read_pickle(c.d.OUT/'outcomes.pkl'),'extreme').copy();direct['variant']='direct'
    controls=pd.concat([direct,confirms],ignore_index=True)
    for h in HORIZONS:
        g=full[full.horizon==h].copy();unique=g.drop_duplicates('event_id').sort_values(['setup_date','code'])
        ids=[];last={};latest=g.groupby('event_id').exit_date.max().to_dict()
        for event,end in controls.groupby('event_id')[f'exit_date_{h}'].max().items():
            latest[event]=max(latest.get(event,0),int(end) if pd.notna(end) else 0)
        for x in unique.itertuples():
            if x.setup_date<=last.get(x.code,0):continue
            ids.append(x.event_id);last[x.code]=max(x.observation_end,latest[x.event_id])
        g=g[g.event_id.isin(ids)]
        bad=np.union1d(g[g.net.isna()].event_id.unique(),controls.loc[controls.entry_status.eq('filled')&~controls[f'status_{h}'].eq('complete'),'event_id'].unique())
        g=g[~g.event_id.isin(bad)]
        # Temporal labels depend on observability, never outcome or chosen weights.
        g['split']=np.where((g.observation_end<=20241231)&((g.exit_date==0)|(g.exit_date<=20241231)),'train',np.where(g.setup_date>=20250101,'validation','embargo'))
        # If one configuration exits beyond training, embargo the entire setup.
        embargo=g.loc[(g.setup_date<20250101)&(g.split.ne('train')|g.event_id.map(latest).gt(20241231)),'event_id'].unique();g.loc[g.event_id.isin(embargo),'split']='embargo'
        kept.append(g)
        section=dict(horizon=h,setups=g.event_id.nunique(),unresolved_setups=len(set(ids)&set(bad)),modes={},controls={})
        cohort=g.drop_duplicates('event_id')[['event_id','split','code']]
        for variant in c.VARIANTS:
            source=controls[controls.variant==variant].set_index('event_id')
            records=[]
            for row in cohort.itertuples():
                hit=source.loc[row.event_id] if row.event_id in source.index else None
                filled=hit is not None and hit.entry_status=='filled' and hit[f'status_{h}']=='complete'
                records.append(dict(split=row.split,net=float(hit[f'net_{h}']) if filled else 0.,spent=1. if filled else 0.,
                    stopped=int(filled and str(hit[f'reason_{h}'] if f'reason_{h}' in hit else '').startswith('stop_')),
                    mae=float(hit[f'mae_{h}']) if filled else 0.,path_gap=0.,mask=1 if filled else 0,code=row.code))
            t=pd.DataFrame(records);section['controls'][variant]={split:metrics(t[t.split==split]) for split in ('train','validation')}
        for mode in MODES:
            m=g[g['mode']==mode];ranking=[]
            for wi,w in enumerate(weights):
                one=m[m.weight_id==wi];row=dict(mode=mode,horizon=h,weight_id=wi,weights=w.tolist(),
                    train=metrics(one[one.split=='train']),validation=metrics(one[one.split=='validation']),all=metrics(one))
                ranking.append(row);reports.append(row)
            eligible=[x for x in ranking if sum(z>0 for z in x['weights'])>=2 and x['train']['filled']>=100 and x['train']['fill_rate']>=.5]
            selected=max(eligible,key=lambda x:x['train']['mean']) if eligible else None
            equal4=next(x for x in ranking if np.allclose(x['weights'],[.25]*4));equal3=next(x for x in ranking if np.allclose(x['weights'],[1/3]*3+[0]))
            single=max((x for x in ranking if sum(z>0 for z in x['weights'])==1 and x['train']['filled']>=100 and x['train']['fill_rate']>=.5),key=lambda x:x['train']['mean'],default=None)
            section['modes'][mode]=dict(selected=selected,equal4=equal4,equal3=equal3,best_single=single)
            if selected:
                picked=m[m.weight_id==selected['weight_id']].copy()
                selected['intervals']={split:month_interval(picked[picked.split==split]) for split in ('train','validation')}
                control=controls[controls.variant=='stop'].set_index('event_id')
                picked['delta']=picked.apply(lambda z:z.net-(float(control.loc[z.event_id,f'net_{h}']) if z.event_id in control.index and control.loc[z.event_id,'entry_status']=='filled' else 0.),axis=1)
                valid=picked[picked.split=='validation']
                selected['paired_stop_delta']=dict(mean=float(valid.delta.mean()) if len(valid) else None,interval=month_interval(valid,'delta'))
                selected['annual']=[dict(year=int(year),**metrics(group)) for year,group in picked.groupby(picked.setup_date//10000)]
                report['selected'].append(selected)
                m[m.weight_id==selected['weight_id']].to_csv(OUT/f'{mode}_{h}_selected.csv',index=False,encoding='utf-8-sig')
                print(mode,h,'weights',selected['weights'],'train',selected['train']['mean'],'valid',selected['validation']['mean'],'n',selected['validation']['n'],flush=True)
        report['horizons'].append(section)
    pd.concat(kept).to_pickle(OUT/'evaluated.pkl')
    report['grid']=reports
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':build()
