"""Audit selected research candidates with finite capital and sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import research_v2 as r

def stats(a):
    a=np.asarray(a,dtype=float);w=a[a>0];l=a[a<0]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                win=float((a>0).mean()) if len(a) else None,
                payoff=float(w.mean()/-l.mean()) if len(w) and len(l) else None,
                avg_win=float(w.mean()) if len(w) else None,avg_loss=float(-l.mean()) if len(l) else None,
                profit_factor=float(w.sum()/-l.sum()) if len(l) else None,
                worst=float(a.min()) if len(a) else None,best=float(a.max()) if len(a) else None)

def config_mask(e,c):
    return ((e.price>=3)&(e.amount20>=3e7)&r.families(e)[c['family']]
            &r.regimes(e)[c['regime']]&r.filters(e)[c['filter']]&(e.gap<=c['gap_max']))

def trades_from(e,data,mask,mode,k):
    ix=np.lexsort((e.code,e.signal_date));chosen=[];last={}
    for idx in ix:
        x=e.iloc[idx];sell=int(data['sold'][idx,mode,k]);year=x.entry_date//10000
        if not mask.iloc[idx] or not x.complete30 or x.planned30//10000!=year or sell//10000!=year:continue
        if not np.isfinite(data['gross'][idx,mode,k]) or x.signal_date<=last.get(x.code,0):continue
        chosen.append(idx);last[x.code]=sell
    t=e.iloc[chosen].copy()
    t['event_id']=chosen;t['buy_price']=data['buy'][chosen,mode]
    t['exit_date']=data['sold'][chosen,mode,k]
    t['gross']=data['gross'][chosen,mode,k].astype(float)
    t['net']=(1+t.gross)*(1-r.COST/2)/(1+r.COST/2)-1
    return t

def clustered_interval(t,iterations=5000):
    # Month blocks retain cross-sectional and short-horizon dependence.
    group=t.assign(month=t.entry_date//100).groupby('month').net.agg(['sum','count'])
    rng=np.random.default_rng(21823)
    sampled=rng.integers(0,len(group),size=(iterations,len(group)))
    means=group['sum'].to_numpy()[sampled].sum(axis=1)/group['count'].to_numpy()[sampled].sum(axis=1)
    return dict(months=len(group),ci95=np.quantile(means,[.025,.975]).tolist(),
                probability_mean_positive=float((means>0).mean()),
                warning='Conditional on selected rule; does not correct multiple testing or survivorship.')

def portfolio(e,data,panel,mask,mode,k,slots=10,rank='liquid',cost=None):
    """Orders chosen before observing fills; fixed slots, no leverage, daily MTM.

    Use adjusted units to account for splits/distributions consistently with
    trade-level total returns. Unfilled orders expire and their reserved budget
    is not transferred to other same-day orders. Portfolio periods use the same
    30-day maturity and year purging as the search.
    """
    cost=r.COST if cost is None else cost
    cal=data['calendar'];lookup={code:{int(day):i for i,day in enumerate(p[0])} for code,p in panel.items()}
    years=e.entry_date//10000
    eligible=mask&e.complete30.eq(1)&(e.planned30//10000).eq(years)
    selected=e.loc[eligible].copy();selected['event_id']=selected.index
    byday={int(d):x for d,x in selected.groupby('entry_date')}
    equity=1.;cash=1.;positions={};curve=[];executions=[];reservations=0
    rng=np.random.default_rng(45267)
    for day in cal[cal>=20220101]:
        # Reserve only cash available before the open. Intraday sale proceeds
        # cannot fund a same-day opening order in this model.
        available_slots=slots-len(positions)
        orders=byday.get(int(day))
        if available_slots>0 and cash>0 and orders is not None:
            orders=orders.loc[~orders.code.isin(positions)]
            if rank=='liquid': orders=orders.sort_values(['amount20','code'],ascending=[False,True])
            elif rank=='code':orders=orders.sort_values('code')
            elif rank=='random':orders=orders.iloc[rng.permutation(len(orders))]
            orders=orders.head(available_slots)
            allocation=min(equity/slots,cash/max(1,len(orders)))
            reservations+=len(orders)
            for x in orders.itertuples():
                idx=x.event_id;buy=data['buy'][idx,mode];sell=int(data['sold'][idx,mode,k])
                if not np.isfinite(buy) or not sell:continue
                dates,raw,adj,factor,*_=panel[x.code]
                j=lookup[x.code].get(int(day))
                if j is None:continue
                entry_adjusted=buy*100*factor[j]
                units=allocation/(entry_adjusted*(1+cost/2))
                positions[x.code]=dict(units=units,entry=entry_adjusted,allocation=allocation,
                                       event_id=idx,sell=sell,entry_date=int(day),last_price=entry_adjusted)
                cash-=allocation
        # Settle exits after new orders. This is conservative for opening exits.
        for code,p in list(positions.items()):
            dates,raw,adj,factor,*_=panel[code]
            j=lookup[code].get(int(day))
            if j is not None:p['last_price']=adj[j]
            if day==p['sell']:
                gross=float(data['gross'][p['event_id'],mode,k])
                value=p['units']*p['entry']*(1+gross)*(1-cost/2)
                cash+=value
                executions.append(dict(event_id=p['event_id'],code=code,entry_date=p['entry_date'],
                                       exit_date=p['sell'],weight=p['allocation'],gross=gross,
                                       net=value/p['allocation']-1))
                del positions[code]
        invested=sum(p['units']*p['last_price'] for p in positions.values())
        equity=cash+invested
        assert cash>=-1e-8 and len(positions)<=slots
        curve.append(dict(date=int(day),equity=equity,cash=cash,exposure=invested/equity,positions=len(positions)))
    curve=pd.DataFrame(curve);trades=pd.DataFrame(executions)
    assert not positions, 'Incomplete trades remain in purged portfolio'
    periods=(pd.Timestamp(str(int(curve.date.iloc[-1])))-pd.Timestamp(str(int(curve.date.iloc[0])))).days/365.25
    annual=[];previous=1.
    for year,g in curve.groupby(curve.date//10000):
        annual.append(dict(year=int(year),return_=float(g.equity.iloc[-1]/previous-1),
                           drawdown=float((g.equity/np.maximum.accumulate(np.r_[previous,g.equity])[1:]-1).min()),
                           trades=int((trades.entry_date//10000==year).sum())))
        previous=g.equity.iloc[-1]
    report=dict(total_return=float(equity-1),cagr=float(equity**(1/periods)-1),
                max_drawdown=float((curve.equity/curve.equity.cummax()-1).min()),
                average_exposure=float(curve.exposure.mean()),reservations=int(reservations),
                filled=len(trades),fill_rate=len(trades)/reservations if reservations else 0.,annual=annual,
                trade_stats=stats(trades.net),slots=slots,rank=rank,cost=cost)
    return report,curve,trades

def main():
    e=pd.read_pickle(r.OUT/'events.pkl');data=dict(np.load(r.OUT/'outcomes.npz'));panel=pd.read_pickle(r.OUT/'panel.pkl')
    cfgs=json.loads((r.OUT/'configs.json').read_text())
    picks=json.loads((r.OUT/'picks_before_2026.json').read_text())
    primary=picks[0];nextopen=next(p for p in picks if p['buy']=='next_open')
    reports={};yearly=[];curves=[]
    for label,pick in [('primary',primary),('next_open',nextopen),('primary_stop7',{**primary,'exit':41})]:
        c=cfgs[pick['config']];mode=r.BUYS.index(c['buy']);k=pick['exit'];mask=config_mask(e,c)
        t=trades_from(e,data,mask,mode,k)
        t.to_csv(r.OUT/f'{label}_trades.csv',index=False,encoding='utf-8-sig')
        ann=[]
        for year,g in t.groupby(t.entry_date//10000):
            s=dict(strategy=label,year=int(year),**stats(g.net));ann.append(s);yearly.append(s)
        summary=stats(t.net)
        after=t.loc[t.entry_date>=20240101];t2026=t.loc[t.entry_date>=20260101]
        report=dict(config=c,exit_rule=r.EXITS[k],all=summary,annual=ann,
                    check_2024_onward=stats(after.net),check_2026=stats(t2026.net),
                    monthly_bootstrap_all=clustered_interval(t),monthly_bootstrap_2026=clustered_interval(t2026),
                    without_top5=stats(t.net.sort_values().iloc[:-5]),
                    without_top5_2026=stats(t2026.net.sort_values().iloc[:-5]),
                    signals=int(mask.sum()),independent_dates=int(t.entry_date.nunique()),
                    top_dates=t.groupby('entry_date').net.agg(['size','mean']).sort_values('size',ascending=False).head(8).reset_index().to_dict('records'))
        variants=[]
        for slots,rank,cost in [(10,'liquid',r.COST),(5,'liquid',r.COST),(20,'liquid',r.COST),(10,'code',r.COST),(10,'random',r.COST),(10,'liquid',.007)]:
            pr,curve,actual=portfolio(e,data,panel,mask,mode,k,slots,rank,cost)
            variants.append(pr)
            if slots==10 and rank=='liquid' and cost==r.COST:
                curve.to_csv(r.OUT/f'{label}_equity.csv',index=False)
                actual.to_csv(r.OUT/f'{label}_portfolio_trades.csv',index=False)
        report['portfolios']=variants;reports[label]=report
        print(label,json.dumps(report),flush=True)
    pd.DataFrame(yearly).to_csv(r.OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    (r.OUT/'review.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
