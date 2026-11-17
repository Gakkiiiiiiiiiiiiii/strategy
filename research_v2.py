"""Causal price/volume research, with dated selection and executable exits.

Run build once, then search. Input is the frozen 2026-11-16 TDX/BaoStock
snapshot. This dataset has already been inspected in earlier research, so
chronological evaluation is a retrospective check, not a pristine holdout.
"""
from contextlib import ExitStack
from pathlib import Path
from itertools import product
import argparse
import json
import time
import numpy as np
import pandas as pd
from numba import njit
import backtest as bt

OUT = bt.ROOT / 'results' / 'research_v2'
END = 20261116
COST = .0035
BUYS = ('next_open', 'limit_close', 'limit_minus2', 'limit_minus4')
EXITS = [(h,t,s,0.) for h in (3,5,10,20,30)
         for t,s in ((0.,0.),(.04,.04),(.08,.04),(.12,.04),(.18,.04),
                     (.08,.07),(.12,.07),(.18,.07),(0.,.07),(.12,.10),(.18,.10))]
EXITS += [(h,0.,.07,trail) for h in (10,20,30) for trail in (.08,.12)]

def rolling(a,n,operation='mean'):
    return getattr(pd.Series(a).rolling(n),operation)().to_numpy()

@njit(cache=True)
def outcomes(raw, factor, preclose, market, signal, upper, locked, exits, discounts=(0.,.02,.04)):
    # A buy limit is submitted after the signal and expires after the next day.
    # One-price upper-limit bars never create an assumed fill. A touch at the
    # low alone is not a fill: a limit must be crossed by at least one cent.
    n = len(exits)
    gross = np.full((4,n),np.nan)
    sold = np.zeros((4,n),np.int32)
    prices = np.full(4,np.nan)
    if signal+1>=len(raw) or market[signal+1]!=market[signal]+1:
        return gross,sold,prices
    j=signal+1
    if raw[j,0]>=upper[j] or (raw[j,1]==raw[j,2] and raw[j,1]>=upper[j]):
        return gross,sold,prices
    for mode in range(4):
        if mode==0:
            buy=raw[j,0]
        else:
            discount=discounts[mode-1]
            cap=np.floor(preclose[j]*(1-discount)+1e-8)
            if raw[j,0]<=cap:
                buy=raw[j,0]
            elif raw[j,2]<cap:
                buy=cap
            else:
                continue
        prices[mode]=buy
        entry=buy*factor[j]
        for k in range(n):
            hold,take,stop,trail=exits[k]
            # Entry-day high may have preceded an intraday buy. Never use it.
            high_water=max(entry,raw[j,3]*factor[j])
            for z in range(j+1,len(raw)):
                off=market[z]-market[j]+1
                if off>hold+15:
                    break
                if locked[z]:
                    high_water=max(high_water,raw[z,3]*factor[z])
                    continue
                take_raw=np.ceil(entry*(1+take)/factor[z]-1e-9)
                stop_price=entry*(1-stop) if stop else 0.
                if trail:
                    stop_price=max(stop_price,high_water*(1-trail))
                stop_raw=np.floor(stop_price/factor[z]+1e-9)
                o,h,low,c=raw[z]
                if off>hold:
                    sell=o
                elif stop_price and o<=stop_raw:
                    sell=o
                elif take and o>=take_raw:
                    sell=take_raw
                elif stop_price and low<=stop_raw:
                    sell=stop_raw
                elif take and h>take_raw:
                    sell=take_raw
                elif off==hold:
                    sell=c
                else:
                    high_water=max(high_water,c*factor[z])
                    continue
                gross[mode,k]=sell*factor[z]/entry-1
                sold[mode,k]=z
                break
    return gross,sold,prices

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    actions=bt.load_actions()
    st={(r.code,int(r.date)):int(r.isST) for r in pd.read_csv(bt.ROOT/'data/st_checks.csv').itertuples()}
    names=bt.current_names(Path('D:/tdx'))
    event_rows=[]; all_r=[]; all_sell=[]; all_buy=[]; markets=[]
    panel={}; started=time.time()
    with ExitStack() as stack:
        files,read=bt.market_files(Path('D:/tdx/vipdoc'),stack)
        cal=bt.end_of_market(np.frombuffer(read('sh000001'),bt.BAR_DTYPE),END)
        base_end=int(cal[-1]); extra_dates,additions=bt.updated_bars(base_end,END,st)
        cal=np.r_[cal,extra_dates]
        counts=np.zeros(len(cal)); breadth=np.zeros(len(cal)); market_ret=np.zeros(len(cal))
        for number,stem in enumerate(sorted(files)):
            if not bt.STOCK.fullmatch(stem): continue
            code=stem[:2]+'.'+stem[2:]
            bars=np.frombuffer(read(stem),bt.BAR_DTYPE)
            bars=bars[(bars['date']<=base_end)&(bars['volume']>0)&(bars['low']>0)]
            extra,previous=additions.get(code,(None,{}))
            if extra is not None: bars=np.r_[bars,extra]
            bars=bars[np.isin(bars['date'],cal)]
            if len(bars)<120 or np.any(np.diff(bars['date'])<=0): continue
            pre,c,factor=bt.adjusted_prices(bars,actions.get(code,()),previous)
            dates=bars['date']; mi=np.searchsorted(cal,dates)
            raw=np.column_stack([bars[k] for k in ('open','high','low','close')]).astype(float)
            vol=bars['volume'].astype(float); ret=bars['close']/pre-1
            ma10=rolling(c,10); ma20=rolling(c,20); ma60=rolling(c,60)
            amount20=rolling(bars['amount'].astype(float),20)
            good=np.isfinite(ma20)&(np.arange(len(c))>=60)
            counts[mi[good]]+=1; breadth[mi[good]]+=(c[good]>ma20[good]); market_ret[mi[good]]+=ret[good]
            growth=code[3:6] in {'300','301','688'}
            rate=20 if growth else 10
            upper=np.maximum((pre*(100+rate)+50)//100,pre+1).astype(float)
            down=np.minimum((pre*(100-rate)+50)//100,pre-1)
            locked=(bars['high']==bars['low'])&(bars['low']<=down)
            if not growth:
                five=np.maximum((pre*105+50)//100,pre+1)
                down5=np.minimum((pre*95+50)//100,pre-1)
                for j in np.flatnonzero(dates<bt.ST_LIMIT_CHANGE):
                    status=st.get((code,int(dates[j])))
                    if status==1 or (status is None and raw[j,0]==five[j]): upper[j]=five[j]
                    if raw[j,1]==raw[j,2] and raw[j,2]==down5[j] and status!=0: locked[j]=True
            panel[code]=(dates,raw,c,factor,pre,mi,upper,locked)
            v5=rolling(vol,5); v10=rolling(vol,10)
            body=raw[:,3]-raw[:,0]; wick=raw[:,1]-raw[:,3]
            candidates=np.flatnonzero((dates>=20220101)&(ret>.06)&(body>0)&(wick<=body/3+1e-8)
                                     &(vol>np.r_[np.nan,v5[:-1]])&(vol>np.r_[np.nan,v10[:-1]]))
            original={i:metrics for i,metrics in bt.detect(bars,pre,c,bt.Rules(),20220101,deduplicate=False)}
            for i in candidates:
                if i<120: continue
                floor=i-60+int(np.argmin(c[i-60:i]))
                peak=i-120+int(np.argmax(c[i-120:floor+1]))
                metrics=original.get(int(i),{})
                true_peak=False
                if metrics:
                    f=int(np.searchsorted(dates,metrics['floor_date']))
                    p=int(np.searchsorted(dates,metrics['rally_peak_date']))
                    true_peak=c[f+1:i].max()<=c[p]
                prev_high20=c[i-20:i].max(); prev_high60=c[i-60:i].max()
                atr=np.maximum(raw[1:i+1,1]*factor[1:i+1]-raw[1:i+1,2]*factor[1:i+1],
                               np.maximum(abs(raw[1:i+1,1]*factor[1:i+1]-c[:i]),abs(raw[1:i+1,2]*factor[1:i+1]-c[:i])))
                row=dict(code=code,name=names.get(code,''),signal_date=int(dates[i]),signal_index=int(mi[i]),
                         stock_index=int(i),board=int(growth),signal_return=ret[i],price=raw[i,3]/100,
                         volume5=vol[i]/v5[i-1],volume10=vol[i]/v10[i-1],amount20=amount20[i],
                         mom5=c[i-1]/c[i-6]-1,mom20=c[i-1]/c[i-21]-1,
                         mom60=c[i-1]/c[i-61]-1,ma20=c[i]/ma20[i]-1,ma60=c[i]/ma60[i]-1,
                         ma20_slope=ma20[i]/ma20[i-5]-1,trend=ma20[i]/ma60[i]-1,
                         break20=c[i]/prev_high20-1,break60=c[i]/prev_high60-1,
                         drawdown=1-c[floor]/c[peak],floor_days=i-floor,
                         from_floor=c[i]/c[floor]-1,prior_from_floor=c[i-1]/c[floor]-1,
                         true_peak=int(true_peak),original=int(bool(metrics)),
                         gap=raw[i+1,0]/pre[i+1]-1 if i+1<len(c) else np.nan,
                         atr=atr[-14:].mean()/c[i],wick=wick[i]/body[i],
                         rally=metrics.get('rally_return',np.nan),wash=metrics.get('wash_drop',np.nan),
                         dry=metrics.get('wash_volume_ratio',np.nan),pit=metrics.get('pit_drawdown',np.nan),
                         recorded_peak=metrics.get('rally_peak_date',0),
                         entry_date=int(cal[mi[i]+1]) if mi[i]+1<len(cal) else 0)
                r,s,b=outcomes(raw,factor,pre,mi,int(i),upper,locked,np.asarray(EXITS))
                sell_dates=np.zeros_like(s); found=s>0; sell_dates[found]=dates[s[found]]
                # Common 30-session endpoint removes preferential observation
                # of early winners near the end of the dataset.
                complete=mi[i]+30<len(cal)
                row['complete30']=int(complete)
                row['planned30']=int(cal[mi[i]+30]) if complete else 0
                event_rows.append(row); all_r.append(r); all_sell.append(sell_dates); all_buy.append(b/100)
            if number%1000==0:
                print(f'{number} files; {len(event_rows):,} events; {time.time()-started:.1f}s',flush=True)
    market=pd.DataFrame(dict(date=cal,breadth=np.divide(breadth,counts,out=np.zeros_like(counts),where=counts>0),
                             daily=np.divide(market_ret,counts,out=np.zeros_like(counts),where=counts>0)))
    market['market20']=(1+market.daily).rolling(20).apply(np.prod,raw=True)-1
    market['market5']=(1+market.daily).rolling(5).apply(np.prod,raw=True)-1
    events=pd.DataFrame(event_rows).merge(market,left_on='signal_date',right_on='date',how='left').drop(columns='date')
    events.to_pickle(OUT/'events.pkl')
    pd.to_pickle(panel,OUT/'panel.pkl')
    np.savez_compressed(OUT/'outcomes.npz',gross=np.asarray(all_r,dtype=np.float32),sold=np.asarray(all_sell),buy=np.asarray(all_buy),calendar=cal)
    market.to_csv(OUT/'market.csv',index=False)
    (OUT/'spec.json').write_text(json.dumps(dict(end=END,cost=COST,buys=BUYS,exits=EXITS,events=len(events),
                                                notes='2026 already used in prior research; no untouched test claimed'),indent=2),encoding='utf-8')
    print(f'Built {len(events):,} events in {time.time()-started:.1f}s',flush=True)

def families(e):
    # Deliberately broad economic hypotheses, all known after the signal close.
    return {
      'second_wave': e.original.eq(1),
      'true_second': e.original.eq(1)&e.true_peak.eq(1),
      'first_reversal': (e.drawdown>=.20)&(e.prior_from_floor<=.10)&(e.floor_days<=12),
      'deep_reversal': (e.drawdown>=.30)&(e.prior_from_floor<=.08)&(e.floor_days<=8),
      'trend_breakout': (e.trend>0)&(e.break20>=0)&(e.ma60>0),
      'high_breakout': (e.break60>=0)&(e.ma20_slope>0),
      'trend_pullback': (e.trend>0)&(e.mom5<0)&(e.mom20>0)&(e.ma60>0),
      'recovery': (e.drawdown>=.20)&(e.ma20>0)&(e.ma20_slope>0)&(e.prior_from_floor<=.20),
    }

def filters(e):
    return {
      'all':np.ones(len(e),bool),
      'moderate_volume':(e.volume5<=2.5)&(e.volume10<=3),
      'high_volume':(e.volume5>=2)&(e.volume10>=2),
      'not_limit':e.signal_return<.095,
      'strong_close':e.signal_return>.09,
      'low_volatility':e.atr<.04,
      'liquid':e.amount20>=2e8,
      'relative_strength':e.mom20>e.market20,
    }

def regimes(e):
    return {
      'all':np.ones(len(e),bool),
      'breadth_above50':e.breadth>=.50,
      'breadth_30to70':e.breadth.between(.30,.70),
      'market_up':e.market20>0,
      'market_down':e.market20<0,
      'not_hot':e.market5<.03,
    }

@njit(cache=True)
def aggregate(mask,net,years,valid,codes,signal_dates,sold):
    # Independent event sample, with same-stock overlaps excluded per rule.
    out=np.zeros((net.shape[1],5,7))
    for k in range(net.shape[1]):
        last_exit=np.zeros(codes.max()+1,np.int64)
        for i in range(len(mask)):
            y=years[i]-2022
            if not mask[i] or y<0 or y>=5 or not valid[i,k] or signal_dates[i]<=last_exit[codes[i]]: continue
            last_exit[codes[i]]=sold[i,k]
            v=net[i,k]
            out[k,y,0]+=1; out[k,y,1]+=v; out[k,y,2]+=v*v
            if v>0:
                out[k,y,3]+=1;out[k,y,4]+=v
            elif v<0:
                out[k,y,5]+=1;out[k,y,6]-=v
    return out

def search():
    e=pd.read_pickle(OUT/'events.pkl'); data=np.load(OUT/'outcomes.npz')
    # Deterministic sorting preserves original row IDs for the event cache.
    ix=np.lexsort((e.code,e.signal_date)); e=e.iloc[ix].reset_index().rename(columns={'index':'event_id'})
    gross=data['gross'][ix]; sold=data['sold'][ix]
    net=(1+gross)*(1-COST/2)/(1+COST/2)-1
    years=e.entry_date.to_numpy()//10000
    complete=e.complete30.eq(1).to_numpy()&(e.planned30.to_numpy()//10000==years)
    valid=np.isfinite(net)&(sold//10000==years[:,None,None])&complete[:,None,None]
    codes=pd.factorize(e.code)[0]; dates=e.signal_date.to_numpy(np.int64)
    base=(e.price>=3)&(e.amount20>=3e7)
    rows=[]; tables=[]; number=0
    for (fam,fm),(reg,rm),(extra,xm),gap,mode in product(families(e).items(),regimes(e).items(),filters(e).items(),(.00,.025,1.),range(4)):
        # For limit orders the limit itself controls purchase price, so using
        # the auction opening as an additional condition is unnecessary.
        if mode>0 and gap!=1.: continue
        mask=(base&fm&rm&xm&(e.gap<=gap)).to_numpy()
        if mask.sum()<150: continue
        table=aggregate(mask,net[:,mode,:],years,valid[:,mode,:],codes,dates,sold[:,mode,:])
        config=dict(family=fam,regime=reg,filter=extra,gap_max=gap,buy=BUYS[mode])
        rows.append(config);tables.append(table);number+=1
    table=np.asarray(tables)
    np.savez_compressed(OUT/'search.npz',table=table)
    (OUT/'configs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'{number} entry hypotheses x {len(EXITS)} exits = {number*len(EXITS):,}',flush=True)
    n=table[...,0]; sums=table[...,1]
    means=np.divide(sums,n,out=np.full_like(sums,np.nan),where=n>0)
    for cutoff in (2024,2025,2026,2027):
        ny=cutoff-2022
        counts=n[:,:,:ny]; avg=means[:,:,:ny]
        # Worst prior year, then statistical margin. No future years enter
        # selection for any of the chronological cutoffs.
        eligible=(counts.min(axis=-1)>=25)&(counts.sum(axis=-1)>=180)&(avg.min(axis=-1)>.003)
        a,b=np.where(eligible)
        print(f'cutoff {cutoff}: {len(a):,} eligible',flush=True)
        if not len(a): continue
        overall=sums[:,:,:ny].sum(axis=-1)/counts.sum(axis=-1)
        worst=avg.min(axis=-1)
        scores=worst+.25*overall
        order=np.lexsort((-counts[a,b].sum(axis=-1),-scores[a,b]))[:20]
        picks=[]
        for z in order:
            i,j=int(a[z]),int(b[z]);stats=table[i,j]
            win=np.divide(stats[:,3],stats[:,0],out=np.zeros(5),where=stats[:,0]>0)
            ratio=(stats[:,4]/np.maximum(stats[:,3],1))/(stats[:,6]/np.maximum(stats[:,5],1))
            picks.append(dict(config=i,exit=j,**rows[i],exit_rule=EXITS[j],n=n[i,j].tolist(),
                              mean=means[i,j].tolist(),win=win.tolist(),payoff=ratio.tolist()))
        (OUT/f'picks_before_{cutoff}.json').write_text(json.dumps(picks,indent=2),encoding='utf-8')
        for p in picks[:5]: print(json.dumps(p),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['build','search']);args=p.parse_args()
    globals()[args.task]()
