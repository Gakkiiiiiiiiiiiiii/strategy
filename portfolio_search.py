"""Search finite-capital reversal portfolios using information at signal close."""
import json
import time
from itertools import product
import numpy as np
import pandas as pd
from numba import njit
import research_v2 as r

@njit(cache=True)
def simulate_port(mask,mode,k,gap_cap,slots,rows,offsets,codes,prices,entryprices,gaps,
                  returns,exit_day,calendar,cost,raw_buy,initial_capital=1000000.,lot=100):
    n=len(calendar);cash=1.;equity=1.;peak=1.;dd=0.;live=0
    held=np.full(slots,-1,np.int64);units=np.zeros(slots);exits=np.zeros(slots,np.int64)
    chosen=np.full(slots,-1,np.int64);capital=np.zeros(slots)
    curve=np.ones(n);exposure=np.zeros(n);annual=np.ones(5);annual_dd=np.zeros(5)
    counts=np.zeros(5);sumr=np.zeros(5);wins=np.zeros(5);gains=np.zeros(5);losses=np.zeros(5)
    lastyear=2022;year_start=1.;year_peak=1.;reservations=0;filled=0
    records=np.full((len(rows),4),np.nan);nn=0
    for day in range(n):
        year=calendar[day]//10000
        if year<2022:continue
        if year!=lastyear:
            year_start=equity;year_peak=equity;lastyear=year
        # All candidate orders are selected using signal-close information.
        available=slots-live
        if available>0 and cash>1e-9:
            orders=np.empty(slots,np.int64);order_n=0
            for z in range(offsets[day],offsets[day+1]):
                idx=rows[z];code=codes[idx]
                if not mask[idx]:continue
                repeated=False
                for p in range(slots):
                    if held[p]==code:repeated=True;break
                if repeated:continue
                orders[order_n]=idx;order_n+=1
                if order_n>=available:break
            allocation=min(equity/slots,cash/max(1,order_n))
            reservations+=order_n
            for z in range(order_n):
                idx=orders[z]
                if gaps[idx]>gap_cap or not np.isfinite(entryprices[idx,mode]):continue
                if exit_day[idx,mode,k]<=0 or not np.isfinite(returns[idx,mode,k]):continue
                spend=allocation
                if lot>0:
                    quantity=np.floor(allocation*initial_capital/(raw_buy[idx,mode]*(1+cost/2))/lot)*lot
                    spend=quantity*raw_buy[idx,mode]*(1+cost/2)/initial_capital
                    if spend<=0:continue
                for slot in range(slots):
                    if held[slot]<0:break
                held[slot]=codes[idx];units[slot]=spend/(entryprices[idx,mode]*(1+cost/2))
                exits[slot]=exit_day[idx,mode,k];chosen[slot]=idx;capital[slot]=spend
                cash-=spend;live+=1;filled+=1
        # Prior-close equity sizes orders, no use of future fills or sale cash.
        for slot in range(slots):
            if held[slot]<0:continue
            if exits[slot]==calendar[day]:
                idx=chosen[slot]
                net=(1+returns[idx,mode,k])*(1-cost/2)/(1+cost/2)-1
                cash+=capital[slot]*(1+net)
                y=year-2022;counts[y]+=1;sumr[y]+=net
                if net>0:wins[y]+=1;gains[y]+=net
                else:losses[y]-=net
                records[nn,0]=idx;records[nn,1]=net;records[nn,2]=calendar[day];records[nn,3]=capital[slot];nn+=1
                held[slot]=-1;live-=1
        invested=0.
        for slot in range(slots):
            if held[slot]>=0:invested+=units[slot]*prices[held[slot],day]
        equity=cash+invested;curve[day]=equity;exposure[day]=invested/equity
        peak=max(peak,equity);dd=min(dd,equity/peak-1)
        year_peak=max(year_peak,equity);y=year-2022
        annual[y]=equity/year_start;annual_dd[y]=min(annual_dd[y],equity/year_peak-1)
    return annual-1,annual_dd,counts,sumr,wins,gains,losses,curve,exposure,dd,reservations,filled,records[:nn],live

def prepare():
    e=pd.read_pickle(r.OUT/'events.pkl');d=dict(np.load(r.OUT/'outcomes.npz'));panel=pd.read_pickle(r.OUT/'panel.pkl')
    cal=d['calendar'];codes=list(panel);code_map={c:i for i,c in enumerate(codes)}
    prices=np.zeros((len(codes),len(cal)),dtype=np.float64)
    entryprices=d['buy'].copy()*100
    for code in codes:
        dates,raw,c,factor,pre,mi,*_=panel[code]
        look=np.searchsorted(dates,cal,side='right')-1
        prices[code_map[code]]=c[np.maximum(look,0)]
        idx=np.flatnonzero(e.code.to_numpy()==code)
        js=e.stock_index.to_numpy()[idx]+1
        ok=js<len(factor)
        entryprices[idx[ok]]*=factor[js[ok],None]
    e['code_index']=e.code.map(code_map)
    e['density']=e.signal_date.map(e.groupby('signal_date').size())
    market=pd.read_csv(r.OUT/'market.csv')
    market['prev_breadth']=market.breadth.shift(1);market['breadth_jump']=market.breadth.diff()
    e['breadth_jump']=e.signal_date.map(market.set_index('date').breadth_jump)
    e['prev_breadth']=e.signal_date.map(market.set_index('date').prev_breadth)
    days=e.signal_index.to_numpy()+1
    valid=(e.complete30.eq(1)&(e.planned30//10000).eq(e.entry_date//10000)).to_numpy()
    # Selection can know the planned holding calendar. It cannot know fills.
    ids=np.flatnonzero(valid)
    rows=ids[np.lexsort((e.code.to_numpy()[ids],-e.amount20.to_numpy()[ids],days[ids]))]
    offsets=np.searchsorted(days[rows],np.arange(len(cal)+1))
    common=dict(rows=rows,offsets=offsets,codes=e.code_index.to_numpy(),prices=prices,
                entryprices=entryprices,gaps=e.gap.to_numpy(),returns=d['gross'],exit_day=d['sold'],calendar=cal,raw_buy=d['buy'])
    return e,d,common

def market_rules(e):
    return {
      'market_down':e.market20<0,
      'down_day1':(e.market20<0)&(e.daily>=.01),
      'down_day2':(e.market20<0)&(e.daily>=.02),
      'down_day3':(e.market20<0)&(e.daily>=.03),
      'down_density100':(e.market20<0)&(e.density>=100),
      'down_density200':(e.market20<0)&(e.density>=200),
      'down_density400':(e.market20<0)&(e.density>=400),
      'deep_down_day1':(e.market20<-.05)&(e.daily>=.01),
      'deep_down_day2':(e.market20<-.05)&(e.daily>=.02),
      'oversold_day1':(e.prev_breadth<.20)&(e.daily>=.01),
      'oversold_day2':(e.prev_breadth<.30)&(e.daily>=.02),
      'breadth_turn':(e.prev_breadth<.30)&(e.breadth_jump>=.10),
    }

def search():
    e,d,common=prepare();started=time.time();rows=[];mat=[]
    base=(e.price>=3)&(e.amount20>=3e7)
    fs=r.families(e);extras=r.filters(e)
    for fam,extra,(reg,rm),mode in product(('first_reversal','deep_reversal'),('all','moderate_volume','not_limit'),market_rules(e).items(),range(4)):
        mask=(base&fs[fam]&extras[extra]&rm).to_numpy();gap=.025 if mode==0 else 1.
        if mask.sum()<100:continue
        for k,ex in enumerate(r.EXITS):
            result=simulate_port(mask,mode,k,gap,10,**common,cost=r.COST)
            annual,dd,n,s,w,g,l=result[:7]
            rows.append(dict(family=fam,filter=extra,regime=reg,buy=r.BUYS[mode],gap_max=gap,exit=k))
            mat.append(np.column_stack((annual,dd,n,s,w,g,l)))
    table=np.asarray(mat);np.savez_compressed(r.OUT/'portfolio_search.npz',table=table)
    (r.OUT/'portfolio_configs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'{len(rows)} portfolios in {time.time()-started:.1f}s',flush=True)
    for year in (2024,2025,2026,2027):
        ny=year-2022;annual=table[:,:ny,0];dd=table[:,:ny,1];counts=table[:,:ny,2]
        worst=annual.min(axis=1);avg=np.prod(1+annual,axis=1)**(1/ny)-1
        eligible=(counts.min(axis=1)>=10)&(counts.sum(axis=1)>=60)&(worst>0)&(dd.min(axis=1)>-.25)
        ids=np.flatnonzero(eligible)
        score=worst+.5*avg
        ordered=ids[np.lexsort((-counts[ids].sum(axis=1),-score[ids]))][:20]
        picks=[dict(config=int(i),**rows[i],exit_rule=r.EXITS[rows[i]['exit']],
                    annual=table[i,:,0].tolist(),drawdown=table[i,:,1].tolist(),n=table[i,:,2].tolist()) for i in ordered]
        (r.OUT/f'portfolio_before_{year}.json').write_text(json.dumps(picks,indent=2),encoding='utf-8')
        print(f'Before {year}: {len(ids)} eligible',flush=True)
        for p in picks[:4]:print(json.dumps(p),flush=True)

if __name__=='__main__':search()
