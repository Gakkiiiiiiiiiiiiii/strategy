"""Frozen entries, no 9.5% ceiling, and two-stage signal-gain retracement stops.

No parameter search: compare the original cap, removal of the cap, and the
user-specified exits. Cash from a partial exit becomes available next session.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import research_v2 as r
import research_v3 as v
import portfolio_search as ps
import finalize_v2 as f
from review_v2 import stats

OUT=r.bt.ROOT/'results/research_v4'
CAPITAL=1_000_000.


def tick_below(level):
    """First integer-cent price strictly below an unrounded reference line."""
    return float(math.ceil(level-1e-7)-1)


def staged_exit(raw,factor,pre,market,locked,i,hold):
    """Two equal conceptual tranches; actual share rounding is done at entry.

    Signals are selected at close i. The next-session 98% limit is always above
    the midpoint for eligible >6% signals, so a lower entry-day low crossing a
    stop cannot precede the first buy fill. T+1 breaches are latched, not ignored.
    """
    j=i+1;middle=(pre[i]+raw[i,3])/2*factor[i];bottom=pre[i]*factor[i]
    pending_full=raw[j,2]<=tick_below(bottom/factor[j])
    pending_half=raw[j,2]<=tick_below(middle/factor[j])
    entry_breach=2 if pending_full else 1 if pending_half else 0
    legs=[None,None];blocked=0

    def leg(z,price,reason):
        return dict(day=int(market[z]),price=float(price*factor[z]),raw_price=float(price/100),reason=reason)

    for z in range(j+1,len(raw)):
        age=int(market[z]-market[j]+1);o,h,low,close=raw[z]
        half_tick=tick_below(middle/factor[z]);full_tick=tick_below(bottom/factor[z])
        if locked[z]:
            breached=low<=full_tick or (legs[0] is None and low<=half_tick) or age>=hold
            blocked+=int(breached)
            pending_full|=low<=full_tick
            pending_half|=legs[0] is None and low<=half_tick
            continue
        if pending_full or o<=full_tick:
            reason='full_queued' if pending_full else 'full_gap'
            for p in range(2):
                if legs[p] is None:legs[p]=leg(z,o,reason)
        elif age>hold:
            for p in range(2):
                if legs[p] is None:legs[p]=leg(z,o,'expiry_delayed')
        else:
            if legs[0] is None:
                if pending_half:legs[0]=leg(z,o,'half_queued')
                elif o<=half_tick:legs[0]=leg(z,o,'half_gap')
                elif low<=half_tick:legs[0]=leg(z,half_tick,'half_intraday')
            if low<=full_tick:legs[1]=leg(z,full_tick,'full_intraday')
            if age==hold:
                for p in range(2):
                    if legs[p] is None:legs[p]=leg(z,close,'expiry')
        if legs[1] is not None:
            assert legs[0] is not None and legs[0]['day']<=legs[1]['day']
            return dict(legs=legs,entry_breach=entry_breach,blocked_days=blocked,
                        half_ref=middle/factor[i]/100,full_ref=bottom/factor[i]/100)
    raise ValueError('Unresolved staged exit')


def split_quantity(quantity):
    half=(quantity//200)*100
    return half,quantity-half


def entry_masks(e):
    # Signal return >6%, positive body, wick and prior5/prior10 volume checks
    # are already enforced in the frozen raw event universe. NO upper return cap.
    common=(e.price>=3)&(e.amount20>=3e7)&(e.market20<0)
    return {
        'balanced':np.asarray(common&(e.daily>=.015)&(e.drop_60_250>=.30)&
                              (e.age_60_250<=15)&(e.rise_60_250<=.12)&
                              (e.vol_mean_40>=1)&(e.w_mom_improving==1),bool),
        'macd':np.asarray(common&(e.daily>=.02)&(e.drawdown>=.30)&
                          (e.floor_days<=8)&(e.prior_from_floor<=.08),bool),
    }


def build_plans(e,d,c,panel,mask,hold):
    k=r.EXITS.index((hold,0.,0.,0.));base={};stops={}
    for idx in np.flatnonzero(mask&e.complete30.to_numpy(dtype=bool)):
        if not np.isfinite(d['buy'][idx,2]) or not np.isfinite(d['gross'][idx,2,k]):continue
        x=e.iloc[idx];dates,raw,close,factor,pre,mi,upper,locked=panel[x.code]
        i=int(x.stock_index);j=i+1;entry=float(c['entryprices'][idx,2])
        end_date=int(d['sold'][idx,2,k]);end=np.searchsorted(dates,end_date)
        assert dates[end]==end_date and mi[j]==mi[i]+1
        # Identical cached exit price for an exact regression against v3.
        leg=dict(day=int(mi[end]),price=entry*(1+float(d['gross'][idx,2,k])),
                 raw_price=float(raw[end,3 if mi[end]-mi[j]+1==hold else 0]/100),reason='expiry')
        cap=np.floor(pre[j]*.98+1e-8)
        assert cap>tick_below((pre[i]+raw[i,3])/2*factor[i]/factor[j])
        q=staged_exit(raw,factor,pre,mi,locked,i,hold)
        stops[int(idx)]=q
        base[int(idx)]=dict(legs=[leg.copy(),leg.copy()],entry_breach=q['entry_breach'],blocked_days=0,
                            half_ref=q['half_ref'],full_ref=q['full_ref'])
    return base,stops


def simulate(e,c,ordering,mask,plans,hold,label,cost=r.COST):
    cal=c['calendar'];ids=ordering['rows'];ids=ids[mask[ids]]
    days=e.signal_index.to_numpy()+1;offsets=np.searchsorted(days[ids],np.arange(len(cal)+1))
    cash=equity=peak=1.;lastyear=2022;year_start=year_peak=1.;dd=0.;reserved=filled=0
    positions=[None]*10;curve=np.ones(len(cal));exposure=np.zeros(len(cal))
    annual=np.zeros(5);annual_dd=np.zeros(5);records=[];sells=[];fee=cost/2
    for day,date in enumerate(cal):
        year=int(date)//10000
        if year<2022:continue
        if year!=lastyear:year_start=year_peak=equity;lastyear=year
        held={p['code_index'] for p in positions if p is not None};available=10-len(held)
        orders=[]
        if available and cash>1e-9:
            for idx in ids[offsets[day]:offsets[day+1]]:
                if c['codes'][idx] in held:continue
                orders.append(int(idx))
                if len(orders)>=available:break
            allocation=min(equity/10,cash/max(1,len(orders)));reserved+=len(orders)
            for idx in orders:
                if idx not in plans:continue
                raw_buy=float(c['raw_buy'][idx,2]);entry=float(c['entryprices'][idx,2])
                quantity=int(np.floor(allocation*CAPITAL/(raw_buy*(1+fee))/100)*100)
                if quantity<=0:continue
                spend=quantity*raw_buy*(1+fee)/CAPITAL;half,remaining=split_quantity(quantity)
                units=spend/(entry*(1+fee));x=e.iloc[idx];plan=plans[idx]
                p=dict(event_id=idx,code_index=int(c['codes'][idx]),code=x.code,name=x['name'],
                       signal_date=int(x.signal_date),entry_date=int(x.entry_date),buy_price=raw_buy,
                       entry_adjusted=entry,units=units,remaining_units=units,spend=spend,proceeds=0.,
                       quantity=quantity,leg_quantity=(half,remaining),done=[False,False],plan=plan,
                       action_list=[],signal_return=float(x.signal_return))
                slot=next(j for j,a in enumerate(positions) if a is None)
                positions[slot]=p;cash-=spend;filled+=1
        # Process sales only AFTER order reservation: no future sale proceeds fund today's buys.
        for slot,p in enumerate(positions):
            if p is None:continue
            for j,leg in enumerate(p['plan']['legs']):
                if p['done'][j] or leg['day']!=day:continue
                p['done'][j]=True;q=p['leg_quantity'][j]
                if not q:continue
                fraction=q/p['quantity'];sold_units=p['units']*fraction
                proceeds=sold_units*leg['price']*(1-fee)
                p['remaining_units']-=sold_units;p['proceeds']+=proceeds;cash+=proceeds
                a=dict(event_id=p['event_id'],code=p['code'],signal_date=p['signal_date'],date=int(date),
                       quantity=q,fraction=fraction,raw_price=leg['raw_price'],adjusted_price=leg['price'],
                       reason=leg['reason'],cash=proceeds*CAPITAL)
                previous=p['action_list'][-1] if p['action_list'] else None
                if previous and previous['date']==a['date'] and previous['reason']==a['reason'] and previous['raw_price']==a['raw_price']:
                    for field in ('quantity','fraction','cash'):previous[field]+=a[field]
                else:p['action_list'].append(a);sells.append(a)
            if all(p['done']):
                idx=p['event_id'];x=e.iloc[idx];plan=p['plan']
                mid=float(plan['half_ref']);full=float(plan['full_ref'])
                # Line distances use entry price adjusted to the signal-day basis.
                # Recover signal adjustment from feature panel scale in plans.
                entry_ref=p['buy_price']*plan.get('entry_to_signal_factor',1.)
                net=p['proceeds']/p['spend']-1
                records.append(dict(strategy=label,event_id=idx,code=p['code'],name=p['name'],
                    signal_date=p['signal_date'],entry_date=p['entry_date'],exit_date=int(date),hold=hold,
                    buy_price=p['buy_price'],quantity=p['quantity'],net=net,
                    capital_fraction_of_initial=p['spend'],signal_return=p['signal_return'],
                    stop_half_price=mid,stop_full_price=full,entry_signal_basis=entry_ref,
                    stop_half_distance=mid/entry_ref-1,stop_full_distance=full/entry_ref-1,
                    entry_day_breach=int(plan['entry_breach']),blocked_days=int(plan['blocked_days']),
                    half_skipped_lot=bool(p['leg_quantity'][0]==0),
                    half_sell_date=next((a['date'] for a in p['action_list'] if a['reason'].startswith('half_')),0),
                    full_stop_date=next((a['date'] for a in p['action_list'] if a['reason'].startswith('full_')),0),
                    market_day=float(x.daily),market20=float(x.market20),volume5=float(x.volume5),volume10=float(x.volume10),
                    actions=p['action_list']))
                positions[slot]=None
        invested=sum(p['remaining_units']*c['prices'][p['code_index'],day] for p in positions if p is not None)
        equity=cash+invested;curve[day]=equity;exposure[day]=invested/equity
        peak=max(peak,equity);dd=min(dd,equity/peak-1);year_peak=max(year_peak,equity)
        annual[year-2022]=equity/year_start-1;annual_dd[year-2022]=min(annual_dd[year-2022],equity/year_peak-1)
        assert cash>=-1e-10
    assert not any(p is not None for p in positions)
    t=pd.DataFrame(records);a=pd.DataFrame(sells);dates=cal[cal>=20220101]
    years=(pd.Timestamp(str(dates[-1]))-pd.Timestamp(str(dates[0]))).days/365.25
    summary=dict(total=float(equity-1),cagr=float(equity**(1/years)-1),max_drawdown=float(dd),
                 annual=annual.tolist(),annual_drawdown=annual_dd.tolist(),
                 average_exposure=float(exposure[cal>=20220101].mean()),reservations=reserved,fills=filled,
                 trade_stats=stats(t.net),active_entry_dates=int(t.entry_date.nunique()),
                 half_reductions=int((t.half_sell_date>0).sum()),full_stops=int((t.full_stop_date>0).sum()),
                 entry_day_breaches=int((t.entry_day_breach>0).sum()),blocked_trades=int((t.blocked_days>0).sum()),
                 half_lot_skips=int(t.half_skipped_lot.sum()),
                 annual_trade_stats=[dict(year=int(y),**stats(g.net)) for y,g in t.groupby(t.entry_date//10000)])
    return summary,t,a,pd.DataFrame(dict(date=cal,equity=curve,exposure=exposure)).query('date>=20220101')


def main():
    OUT.mkdir(parents=True,exist_ok=True);e,d,c,orders=v.prepare();panel=pd.read_pickle(r.OUT/'panel.pkl')
    masks=entry_masks(e);report=dict(snapshot_end=r.END,search_count=0,strategies={},
        rule=dict(half='(signal reference previous close + signal close)/2',full='signal reference previous close',
                  trigger='strictly below; first lower cent tick; open gaps at actual open',
                  t_plus_one='entry-day breaches latch and execute on first sellable opening',
                  locked_limit_down='queue until first sellable opening; never assume a fill on locked bar',
                  split='floor(initial shares/200)*100 shares first; remainder second; 100-share position skips half sale',
                  ceiling_removed=True,entry_execution_changed=False,hold_entry_is_day1=True,cost=r.COST,
                  cash_reuse='partial/full proceeds available for new entry orders next session',
                  corporate_actions='signal-basis adjusted lines; same dividend-reinvestment approximation as v3'),
        status='Frozen retrospective ablation, not independent validation; no new parameter optimization.')
    for name,hold,rank in [('balanced',20,'liquidity'),('macd',30,'volume_macd')]:
        uncapped=masks[name];capped=uncapped&(e.signal_return.to_numpy()<.095)
        base,stop=build_plans(e,d,c,panel,uncapped,hold)
        for idx in base:
            x=e.iloc[idx];factor=panel[x.code][3];i=int(x.stock_index)
            base[idx]['entry_to_signal_factor']=stop[idx]['entry_to_signal_factor']=float(factor[i+1]/factor[i])
        variants={};outputs={}
        for key,mask,plans in [('original',capped,base),('uncapped',uncapped,base),
                               ('original_stop',capped,stop),('uncapped_stop',uncapped,stop)]:
            s,t,a,curve=simulate(e,c,orders[rank],mask,plans,hold,name+'_'+key)
            variants[key]=s;outputs[key]=t
            t.assign(actions=t.actions.map(lambda x:json.dumps(x,ensure_ascii=False))).to_csv(OUT/f'{name}_{key}_trades.csv',index=False,encoding='utf-8-sig')
            a.to_csv(OUT/f'{name}_{key}_sales.csv',index=False,encoding='utf-8-sig')
            curve.to_csv(OUT/f'{name}_{key}_equity.csv',index=False)
            print(name,key,json.dumps({k:s[k] for k in ('cagr','max_drawdown','fills','half_reductions','full_stops','trade_stats')},ensure_ascii=False),flush=True)
        # Exact regression against original finite-cash implementation and cached v3 trades.
        k=r.EXITS.index((hold,0.,0.,0.));old=ps.simulate_port(capped,2,k,1.,10,**{**c,**orders[rank]},cost=r.COST)
        old_s=f.summarize(old,c['calendar']);reproduced=variants['original']
        assert reproduced['fills']==old_s['fills'] and abs(reproduced['cagr']-old_s['cagr'])<1e-9
        assert set(outputs['original'].event_id)==set(old[12][:,0].astype(int))
        prior=pd.read_csv(v.OUT/('selected_trades.csv' if name=='balanced' else 'macd_trades.csv'))
        assert set(prior.event_id)==set(outputs['original'].event_id)
        # Same uncapped entry transactions, change only the exits (no reinvestment).
        matched=[]
        for x in outputs['uncapped'].itertuples():
            q1,q2=split_quantity(x.quantity);legs=stop[x.event_id]['legs']
            gross=(q1*legs[0]['price']+q2*legs[1]['price'])/x.quantity/c['entryprices'][x.event_id,2]-1
            net=(1+gross)*(1-r.COST/2)/(1+r.COST/2)-1
            matched.append(dict(event_id=x.event_id,code=x.code,signal_date=x.signal_date,entry_date=x.entry_date,
                                original_net=x.net,staged_net=net,delta=net-x.net))
        matched=pd.DataFrame(matched);matched.to_csv(OUT/f'{name}_matched.csv',index=False,encoding='utf-8-sig')
        report['strategies'][name]=dict(hold=hold,rank=rank,variants=variants,
            fixed_entries=dict(before=stats(matched.original_net),after=stats(matched.staged_net),
                               mean_delta=float(matched.delta.mean()),improved=int((matched.delta>1e-6).sum()),
                               worsened=int((matched.delta< -1e-6).sum()),unchanged=int((matched.delta.abs()<=1e-6).sum())),
            regression=dict(trades=int(old_s['fills']),max_cagr_error=abs(reproduced['cagr']-old_s['cagr']),same_event_ids=True))
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
