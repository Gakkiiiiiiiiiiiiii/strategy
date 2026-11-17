"""Reproducible selected model, parameter perturbations, trade audit and report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import research_v2 as r
import portfolio_search as ps
from review_v2 import stats,clustered_interval,trades_from

PARAMETERS=dict(family='deep_first_reversal',prior_floor_window=60,prior_peak_window=120,
    pit_drop_min=.30,floor_age_max=8,prior_close_over_floor_max=1.08,
    signal_return_min_exclusive=.06,signal_return_max_exclusive=.095,
    positive_body=True,upper_wick_body_max=1/3,volume_gt_prior5=True,volume_gt_prior10=True,
    signal_price_min=3,mean_amount20_min=30000000,market_equal_weight_day_min=.02,
    market_equal_weight_compounded20_max_exclusive=0,next_day_limit_discount=.02,
    order_expiry='next trading day',skip_upper_limit_open=True,hold_market_sessions=20,
    entry_is_session1=True,take_profit=None,stop_loss=None,max_positions=10,
    position_equity_fraction_max=.10,rank='descending signal-day known mean_amount20, then code',
    round_trip_cost=.0035,initial_capital=1000000,order_lot=100,
    same_day_unfilled_cash_reallocated=False,common_maturity_window=30)

def summarize(result,cal):
    annual,annual_dd,n,s,w,g,l,curve,exposure,dd,reserved,filled,records,live=result
    assert live==0
    dates=cal[cal>=20220101];equity=curve[cal>=20220101]
    years=(pd.Timestamp(str(dates[-1]))-pd.Timestamp(str(dates[0]))).days/365.25
    return dict(total=float(equity[-1]-1),cagr=float(equity[-1]**(1/years)-1),
                max_drawdown=float(dd),average_exposure=float(exposure[cal>=20220101].mean()),
                annual=annual.tolist(),annual_drawdown=annual_dd.tolist(),n=n.astype(int).tolist(),
                reservations=int(reserved),fills=int(filled),trade_stats=stats(records[:,1]))

def chosen_mask(e,pit=.30,age=8,rise=.08,move=.095,market=.02,**kwargs):
    return ((e.price>=3)&(e.amount20>=3e7)&(e.drawdown>=pit)&(e.floor_days<=age)&
            (e.prior_from_floor<=rise)&(e.signal_return<move)&(e.market20<0)&(e.daily>=market)).to_numpy()

def main():
    e,d,common=ps.prepare();cal=common['calendar'];mode=2;k=33
    purged_common=common.copy()
    # Continuous final portfolio permits cross-year positions. The selection
    # runs retain purged annual boundaries. This does not change primary fills.
    days=e.signal_index.to_numpy()+1
    ids0=np.flatnonzero(e.complete30.eq(1))
    rows=ids0[np.lexsort((e.code.to_numpy()[ids0],-e.amount20.to_numpy()[ids0],days[ids0]))]
    common={**common,'rows':rows,'offsets':np.searchsorted(days[rows],np.arange(len(cal)+1))}
    mask=chosen_mask(e)
    result=ps.simulate_port(mask,mode,k,1.,10,**common,cost=r.COST)
    headline=summarize(result,cal)
    records=result[12];ids=records[:,0].astype(int)
    trades=e.iloc[ids].copy();trades['event_id']=ids;trades['net']=records[:,1];trades['exit_date']=records[:,2].astype(int)
    trades['capital_fraction_of_initial']=records[:,3];trades['buy_price']=d['buy'][ids,mode]
    trades['gross']=d['gross'][ids,mode,k].astype(float)
    # Independent raw-bar reconstruction for every actual executed fill.
    panel=pd.read_pickle(r.OUT/'panel.pkl');errors=[];maturities=[]
    for x in trades.itertuples():
        dates,raw,adj,factor,pre,mi,upper,locked=panel[x.code]
        j=int(np.searchsorted(dates,x.entry_date));z=int(np.searchsorted(dates,x.exit_date))
        cap=np.floor(pre[j]*.98+1e-8)
        assert raw[j,0]<upper[j]
        if raw[j,0]<=cap:buy=raw[j,0]
        else:
            assert raw[j,2]<cap
            buy=cap
        assert buy/100==x.buy_price and not locked[z]
        offset=mi[z]-mi[j]+1
        assert offset>=20
        sell=raw[z,3] if offset==20 else raw[z,0]
        expected=sell*factor[z]/(buy*factor[j])-1
        errors.append(abs(expected-x.gross));maturities.append(offset)
    assert max(errors)<2e-7
    headline['audit']=dict(trades=len(trades),max_return_error=max(errors),
                           deferred_exits=int(np.sum(np.asarray(maturities)>20)),
                           fill_method='Next-day limit close*0.98; strict price cross for intraday fill; no upper-limit open fills',
                           capital=1000000,board_lot=100)
    annual_stats=[]
    for year,g in trades.groupby(trades.entry_date//10000):
        annual_stats.append(dict(year=int(year),**stats(g.net),unique_dates=int(g.entry_date.nunique())))
    headline['annual_trade_stats']=annual_stats
    headline['active_entry_dates']=int(trades.entry_date.nunique())
    headline['monthly_bootstrap']=clustered_interval(trades)
    headline['without_top5']=stats(trades.net.sort_values().iloc[:-5])
    headline['without_top5_2026']=stats(trades.loc[trades.entry_date>=20260101].net.sort_values().iloc[:-5])
    # Distinguish fill-quality signal sample from the finite cash portfolio.
    signal_trades=trades_from(e,d,pd.Series(mask),mode,k)
    headline['all_signals']=stats(signal_trades.net)
    headline['all_signal_dates']=int(signal_trades.entry_date.nunique())
    signal_trades.to_csv(r.OUT/'final_all_signal_trades.csv',index=False,encoding='utf-8-sig')
    trades.to_csv(r.OUT/'final_portfolio_trades.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(date=cal,equity=result[7],exposure=result[8])).query('date>=20220101').to_csv(r.OUT/'final_equity.csv',index=False)
    pd.DataFrame(annual_stats).to_csv(r.OUT/'final_yearly.csv',index=False,encoding='utf-8-sig')
    variations=[]
    for field,values in [('pit',(.25,.35)),('age',(6,10)),('rise',(.06,.10)),('move',(.09,.10)),('market',(.015,.025,.03))]:
        for value in values:
            changed=chosen_mask(e,**{field:value})
            rr=ps.simulate_port(changed,mode,k,1.,10,**common,cost=r.COST)
            variations.append(dict(change=field,value=value,**summarize(rr,cal)))
    for slots in (5,20):
        rr=ps.simulate_port(mask,mode,k,1.,slots,**common,cost=r.COST)
        variations.append(dict(change='slots',value=slots,**summarize(rr,cal)))
    for cost in (.007,.01):
        rr=ps.simulate_port(mask,mode,k,1.,10,**common,cost=cost)
        variations.append(dict(change='cost',value=cost,**summarize(rr,cal)))
    # Alternative pre-signal rankings are kept separate, never outcome-ranked.
    ids0=common['rows'];days=e.signal_index.to_numpy()+1
    rng=np.random.default_rng(45179)
    for rank in ('code','random','low_liquidity'):
        tie=e.code.to_numpy()[ids0] if rank=='code' else rng.random(len(ids0)) if rank=='random' else e.amount20.to_numpy()[ids0]
        rows=ids0[np.lexsort((tie,days[ids0]))];cc={**common,'rows':rows}
        rr=ps.simulate_port(mask,mode,k,1.,10,**cc,cost=r.COST)
        variations.append(dict(change='rank',value=rank,**summarize(rr,cal)))
    # Entry and holding changes require re-simulation, not return rescaling.
    relevant=np.flatnonzero(mask)
    for typ,value in [('discount',.01),('discount',.03),('hold',15),('hold',25)]:
        cc={key:val for key,val in common.items()}
        rr=common['returns'].copy();ss=common['exit_day'].copy();bb=common['raw_buy'].copy();aa=common['entryprices'].copy()
        discount=value if typ=='discount' else .02;hold=value if typ=='hold' else 20
        for idx in relevant:
            x=e.iloc[idx];dates,raw,c,factor,pre,mi,upper,locked=panel[x.code]
            ret,sold,buy=r.outcomes(raw,factor,pre,mi,int(x.stock_index),upper,locked,np.asarray([(hold,0.,0.,0.)]),(0.,discount,.04))
            rr[idx,mode,k]=ret[mode,0];ss[idx,mode,k]=int(dates[sold[mode,0]]) if sold[mode,0]>0 else 0
            bb[idx,mode]=buy[mode]/100
            j=int(x.stock_index)+1
            aa[idx,mode]=buy[mode]*factor[j] if j<len(factor) else np.nan
        cc.update(returns=rr,exit_day=ss,raw_buy=bb,entryprices=aa)
        result2=ps.simulate_port(mask,mode,k,1.,10,**cc,cost=r.COST)
        variations.append(dict(change=typ,value=value,**summarize(result2,cal)))
    # Original requested horizons under the same selected entry, cash and cost.
    horizons=[]
    for h in (5,10,20,30):
        kk=r.EXITS.index((h,0.,0.,0.));z=ps.simulate_port(mask,mode,kk,1.,10,**common,cost=r.COST)
        horizons.append(dict(hold=h,**summarize(z,cal)))
    # Preserve the initial chronological maximum-return selector's performance,
    # including its weaker 2026. The final hindsight fit is not substituted.
    chronological=[]
    configs=json.loads((r.OUT/'portfolio_configs.json').read_text())
    for year in (2024,2025,2026):
        pick=json.loads((r.OUT/f'portfolio_before_{year}.json').read_text())[0]
        mm=((e.price>=3)&(e.amount20>=3e7)&r.families(e)[pick['family']]&r.filters(e)[pick['filter']]&ps.market_rules(e)[pick['regime']]).to_numpy()
        zz=ps.simulate_port(mm,r.BUYS.index(pick['buy']),pick['exit'],pick['gap_max'],10,**purged_common,cost=r.COST)
        chronological.append(dict(year=year,selection=pick,actual_return=float(zz[0][year-2022]),
                                   drawdown=float(zz[1][year-2022]),trades=int(zz[2][year-2022])))
    # Opening-only version is reported as a variant, keeping the entry change explicit.
    zz=ps.simulate_port(mask,0,k,.025,10,**common,cost=r.COST)
    headline['next_open_variant']=summarize(zz,cal)
    out=dict(parameters=PARAMETERS,selected=headline,sensitivity=variations,horizons=horizons,chronological=chronological,
             selection_status='Exploratory full-sample fit, including 2026. The same history was repeatedly inspected. No pristine OOS or future guarantee.',
             snapshot_end=r.END,search_counts=dict(event_hypotheses=138897,finite_portfolios=17568),
             boundaries='Common 30-day maturity. Annual boundaries purged for selection, continuous final portfolio permits cross-year holdings. Frozen current TDX universe; delisted universe not reconstructed.')
    (r.OUT/'final_report.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
