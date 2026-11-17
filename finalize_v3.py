"""Audit multi-timeframe models and preserve negative ablation evidence."""
import json
import numpy as np
import pandas as pd
import research_v2 as r
import research_v3 as v
import portfolio_search as ps
import finalize_v2 as f
from review_v2 import stats,clustered_interval,trades_from

def main():
    e,d,c,order=v.prepare();base=v.base_rules(e);gates=v.gates(e)
    configs=json.loads((v.OUT/'configs.json').read_text());table=np.load(v.OUT/'search.npz')['table']
    annual=table[:,:,0];dd=table[:,:,1];n=table[:,:,2]
    approx=np.prod(1+annual,axis=1)**(1/4.865)-1
    selected={}
    for label,exit_id,minn,cap in [('balanced',33,90,.10),('return_focused',44,80,.12)]:
        ok=(n.min(1)>=8)&(n.sum(1)>=minn)&(annual.min(1)>0)&(dd.min(1)>-cap)&np.array([q['exit']==exit_id for q in configs])
        ids=np.flatnonzero(ok);ids=ids[np.argsort(-approx[ids])]
        for idx in ids:
            cfg=configs[idx];mask=np.asarray(base[cfg['base']]&gates[cfg['gate']],bool)
            z=ps.simulate_port(mask,2,cfg['exit'],1.,10,**{**c,**order[cfg['rank']]},cost=r.COST)
            s=f.summarize(z,c['calendar'])
            if s['max_drawdown']<=-cap:continue
            selected[label]=(cfg,mask,z,s);break
    comparisons=[]
    baseline_cfg=dict(base='original_market0.02',gate='none',rank='liquidity',exit=33)
    oldmask=np.asarray(base[baseline_cfg['base']],bool)
    old=ps.simulate_port(oldmask,2,33,1.,10,**{**c,**order['liquidity']},cost=r.COST)
    comparisons.append(dict(label='上一版固定参数',config=baseline_cfg,**f.summarize(old,c['calendar'])))
    balanced_cfg,mask,z,s=selected['balanced'];rec=z[12];ids=rec[:,0].astype(int)
    trades=e.iloc[ids].copy();trades['event_id']=ids;trades['net']=rec[:,1];trades['exit_date']=rec[:,2].astype(int)
    trades['capital_fraction_of_initial']=rec[:,3];trades['buy_price']=d['buy'][ids,2];trades['gross']=d['gross'][ids,2,33].astype(float)
    trades.to_csv(v.OUT/'selected_trades.csv',index=False,encoding='utf-8-sig')
    alltrades=trades_from(e,d,pd.Series(mask),2,33)
    alltrades.to_csv(v.OUT/'all_signal_trades.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(date=c['calendar'],equity=z[7],exposure=z[8])).query('date>=20220101').to_csv(v.OUT/'equity.csv',index=False)
    s['active_entry_dates']=int(trades.entry_date.nunique());s['all_signals']=stats(alltrades.net)
    s['annual_trade_stats']=[dict(year=int(y),**stats(g.net)) for y,g in trades.groupby(trades.entry_date//10000)]
    s['monthly_bootstrap']=clustered_interval(trades);s['without_top5']=stats(trades.net.sort_values().iloc[:-5])
    s['without_top5_2026']=stats(trades.loc[trades.entry_date>=20260101].net.sort_values().iloc[:-5])
    for label,(cfg,m,zz,ss) in selected.items():
        rr=zz[12];ii=rr[:,0].astype(int);tt=e.iloc[ii].copy();tt['event_id']=ii;tt['net']=rr[:,1]
        tt['exit_date']=rr[:,2].astype(int);tt['capital_fraction_of_initial']=rr[:,3]
        tt['buy_price']=d['buy'][ii,2];tt['gross']=d['gross'][ii,2,cfg['exit']].astype(float)
        ss['active_entry_dates']=int(tt.entry_date.nunique())
        ss['annual_trade_stats']=[dict(year=int(y),**stats(g.net)) for y,g in tt.groupby(tt.entry_date//10000)]
        if label=='return_focused':
            tt.to_csv(v.OUT/'macd_trades.csv',index=False,encoding='utf-8-sig')
            a=trades_from(e,d,pd.Series(m),2,cfg['exit']);a.to_csv(v.OUT/'macd_all_signals.csv',index=False,encoding='utf-8-sig')
            ss['all_signals']=stats(a.net)
            pd.DataFrame(dict(date=c['calendar'],equity=zz[7],exposure=zz[8])).query('date>=20220101').to_csv(v.OUT/'macd_equity.csv',index=False)
        comparisons.append(dict(label='量能与周线平衡版' if label=='balanced' else '量能与MACD排序版',config=cfg,**ss))
    # Same base/entry/exit, one additional indicator at a time.
    ablations=[]
    for name in ('none','bullavg40_ge1.0','bullavg60_ge1.2','bullavg120_ge1.0','down_volume_dry',
                 'divergence','near_divergence','macd_hist_turn','weekly_momentum','monthly_momentum','monthly_hist_up','volume_weekly'):
        mm=np.asarray(base['original_market0.02']&gates[name],bool)
        zz=ps.simulate_port(mm,2,33,1.,10,**{**c,**order['liquidity']},cost=r.COST)
        ablations.append(dict(condition=name,**f.summarize(zz,c['calendar'])))
    # Show the contribution of each selected new condition on the 250-day base.
    for name in ('none','bullavg40_ge1.0','weekly_momentum','volume_weekly'):
        mm=np.asarray(base[balanced_cfg['base']]&gates[name],bool)
        zz=ps.simulate_port(mm,2,33,1.,10,**{**c,**order['liquidity']},cost=r.COST)
        ablations.append(dict(condition='long250_'+name,**f.summarize(zz,c['calendar'])))
    # Neighbor perturbations are predeclared, changing one dimension at a time.
    def selected_mask(pit=.30,age=15,rise=.12,market=.015,volume=1.,window=40):
        return np.asarray((e.price>=3)&(e.amount20>=3e7)&(e.signal_return<.095)&(e.market20<0)&(e.daily>=market)&
                          (e.drop_60_250>=pit)&(e.age_60_250<=age)&(e.rise_60_250<=rise)&
                          (e[f'vol_mean_{window}']>=volume)&(e.w_mom_improving==1),bool)
    assert np.array_equal(mask,selected_mask()),balanced_cfg
    sensitivity=[]
    for field,values in [('pit',(.25,.35)),('age',(10,20)),('rise',(.10,.15)),('market',(.01,.02)),
                         ('volume',(.9,1.1,1.2)),('window',(20,60,120))]:
        for value in values:
            mm=selected_mask(**{field:value});zz=ps.simulate_port(mm,2,33,1.,10,**{**c,**order['liquidity']},cost=r.COST)
            sensitivity.append(dict(change=field,value=value,**f.summarize(zz,c['calendar'])))
    for slots in (5,20):
        zz=ps.simulate_port(mask,2,33,1.,slots,**{**c,**order['liquidity']},cost=r.COST)
        sensitivity.append(dict(change='slots',value=slots,**f.summarize(zz,c['calendar'])))
    for cost in (.007,.01):
        zz=ps.simulate_port(mask,2,33,1.,10,**{**c,**order['liquidity']},cost=cost)
        sensitivity.append(dict(change='cost',value=cost,**f.summarize(zz,c['calendar'])))
    horizons=[]
    for h in (5,10,20,30):
        k=r.EXITS.index((h,0.,0.,0.));zz=ps.simulate_port(mask,2,k,1.,10,**{**c,**order['liquidity']},cost=r.COST)
        horizons.append(dict(hold=h,**f.summarize(zz,c['calendar'])))
    # Actual rolling policy: join signal-year masks and rankings, retain prior
    # positions across New Year, and attach each order's locked exit rule.
    rolling_mask=np.zeros(len(e),bool);rolling_rows=[];gross=c['returns'].copy();sold=c['exit_day'].copy();choices=[]
    signal_year=e.signal_date.to_numpy()//10000
    for year in (2024,2025,2026):
        picks=json.loads((v.OUT/f'picks_before_{year}.json').read_text());pick=picks[0];choices.append(dict(year=year,selection=pick))
        within=signal_year==year;mm=np.asarray(base[pick['base']]&gates[pick['gate']],bool)
        rolling_mask|=within&mm
        rows=order[pick['rank']]['rows'];rolling_rows.extend(rows[within[rows]].tolist())
        gross[within,2,0]=c['returns'][within,2,pick['exit']];sold[within,2,0]=c['exit_day'][within,2,pick['exit']]
    rows=np.asarray(rolling_rows,dtype=int);days=e.signal_index.to_numpy()+1
    # Year blocks are already chronological; each block preserves day/rank order.
    assert np.all(np.diff(days[rows])>=0)
    cc={**c,'rows':rows,'offsets':np.searchsorted(days[rows],np.arange(len(c['calendar'])+1)),
        'returns':gross,'exit_day':sold}
    zz=ps.simulate_port(rolling_mask,2,0,1.,10,**cc,cost=r.COST)
    rolling=f.summarize(zz,c['calendar'])
    chronological=[dict(year=year,actual_return=rolling['annual'][year-2022],drawdown=rolling['annual_drawdown'][year-2022],
                        trades=rolling['n'][year-2022],selection=choices[year-2024]['selection']) for year in (2024,2025,2026)]
    # Reconstruct all selected fills from raw prices, with no portfolio engine.
    panel=pd.read_pickle(r.OUT/'panel.pkl');errors=[]
    for x in trades.itertuples():
        dates,raw,cx,factor,pre,mi,upper,locked=panel[x.code];j=np.searchsorted(dates,x.entry_date);end=np.searchsorted(dates,x.exit_date)
        cap=np.floor(pre[j]*.98+1e-8);buy=raw[j,0] if raw[j,0]<=cap else cap
        assert raw[j,0]<upper[j] and (raw[j,0]<=cap or raw[j,2]<cap)
        assert not locked[end] and x.w_bar_date<=x.signal_date and x.m_bar_date<=x.signal_date
        assert mi[end]-mi[j]+1>=20
        price=raw[end,3] if mi[end]-mi[j]+1==20 else raw[end,0]
        errors.append(abs(price*factor[end]/buy/factor[j]-1-x.gross))
    assert max(errors)<2e-7
    s['audit']=dict(trades=len(trades),max_return_error=max(errors),period_end_never_after_signal=True)
    out=dict(selected=s,parameters={**f.PARAMETERS,'prior_peak_window':250,'floor_age_max':15,
                'prior_close_over_floor_max':1.12,'market_equal_weight_day_min':.015,
                'volume_balance_window':40,'bull_bear_mean_volume_min':1.,
                'volume_balance_excludes_signal':True,'candle_direction':'close vs open; doji ignored',
                'weekly_rule':'last completed week close-to-close return > prior completed week return',
                'weekly_macd_required':False,'monthly_filter_required':False,'macd_divergence_required':False},
             config=balanced_cfg,comparisons=comparisons,ablations=ablations,sensitivity=sensitivity,horizons=horizons,
             chronological=chronological,rolling=rolling,search_count=len(configs),snapshot_end=r.END,
             status='Retrospective fit on repeatedly inspected 2022-2026 history; no pristine holdout.',
             history_note='Daily history starts around 2021-08. 250/500-day rules require full warmup. Monthly MACD requires 26 completed months; absent values are never treated as a pass.',
             selection_note='Balanced: hold20, >=90 trades, >=8 per year, each year positive, full-period DD<10%; maximize CAGR. Return-focused: hold30, >=80 trades, >=8 per year, DD<12%. Both selection rules use all five years.',
             baseline_source='results/research_v2/final_report.json')
    (v.OUT/'report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(ablations).drop(columns=['trade_stats','annual','annual_drawdown','n']).to_csv(v.OUT/'ablations.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
