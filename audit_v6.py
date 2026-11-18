"""Independent price/return and causal confirmation checks for v6."""
import json
import numpy as np
import pandas as pd
import confirm_v6 as c

def main():
    e=pd.read_pickle(c.OUT/'outcomes.pkl');p=pd.read_pickle(c.d.r.OUT/'panel.pkl');vol=c.d.volume_data(p)
    checked=prefixes=0;error=0.
    for x in e[e.entry_status.eq('filled')].itertuples():
        dates,raw,close,factor,pre,mi,*_=p[x.code];i=int(x.stock_index);j=i+1
        assert int(dates[i])==x.signal_date and int(dates[j])==x.entry_date
        if x.variant=='confirm':
            z,why=c.confirmation(raw[:i+1],close[:i+1],vol[x.code][:i+1,0],mi[:i+1],int(x.dry_stock_index),int(x.floor_index))
            assert z==i and why=='confirmed';prefixes+=1
        for h in c.HORIZONS:
            if getattr(x,f'status_{h}')!='complete':continue
            date=int(getattr(x,f'exit_date_{h}'));k=int(np.searchsorted(dates,date));price=getattr(x,f'exit_price_{h}')*100
            assert dates[k]==date and k>j
            reason=getattr(x,f'reason_{h}')
            if reason=='expiry':assert abs(price-raw[k,3])<1e-7 and mi[k]-mi[j]+1==h
            elif reason in ('expiry_delayed','stop_gap','stop_queued'):assert abs(price-raw[k,0])<1e-7
            elif reason=='stop_intraday':
                level=np.floor(raw[j,0]*factor[j]*(1-c.STOP)/factor[k]+1e-7)
                assert abs(price-level)<1e-7 and raw[k,0]>price-1e-7 and price>=raw[k,2]-1e-7, (x.code,h,price,level,raw[k])
            expected=(price*factor[k]/(raw[j,0]*factor[j]))*(1-c.d.r.COST/2)/(1+c.d.r.COST/2)-1
            error=max(error,abs(expected-getattr(x,f'net_{h}')));assert error<1e-12;checked+=1
    summary=pd.read_csv(c.OUT/'summary.csv');old=pd.read_csv(c.d.OUT/'summary.csv').query("profile=='extreme'")
    for h in c.HORIZONS:
        a=summary[(summary.variant=='direct')&(summary.horizon==h)].iloc[0];b=old[old.horizon==h].iloc[0]
        assert a.n==b.n and abs(a['mean']-b['mean'])<1e-12
    result=dict(verified_horizon_returns=checked,max_return_error=error,confirmation_prefix_checks=prefixes,
                direct_baseline_unchanged=True,tests=8,limitation='Daily OHLC cannot prove intraday fills; stop gaps can exceed 15%.')
    (c.OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':main()
