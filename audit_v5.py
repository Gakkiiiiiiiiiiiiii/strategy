"""Independent return checks and prefix invariance for the dry-volume study."""
import json
import numpy as np
import pandas as pd
import dryup_v5 as d
import research_v2 as r


def main():
    e=pd.read_pickle(d.OUT/'outcomes.pkl');p=pd.read_pickle(r.OUT/'panel.pkl')
    vols=d.volume_data(p);cal=pd.read_csv(r.OUT/'market.csv').date.to_numpy()
    count=0;error=0.;prefix=0
    for x in e[e.entry_status.isin(['filled','upper_open'])].itertuples():
        dates,raw,c,factor,pre,mi,*_=p[x.code];i=x.stock_index;j=i+1
        assert dates[j]==x.entry_date and mi[j]==mi[i]+1
        for h in d.HORIZONS:
            if getattr(x,f'status_{h}')=='immature':
                assert mi[j]+h-1>=len(cal);continue
            if getattr(x,f'status_{h}')!='complete':continue
            z=int(np.searchsorted(dates,getattr(x,f'exit_date_{h}')))
            target=mi[j]+h-1;assert mi[z]>=target and z>j
            sell=raw[z,3] if mi[z]==target else raw[z,0]
            gross=sell*factor[z]/(raw[j,0]*factor[j])-1
            net=(1+gross)*(1-r.COST/2)/(1+r.COST/2)-1
            delta=abs(net-getattr(x,f'net_{h}'));error=max(error,delta)
            assert delta<1e-10 and abs(sell/100-getattr(x,f'exit_price_{h}'))<1e-9
            count+=1
    codes=list(e.code.drop_duplicates().iloc[::max(1,e.code.nunique()//20)])[:20]
    if 'sz.300369' not in codes:codes.append('sz.300369')
    for code in codes:
        dates,raw,c,*_=p[code];vol,amt=vols[code].T
        all_rows=d.scan(c,vol,amt,raw[:,3])
        candidates=e[e.code==code].stock_index
        for i in sorted(set(candidates.iloc[[0,len(candidates)//2,-1]].tolist())):
            truncated=d.scan(c[:i+1],vol[:i+1],amt[:i+1],raw[:i+1,3])
            np.testing.assert_allclose(truncated,all_rows[all_rows[:,0]<=i],atol=1e-12,rtol=0)
            prefix+=1
    for key in d.PROFILES:
        signals=pd.read_csv(d.OUT/f'{key}_signals.csv')
        assert not signals.duplicated(['code','floor_date','peak_date']).any()
        for h in d.HORIZONS:
            t=pd.read_csv(d.OUT/f'{key}_{h}_trades.csv')
            for _,g in t.sort_values('signal_date').groupby('code'):
                assert np.all(g.signal_date.to_numpy()[1:]>g[f'exit_date_{h}'].to_numpy()[:-1])
    result=dict(verified_horizon_returns=count,max_return_error=error,prefix_recalculations=prefix,
        same_wave_deduplicated=True,same_stock_overlap_excluded=True,immature_not_filled=True,
        limitation='Numerical/data consistency checks cannot prove intraday fills or remove selection bias.')
    (d.OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)


if __name__=='__main__':main()
