"""Raw-price/prefix audit of the fixed countertrend-leader study."""
import json
import math
import numpy as np
import pandas as pd
import leaders_v8 as v


def main():
    events=pd.read_pickle(v.OUT/'events.pkl');panel=pd.read_pickle(v.d.r.OUT/'panel.pkl');vols=v.d.volume_data(panel)
    market=pd.read_csv(v.d.r.OUT/'market.csv');cal=market.date.to_numpy();level=(1+market.daily).cumprod().to_numpy()
    checked=0
    # Rebuild anchor detection from bars, with every input truncated at T.
    subset=events[events.leader].iloc[::max(1,int(events.leader.sum())//100)]
    for x in subset.itertuples():
        dates,raw,close,factor,pre,mi,*_=panel[x.code];i=x.stock_index
        raw=raw[:i+1];close=close[:i+1];pre=pre[:i+1];mi=mi[:i+1];volume,amount=vols[x.code][:i+1].T
        possible=[]
        for a in range(max(120,i-50),i):
            age=x.signal_index-mi[a]
            if not 5<=age<=30:continue
            body=raw[a,3]-raw[a,0]
            if not (market.daily.iloc[mi[a]]<=0 and market.market20.iloc[mi[a]]<0):continue
            if raw[a,3]<300 or amount[a-19:a+1].mean()<3e7:continue
            if raw[a,3]/pre[a]-1<=.06 or body<=0 or raw[a,1]-raw[a,3]>body/3+1e-8:continue
            if volume[a]<=volume[a-5:a].mean() or volume[a]<=volume[a-10:a].mean():continue
            possible.append(a)
        assert possible and possible[-1]==x.impulse_index
        s=possible[-1];p=s+int(np.argmax(close[s:i]))
        assert p==x.peak_index
        assert level[x.signal_index-1]<level[mi[s]]
        drop=1-close[i-1]/close[p];dry=volume[i-3:i].mean()/volume[s:p+1].mean()
        np.testing.assert_allclose([drop,dry],[x.draw_from_peak,x.dry_ratio],atol=1e-12)
        defend=np.min(close[s:i])>=close[s-1]
        assert x.pullback==bool(.05<=drop<=.20 and defend and dry<=.70)
        maxdd=float(np.max(1-close[s:i]/np.maximum.accumulate(close[s:i])))
        assert abs(maxdd-x.phase_max_drawdown)<1e-12
        assert x.resistant==bool(maxdd<.05 and defend)
        assert x.recovered==bool(maxdd>=.05 and drop<.05 and defend)
        checked+=1
    receipts=json.loads((v.OUT/'industry_sources.json').read_text(encoding='utf8'))
    for source in receipts:
        assert source['max_update']<=source['requested_asof']<f"{source['signal_year']}-01-01"
    sector_checks=0
    for x in subset.iloc[::5].itertuples():
        if not isinstance(x.industry,str) or not x.industry:continue
        mapping=pd.read_csv(v.OUT/'industry_cache'/f'{x.signal_date//10000}.csv')
        peers=mapping.loc[mapping.industry.eq(x.industry)&mapping.code.ne(x.code),'code']
        target=np.arange(x.signal_index-5,x.signal_index+1);values=[]
        for code in peers:
            if code not in panel:continue
            dates,raw,close,factor,pre,mi,*_=panel[code];row=np.full(6,np.nan)
            positions=np.searchsorted(mi,target)
            for n,p in enumerate(positions):
                if p<len(mi) and mi[p]==target[n] and p>=60:row[n]=raw[p,3]/pre[p]-1
            values.append(row)
        values=np.asarray(values);means=np.nanmean(values,axis=0)
        count=np.isfinite(values[:,-1]).sum();breadth=(values[:,-1]>0).sum()/count
        five=np.prod(1+means[:-1])-1
        np.testing.assert_allclose([means[-1],breadth,five],[x.sector_today,x.sector_breadth,x.sector_prior5],atol=1e-12)
        assert count==x.sector_members
        sector_checks+=1
    if not (v.OUT/'outcomes.pkl').exists():
        print('PREFIX',checked,'dated industry snapshots',len(receipts),'leave-one-out sectors',sector_checks,flush=True);return
    results=pd.read_pickle(v.OUT/'outcomes.pkl');flags=v.status_maps();errors=[];returns=0
    sector_results=pd.read_pickle(v.OUT/'sector_outcomes.pkl').set_index('event_id');sector_return_checks=0
    for x in subset.iloc[::10].itertuples():
        expected=sector_results.loc[x.event_id,'sector_return_30']
        if not np.isfinite(expected):continue
        members=pd.read_csv(v.OUT/'industry_cache'/f'{x.signal_date//10000}.csv')
        members=members.loc[members.industry.eq(x.industry)&members.code.ne(x.code),'code']
        entry=x.signal_index+1;end=entry+29;gross=[]
        for code in members:
            if code not in panel:continue
            dates,raw,close,factor,pre,mi,*_=panel[code]
            j=int(np.searchsorted(mi,entry));z=int(np.searchsorted(mi,end,side='right')-1)
            if j>=len(mi) or mi[j]!=entry or z<0:continue
            gross.append(close[z]/(raw[j,0]*factor[j]))
        actual=np.mean(gross)*(1-v.d.r.COST/2)/(1+v.d.r.COST/2)-1
        assert len(gross)==sector_results.loc[x.event_id,'sector_n_30']
        assert abs(actual-expected)<1e-12
        sector_return_checks+=1
    for x in results[results.entry_status.eq('filled')].itertuples():
        dates,raw,close,factor,pre,mi,*_=panel[x.code]
        j=int(np.searchsorted(dates,x.entry_date));i=x.stock_index
        assert mi[j]==mi[i]+1
        assert flags[x.code][int(dates[i])]==0 and flags[x.code][int(dates[j])]==0
        if x.leader:assert flags[x.code][x.impulse_date]==0
        for h in v.HORIZONS:
            value=getattr(x,f'net_{h}')
            if not np.isfinite(value):continue
            target=int(mi[j]+h-1);assert target<len(cal)
            z=int(np.searchsorted(dates,getattr(x,f'stop_exit_{h}')))
            a=int(np.searchsorted(dates,getattr(x,f'nostop_exit_{h}')))
            assert z>j and a>j and mi[a]>=target
            px=getattr(x,f'stop_price_{h}')*100
            assert raw[z,2]-1e-5<=px<=raw[z,1]+1e-5
            expected=px*factor[z]/(raw[j,0]*factor[j])*(1-v.d.r.COST/2)/(1+v.d.r.COST/2)-1
            errors.append(abs(expected-value))
            price=raw[a,3] if mi[a]==target else raw[a,0]
            expected=price*factor[a]/(raw[j,0]*factor[j])*(1-v.d.r.COST/2)/(1+v.d.r.COST/2)-1
            errors.append(abs(expected-getattr(x,f'nostop_{h}')))
            reason=getattr(x,f'reason_{h}')
            if reason=='stop_intraday':
                threshold=math.floor(raw[j,0]*factor[j]*.85/factor[z]+1e-7)
                assert abs(px-threshold)<1e-5 and raw[z,0]>threshold and raw[z,2]<=threshold
            elif reason in ('stop_gap','stop_queued','expiry_delayed'):
                assert abs(px-raw[z,0])<1e-5
            elif reason=='expiry':assert mi[z]==target and abs(px-raw[z,3])<1e-5
            else:raise AssertionError(reason)
            returns+=2
    assert max(errors)<1e-12
    audit=dict(prefix_only_rebuilds=checked,dated_industry_snapshots=len(receipts),leave_one_out_sector_checks=sector_checks,industry_basket_return_checks=sector_return_checks,independent_return_checks=returns,
               max_return_error=max(errors),entry_dates_and_ST_verified=True,stop_ticks_and_T1_verified=True,
               warning='Validates implemented calculations, not economic causality or future profitability.')
    v.dump('audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
