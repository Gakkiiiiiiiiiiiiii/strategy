"""Restore mandatory MACD-bottom eligibility before the countertrend impulse.

The broad v8 universe is only a control. No return-based parameter search.
MACD/volume scores order same-day watchlists; event means are not portfolios.
"""
import json
import math
import socket
import numpy as np
import pandas as pd
import leaders_v8 as v
import research_v3 as m3

OUT=v.d.bt.ROOT/'results/research_v9'
HORIZONS=v.HORIZONS
SPEC=dict(v.SPEC,version='v9',
    mandatory_bottom='At the FIRST impulse S: L=min adjusted close[S-60:S]; H=max adjusted close[S-120:floor+1]; decline>=30%, floor age<=8, close[S-1]<=1.08*L.',
    mandatory_long_decline='Primary profile requires last occurrence of original peak to floor>=60 stock sessions; pre-S MA60<MA60 at S-21<MA60 at S-41. A flat plateau followed by a recent crash must not count as a long decline.',
    comparison='macd_core retains original bottom rules without the new duration/trend requirement; long_decline is the primary profile. V8 broad stock selection is never the main profile.',
    anchor_selection='Choose most recent qualifying bottom impulse 5-30 market sessions before T, after applying the profile filters; do not first select a generic impulse then merely label its depth.',
    ranking='At T: clip(log(bull/bear mean volume over prior40),-1,1)+confirmed near MACD divergence+0.5*histogram recovery; descending, then amount20, then code.',
    ranking_scope='Ordering of same-day candidate watchlists only. Event averages do not model capital/slots and do not quantify ranking alpha.',
    confirmation_floor_distance='Explicitly report T-1 close/L-1 and actual buy/L-1. The old8% rule is mandatory at S, not silently reapplied after the waiting phase; near8 at T is an additional diagnostic.',
    execution_change='None versus v8: next open, no9.5% cap, hard15% stop, same horizons/cost. This isolates the restored selection rules.',
    controls='Fixed v8 nonleader controls on same signal date, board and liquidity quartile; not selected by later returns.')


def dump(name,value):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def bottom(close,i):
    f=i-60+int(np.argmin(close[i-60:i]));prior=close[i-120:f+1]
    # An old equal-price plateau must not artificially lengthen the decline.
    p=i-120+int(np.flatnonzero(prior==np.max(prior))[-1])
    return dict(floor_index=f,old_peak_index=p,decline=float(1-close[f]/close[p]),floor_age=int(i-f),
        prior_from_floor=float(close[i-1]/close[f]-1),decline_days=int(f-p),
        ma60_change=float(close[i-60:i].mean()/close[i-80:i-20].mean()-1),
        ma60_previous_change=float(close[i-80:i-20].mean()/close[i-100:i-40].mean()-1))


def eligible(b,long=False):
    core=b['decline']>=.30 and b['floor_age']<=8 and b['prior_from_floor']<=.08
    return bool(core and (not long or (b['decline_days']>=60 and b['ma60_change']<0 and b['ma60_previous_change']<0)))


def build(panel):
    source=pd.read_pickle(v.d.r.OUT/'events.pkl');cand=pd.read_pickle(v.OUT/'events.pkl')
    features=pd.read_pickle(m3.OUT/'features.pkl');volumes=v.d.volume_data(panel)
    market=pd.read_csv(v.d.r.OUT/'market.csv');level=(1+market.daily).cumprod().to_numpy()
    anchors=source[(source.price>=3)&(source.amount20>=3e7)&(source.market20<0)&(source.daily<=0)&
                   (source.drawdown>=.30)&(source.floor_days<=8)&(source.prior_from_floor<=.08)].copy()
    records=[];bycode={code:g.sort_values('signal_index') for code,g in anchors.groupby('code')}
    for code,group in cand.groupby('code'):
        if code not in bycode:continue
        dates,raw,close,factor,pre,mi,*_=panel[code];volume=volumes[code][:,0]
        anchor_rows=[]
        for a in bycode[code].itertuples():
            b=bottom(close,a.stock_index)
            assert eligible(b)
            anchor_rows.append((a,b))
        for t in group.itertuples():
            choices=[(a,b) for a,b in anchor_rows if 5<=t.signal_index-a.signal_index<=30]
            for profile in ('macd_core','long_decline'):
                q=[(a,b) for a,b in choices if eligible(b,profile=='long_decline')]
                if not q:continue
                a,b=q[-1];i=int(t.stock_index);s=int(a.stock_index)
                if level[t.signal_index-1]>=level[a.signal_index]:continue
                peak=s+int(np.argmax(close[s:i]));drop=float(1-close[i-1]/close[peak]);defend=bool(np.min(close[s:i])>=close[s-1])
                dry=float(volume[i-3:i].mean()/volume[s:peak+1].mean());maxdd=float(np.max(1-close[s:i]/np.maximum.accumulate(close[s:i])))
                feature=features.loc[t.event_id];balance=feature.vol_mean_40
                score=float(np.clip(np.log(balance if np.isfinite(balance) and balance>0 else 1),-1,1)+feature.near_divergence+.5*feature.hist_turn)
                rec=dict(profile=profile,event_id=int(t.event_id),code=code,name=t.name,signal_date=int(t.signal_date),
                    signal_index=int(t.signal_index),stock_index=i,impulse_date=int(a.signal_date),impulse_index=s,
                    floor_date=int(dates[b['floor_index']]),old_peak_date=int(dates[b['old_peak_index']]),**b,
                    amount20=float(t.amount20),board=int(t.board),amount_bucket=int(t.amount_bucket),
                    market_today=float(t.daily),signal_return=float(t.signal_return),impulse_return=float(a.signal_return),
                    score=score,volume_balance40=float(balance) if np.isfinite(balance) else None,
                    macd_divergence=int(feature.near_divergence),macd_hist_turn=int(feature.hist_turn),
                    peak_date=int(dates[peak]),draw_from_peak=drop,dry_ratio=dry,phase_max_drawdown=maxdd,
                    pullback=bool(.05<=drop<=.20 and defend and dry<=.70),resistant=bool(maxdd<.05 and defend),
                    recovered=bool(maxdd>=.05 and drop<.05 and defend),
                    confirm_prior_from_floor=float(close[i-1]/close[b['floor_index']]-1),
                    confirm_near8=bool(close[i-1]<=close[b['floor_index']]*1.08),
                    industry=t.industry if isinstance(t.industry,str) else '',sector_confirm=bool(t.sector_confirm),
                    industry_date=t.industry_date if isinstance(t.industry_date,str) else '')
                records.append(rec)
    e=pd.DataFrame(records).sort_values(['profile','signal_date','score','amount20','code'],ascending=[True,True,False,False,True]).reset_index(drop=True)
    e['candidate_rank']=e.groupby(['profile','signal_date']).cumcount()+1
    e.to_pickle(OUT/'events.pkl')
    print('SETUPS',e.groupby('profile')[['pullback','resistant','confirm_near8']].sum().to_dict(),flush=True)
    return e,market


def flags_for(e,panel):
    import update_market as u
    flags=pd.read_pickle(v.c6.OUT/'status_maps.pkl')
    # Load only the codes in this much narrower experiment, preserving the
    # same historical status sources as v8 without rereading the full cache.
    for code in e.code.unique():
        path=v.OUT/'status_cache'/f'{code}.csv'
        if not path.exists():continue
        x=pd.read_csv(path);x['date']=x.date.str.replace('-','').astype(int)
        flags.setdefault(code,{}).update({int(a.date):int(a.isST) for a in x.dropna(subset=['isST']).itertuples()})
    directory=OUT/'status_cache';directory.mkdir(exist_ok=True)
    for path in directory.glob('*.csv'):
        x=pd.read_csv(path);x['date']=x.date.str.replace('-','').astype(int)
        flags.setdefault(path.stem,{}).update({int(a.date):int(a.isST) for a in x.dropna(subset=['isST']).itertuples()})
    requests=[]
    for code,g in e.groupby('code'):
        dates=panel[code][0];required=set(g.signal_date)|set(g.impulse_date)
        required.update(int(dates[i+1]) for i in g.stock_index if i+1<len(dates))
        missing=required-set(flags.get(code,{}))
        if missing:requests.append((code,min(missing)))
    if requests:
        socket.setdefaulttimeout(25);assert u.bs.login().error_code=='0'
        try:
            for code,start in requests:
                x=u.frame(u.bs.query_history_k_data_plus(code,'date,isST,tradestatus',str(pd.Timestamp(str(start)).date()),'2026-11-16',frequency='d',adjustflag='3'))
                assert len(x)>0;u.save(x,directory/f'{code}.csv');x['date']=x.date.str.replace('-','').astype(int)
                flags.setdefault(code,{}).update({int(a.date):int(a.isST) for a in x.dropna(subset=['isST']).itertuples()})
        finally:u.bs.logout()
    print('EXTRA_STATUS_CODES',len(requests),flush=True)
    return flags


def outcomes(e,panel,market,flags):
    cal=market.date.to_numpy();bm=pd.read_csv(v.d.OUT/'sh.000300.csv');bm['date']=bm.date.str.replace('-','').astype(int);bm=bm.set_index('date')
    rows=[]
    for x in e.itertuples():
        dates,raw,close,factor,pre,mi,*_=panel[x.code];j=x.stock_index+1;st=flags.get(x.code,{})
        rec=x._asdict();rec.pop('Index');rec.update(entry_status='no_next_bar',entry_date=0,buy_price=None,buy_from_floor=None)
        for h in HORIZONS:rec.update({f'net_{h}':np.nan,f'nostop_{h}':np.nan,f'end_{h}':0,f'excess_{h}':np.nan,f'exit_date_{h}':0,f'exit_price_{h}':np.nan,f'reason_{h}':''})
        if j>=len(raw) or mi[j]!=x.signal_index+1:rows.append(rec);continue
        rec.update(entry_date=int(dates[j]),buy_price=float(raw[j,0]/100),buy_from_floor=float(raw[j,0]*factor[j]/close[x.floor_index]-1))
        checks=[x.impulse_date,x.signal_date,int(dates[j])]
        if any(k not in st for k in checks):rec['entry_status']='unknown_st';rows.append(rec);continue
        if any(st[k] for k in checks):rec['entry_status']='st_excluded';rows.append(rec);continue
        rate=20 if x.code[3:6] in {'300','301','688'} else 10
        upper=max(math.floor((pre[j]*(100+rate)+50)/100),pre[j]+1)
        if raw[j,0]>=upper:rec['entry_status']='upper_open';rows.append(rec);continue
        rec['entry_status']='filled'
        for h in HORIZONS:
            target=int(mi[j]+h-1)
            a,_=v.d.executable_exit(raw,factor,mi,pre,st,x.code,j,target,cal)
            b,_=v.c6.hard_exit(raw,factor,mi,pre,dates,st,x.code,j,target,cal)
            if a is None or b is None:continue
            net=lambda px:float(px/(raw[j,0]*factor[j])*(1-v.d.r.COST/2)/(1+v.d.r.COST/2)-1)
            benchmark=float(bm.loc[int(cal[target]),'close']/bm.loc[int(dates[j]),'open']*(1-v.d.r.COST/2)/(1+v.d.r.COST/2)-1)
            rec.update({f'net_{h}':net(b['adjusted']),f'nostop_{h}':net(a[1]),f'end_{h}':int(max(dates[a[0]],dates[b['z']])),
                f'exit_date_{h}':int(dates[b['z']]),f'exit_price_{h}':b['price'],f'reason_{h}':b['reason'],f'excess_{h}':net(b['adjusted'])-benchmark})
        rows.append(rec)
    result=pd.DataFrame(rows);result.to_pickle(OUT/'outcomes.pkl');return result


def analyze(e):
    controls=pd.read_pickle(v.OUT/'outcomes.pkl');summary=[];matched=[];reviews=[]
    for profile,frame in e.groupby('profile'):
        groups={'all_leaders':np.ones(len(frame),bool),'pullback':frame.pullback,'resistant':frame.resistant,
                'both':frame.pullback|frame.resistant,'both_near8':(frame.pullback|frame.resistant)&frame.confirm_near8,
                'all_near8':frame.confirm_near8,'both_sector':(frame.pullback|frame.resistant)&frame.sector_confirm}
        for group,mask in groups.items():
            first=frame[mask].sort_values(['signal_date','code']).drop_duplicates(['code','impulse_date'])
            for h in HORIZONS:
                q=first[first.entry_status.eq('filled')&first[f'net_{h}'].notna()]
                q=v.nonoverlap(q,h)
                key=['signal_date','board','amount_bucket']
                c=controls[controls.entry_status.eq('filled')&~controls.leader&controls[f'net_{h}'].notna()].groupby(key)[f'net_{h}'].agg(['mean','count'])
                for split,z in [('all',q),('through2024',q[q[f'end_{h}']<=20241231]),('from2025',q[q.signal_date>=20250101])]:
                    row=dict(profile=profile,group=group,horizon=h,split=split,**v.stats(z[f'net_{h}']),entry_dates=int(z.signal_date.nunique()))
                    row.update(nostop_mean=float(z[f'nostop_{h}'].mean()) if len(z) else None,median_decline_days=float(z.decline_days.median()) if len(z) else None,
                        median_buy_from_floor=float(z.buy_from_floor.median()) if len(z) else None,interval=v.interval(z,f'net_{h}'))
                    summary.append(row)
                    p=z.join(c,on=key);p=p[p['count']>=3].copy();p['delta']=p[f'net_{h}']-p['mean']
                    matched.append(dict(profile=profile,group=group,horizon=h,split=split,n=len(p),mean_delta=float(p.delta.mean()) if len(p) else None,interval=v.interval(p,'delta')))
                if group in ('pullback','resistant') and h in (20,30):
                    q=q.copy();q['horizon']=h;q['shape']=group;reviews.append(q)
    report=dict(spec=SPEC,counts=e.groupby(['profile','entry_status']).size().reset_index(name='n').to_dict('records'),summary=summary,matched=matched)
    dump('report.json',report);pd.DataFrame(summary).drop(columns='interval').to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    pd.concat(reviews,ignore_index=True).to_csv(OUT/'trades_review.csv',index=False,encoding='utf-8-sig')
    signals=e[(e.profile=='long_decline')&(e.pullback|e.resistant)].sort_values('signal_date').drop_duplicates(['code','impulse_date'])
    signals=signals.sort_values(['signal_date','score','amount20','code'],ascending=[False,False,False,True])
    signals['primary_rank']=signals.groupby('signal_date').cumcount()+1
    signals.to_csv(OUT/'primary_signals.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(summary).query("split=='all' and horizon in [20,30] and group in ['pullback','resistant','both','both_near8']")[["profile","group","horizon","n","mean","win","payoff"]].to_string(index=False),flush=True)


def audit(e,panel,market):
    features=pd.read_pickle(m3.OUT/'features.pkl');checks=0;errors=[]
    for x in e.itertuples():
        dates,raw,close,factor,pre,mi,*_=panel[x.code];s=x.impulse_index;i=x.stock_index
        b=bottom(close[:s],s)
        assert eligible(b,x.profile=='long_decline')
        assert dates[b['floor_index']]==x.floor_date and dates[b['old_peak_index']]==x.old_peak_date
        assert 5<=mi[i]-mi[s]<=30 and x.floor_date<x.impulse_date<x.signal_date
        dif,dea,hist=m3.macd(close[:i+1]);_,near,_,p,_=m3.divergence(close[:i+1],dif,hist,i)
        assert int(near)==x.macd_divergence and int(hist[-1]>hist[-2] and hist[-2]>=hist[-3])==x.macd_hist_turn
        assert p<0 or p+2<i
        if x.entry_status=='filled':
            j=int(np.searchsorted(dates,x.entry_date));assert mi[j]==mi[i]+1
            for h in HORIZONS:
                net=getattr(x,f'net_{h}')
                if not np.isfinite(net):continue
                assert mi[j]+h-1<len(market)
                z=int(np.searchsorted(dates,getattr(x,f'exit_date_{h}')));px=getattr(x,f'exit_price_{h}')*100
                assert z>j and raw[z,2]-1e-5<=px<=raw[z,1]+1e-5
                expected=px*factor[z]/(raw[j,0]*factor[j])*(1-v.d.r.COST/2)/(1+v.d.r.COST/2)-1
                errors.append(abs(expected-net))
        checks+=1
    assert max(errors,default=0)<1e-12
    dump('audit.json',dict(prefix_bottom_and_macd_checks=checks,net_return_checks=len(errors),max_return_error=max(errors,default=0),
        primary_requires_core_and_long_decline=True,prior_results_preserved=True))
    print('AUDIT',checks,len(errors),max(errors,default=0),flush=True)


def main():
    dump('spec.json',SPEC);panel=pd.read_pickle(v.d.r.OUT/'panel.pkl');e,market=build(panel)
    flags=flags_for(e,panel);result=outcomes(e,panel,market,flags);audit(result,panel,market);analyze(result)


if __name__=='__main__':main()
