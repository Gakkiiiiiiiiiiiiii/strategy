"""Bottom volume, causal MACD divergence, and completed weekly/monthly bars.

Extends the frozen v2 event universe and finite-cash simulator. 2026 has
already been repeatedly inspected; every selection remains retrospective.
"""
import argparse
import json
import time
from pathlib import Path
from itertools import product
from contextlib import ExitStack
import numpy as np
import pandas as pd
from numba import njit
import backtest as bt
import research_v2 as r
import portfolio_search as ps
import finalize_v2 as final
from review_v2 import stats

OUT=bt.ROOT/'results/research_v3'

def macd(close):
    s=pd.Series(close,dtype=float)
    dif=s.ewm(span=12,adjust=False).mean()-s.ewm(span=26,adjust=False).mean()
    dea=dif.ewm(span=9,adjust=False).mean()
    return dif.to_numpy(),dea.to_numpy(),(2*(dif-dea)).to_numpy()

@njit(cache=True)
def divergence(c,dif,hist,i,lookback=120):
    # Two-sided pivots are used only once their two confirmation bars have
    # closed BEFORE the signal. No right-hand future bar may confirm a pivot.
    recent=-1;prior=-1;strict=0.;near=0.;histdiv=0.
    for p in range(i-3,max(2,i-18),-1):
        if c[p]<=c[p-1] and c[p]<c[p-2] and c[p]<=c[p+1] and c[p]<c[p+2]:
            recent=p;break
    if recent>=0:
        for q in range(recent-5,max(2,i-lookback),-1):
            if c[q]<=c[q-1] and c[q]<c[q-2] and c[q]<=c[q+1] and c[q]<c[q+2]:
                if dif[q]<0 and dif[recent]<0 and dif[recent]>dif[q]:
                    if c[recent]<c[q]*.99:strict=1.;prior=q
                    if c[recent]<=c[q]*1.02:near=1.;prior=q
                if hist[q]<0 and hist[recent]>hist[q] and c[recent]<c[q]*.99:histdiv=1.
    return strict,near,histdiv,recent,prior

def build():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    e=pd.read_pickle(r.OUT/'events.pkl');panel=pd.read_pickle(r.OUT/'panel.pkl')
    _,updates=bt.updated_bars(20261104,r.END,{})
    output=[]
    with ExitStack() as stack:
        _,read=bt.market_files(Path('D:/tdx/vipdoc'),stack)
        for n,(code,group) in enumerate(e.groupby('code')):
            dates,raw,c,factor,pre,mi,*_=panel[code]
            b=np.frombuffer(read(code.replace('.','')),bt.BAR_DTYPE)
            volume=dict(zip(b['date'],b['volume']))
            if code in updates:
                extra,_=updates[code];volume.update(dict(zip(extra['date'],extra['volume'])))
            v=np.asarray([volume[int(d)] for d in dates],dtype=float)
            up=raw[:,3]>raw[:,0];down=raw[:,3]<raw[:,0]
            dif,dea,hist=macd(c)
            bullsum=np.r_[0.,np.cumsum(v*up)];bearsum=np.r_[0.,np.cumsum(v*down)]
            bulln=np.r_[0,np.cumsum(up)];bearn=np.r_[0,np.cumsum(down)]
            dt=pd.to_datetime(dates.astype(str),format='%Y%m%d')
            # Period labels are calendar period ends. Forward-fill is allowed
            # only from completed labels at or before the event date.
            frames={}
            for freq,label in [('W-FRI','w'),('ME','m')]:
                frame=pd.DataFrame({'c':c,'v':v,'open':raw[:,0]*factor},index=dt).resample(freq).agg({'c':'last','v':'sum','open':'first'}).dropna()
                dc,da,dh=macd(frame.c)
                frame['dif']=dc;frame['dea']=da;frame['hist']=dh
                frame['ma6']=frame.c.rolling(6).mean();frame['ma12']=frame.c.rolling(12).mean()
                frame['ma26']=frame.c.rolling(26).mean()
                frames[label]=frame
            for row in group.itertuples():
                i=int(row.stock_index);rec={'event_id':row.Index}
                for days in (20,40,60,120):
                    if i<days:continue
                    a=i-days
                    uv=bullsum[i]-bullsum[a];dv=bearsum[i]-bearsum[a]
                    un=bulln[i]-bulln[a];dn=bearn[i]-bearn[a]
                    rec[f'vol_mean_{days}']=(uv/un)/(dv/dn) if un>=3 and dn>=3 and dv>0 else np.nan
                    rec[f'vol_share_{days}']=uv/(uv+dv) if uv+dv else np.nan
                    rec[f'vol_total_{days}']=uv/dv if dv>0 else np.nan
                early=(bearsum[i-10]-bearsum[i-40])/max(1,bearn[i-10]-bearn[i-40])
                late=(bearsum[i]-bearsum[i-10])/max(1,bearn[i]-bearn[i-10])
                rec['down_volume_decay']=late/early if early>0 else np.nan
                strict,near,hd,p,q=divergence(c,dif,hist,i)
                rec.update(divergence=strict,near_divergence=near,hist_divergence=hd,
                           pivot_new_date=int(dates[p]) if p>=0 else 0,pivot_old_date=int(dates[q]) if q>=0 else 0,
                           dif=dif[i]/c[i],hist=hist[i]/c[i],dif_rising=int(dif[i]>dif[i-1]),
                           hist_rising3=int(hist[i]>hist[i-1]>hist[i-2]),
                           hist_turn=int(hist[i]>hist[i-1] and hist[i-1]>=hist[i-2]),
                           macd_cross=int(dif[i]>dea[i] and dif[i-1]<=dea[i-1]))
                p2=i-15+np.argmin(c[i-15:i]);q2=max(0,i-120)+np.argmin(c[max(0,i-120):i-20])
                rec['window_divergence']=int(c[p2]<=c[q2]*1.02 and dif[p2]>dif[q2] and dif[q2]<0)
                for fw,pw in ((60,250),(120,250),(120,500),(250,500)):
                    if i<pw:
                        rec.update({f'drop_{fw}_{pw}':np.nan,f'age_{fw}_{pw}':np.nan,f'rise_{fw}_{pw}':np.nan})
                    else:
                        bottom=i-fw+int(np.argmin(c[i-fw:i]));peak=i-pw+int(np.argmax(c[i-pw:bottom+1]))
                        rec.update({f'drop_{fw}_{pw}':1-c[bottom]/c[peak],f'age_{fw}_{pw}':i-bottom,
                                    f'rise_{fw}_{pw}':c[i-1]/c[bottom]-1})
                for label,frame in frames.items():
                    j=frame.index.searchsorted(dt[i],side='right')-1
                    rec[f'{label}_bar_date']=int(frame.index[j].strftime('%Y%m%d')) if j>=0 else 0
                    rec[f'{label}_count']=j+1
                    for key in ('hist_up','dif_up','above_ma6','above_ma12','above_ma26','mom_improving','divergence'):
                        rec[f'{label}_{key}']=np.nan
                    if j>=5:
                        last=frame.iloc[j];prev=frame.iloc[j-1]
                        rec[f'{label}_above_ma6']=int(last.c>=last.ma6)
                        rec[f'{label}_mom_improving']=int(last.c/prev.c>prev.c/frame.c.iloc[j-2])
                    if j>=11:rec[f'{label}_above_ma12']=int(frame.c.iloc[j]>=frame.ma12.iloc[j])
                    if j>=25:
                        rec[f'{label}_above_ma26']=int(frame.c.iloc[j]>=frame.ma26.iloc[j])
                        rec[f'{label}_hist_up']=int(frame['hist'].iloc[j]>frame['hist'].iloc[j-1])
                        rec[f'{label}_dif_up']=int(frame['dif'].iloc[j]>frame['dif'].iloc[j-1])
                        recent=max(0,j-5)+np.argmin(frame.c.iloc[max(0,j-5):j+1])
                        prior=max(0,j-26)+np.argmin(frame.c.iloc[max(0,j-26):max(1,j-7)])
                        rec[f'{label}_divergence']=int(frame.c.iloc[recent]<frame.c.iloc[prior]*1.02 and frame['dif'].iloc[recent]>frame['dif'].iloc[prior])
                assert rec['w_bar_date']<=row.signal_date and rec['m_bar_date']<=row.signal_date
                assert p<0 or p+2<i
                output.append(rec)
            if (n+1)%1000==0:print(f'{n+1} stocks, {len(output):,} events, {time.time()-start:.1f}s',flush=True)
    features=pd.DataFrame(output).set_index('event_id').reindex(e.index)
    features.to_pickle(OUT/'features.pkl')
    print(f'{len(features)} rows x {len(features.columns)} features, {time.time()-start:.1f}s',flush=True)

def base_rules(e):
    base=(e.price>=3)&(e.amount20>=3e7)&(e.signal_return<.095)&(e.market20<0)
    shapes={'original':(e.drawdown>=.30)&(e.floor_days<=8)&(e.prior_from_floor<=.08)}
    for fw,pw in ((60,250),(120,250),(120,500),(250,500)):
        for age,rise in ((8,.08),(15,.12)):
            shapes[f'long_{fw}_{pw}_{age}']=(e[f'drop_{fw}_{pw}']>=.30)&(e[f'age_{fw}_{pw}']<=age)&(e[f'rise_{fw}_{pw}']<=rise)
    return {f'{shape}_market{market}':base&mask&(e.daily>=market)
            for (shape,mask),market in product(shapes.items(),(.015,.02,.025))}

def gates(e):
    f={'none':np.ones(len(e),bool)}
    for days,threshold in product((20,40,60,120),(1.,1.2,1.5)):
        f[f'bullavg{days}_ge{threshold}']=e[f'vol_mean_{days}']>=threshold
    f.update(bullshare40_ge50=e.vol_share_40>=.5,bullshare60_ge50=e.vol_share_60>=.5,
             down_volume_dry=e.down_volume_decay<=.8,divergence=e.divergence==1,
             near_divergence=e.near_divergence==1,window_divergence=e.window_divergence==1,
             macd_hist_turn=e.hist_turn==1,macd_dif_rising=e.dif_rising==1,
             weekly_hist_up=e.w_hist_up==1,weekly_divergence=e.w_divergence==1,
             weekly_momentum=e.w_mom_improving==1,monthly_momentum=e.m_mom_improving==1,
             monthly_above_ma6=e.m_above_ma6==1,monthly_hist_up=e.m_hist_up==1)
    f['volume_macd']=(e.vol_mean_40>=1)&((e.near_divergence==1)|(e.hist_turn==1))
    f['volume_weekly']=(e.vol_mean_40>=1)&(e.w_mom_improving==1)
    f['volume_monthly']=(e.vol_mean_40>=1)&(e.m_mom_improving==1)
    f['multi_agree']=(e.vol_mean_40>=1)&((e.near_divergence==1)|(e.w_hist_up==1)|(e.m_mom_improving==1))
    return f

def ranks(e):
    volume=np.clip(np.log(e.vol_mean_40.fillna(1)), -1,1).to_numpy()
    macdscore=e.near_divergence.to_numpy()+.5*e.hist_turn.to_numpy()
    weeks=e.w_hist_up.fillna(0).to_numpy()+.5*e.w_mom_improving.fillna(0).to_numpy()
    months=e.m_mom_improving.fillna(0).to_numpy()+.5*e.m_above_ma6.fillna(0).to_numpy()
    return dict(liquidity=e.amount20.to_numpy(),volume=volume,macd=macdscore,
                volume_macd=volume+macdscore,multi=volume+macdscore+.5*weeks+.5*months,
                weekly=weeks+.5*volume,monthly=months+.5*volume)

def prepare():
    e,d,c=ps.prepare();e=e.join(pd.read_pickle(OUT/'features.pkl'))
    ids=np.flatnonzero(e.complete30.eq(1));days=e.signal_index.to_numpy()+1
    rowsets={}
    for name,score in ranks(e).items():
        rows=ids[np.lexsort((e.code.to_numpy()[ids],-e.amount20.to_numpy()[ids],-score[ids],days[ids]))]
        rowsets[name]=dict(rows=rows,offsets=np.searchsorted(days[rows],np.arange(len(c['calendar'])+1)))
    return e,d,c,rowsets

def run_search():
    e,d,c,rowsets=prepare();started=time.time();rows=[];tables=[]
    # No re-optimization of 61 exits. Compare four holding periods and one
    # protective variant, preserving the user's original horizon request.
    exit_ids=[r.EXITS.index((h,0.,0.,0.)) for h in (5,10,20,30)]+[r.EXITS.index((20,0.,.07,0.))]
    base=base_rules(e);filters=gates(e)
    for (b,bm),(f,fm),rank in product(base.items(),filters.items(),rowsets):
        # Hard gates and ranking are examined separately first. Joint
        # volume/indicator gates are explicitly listed, not arbitrarily mined.
        if f!='none' and rank!='liquidity' and not (f in {
                'bullavg60_ge1.0','bullavg60_ge1.2','bullavg120_ge1.0',
                'weekly_momentum','monthly_momentum','volume_weekly','multi_agree'}
                and rank in {'volume','macd','multi'}):continue
        mask=np.asarray(bm&fm,dtype=bool)
        if mask.sum()<100:continue
        cc={**c,**rowsets[rank]}
        for k in exit_ids:
            zz=ps.simulate_port(mask,2,k,1.,10,**cc,cost=r.COST)
            if zz[-1]:continue
            a,dd,n,s,w,g,l=zz[:7]
            rows.append(dict(base=b,gate=f,rank=rank,exit=k))
            tables.append(np.column_stack((a,dd,n,s,w,g,l)))
    table=np.asarray(tables);np.savez_compressed(OUT/'search.npz',table=table)
    (OUT/'configs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'{len(rows)} portfolios; {time.time()-started:.1f}s',flush=True)
    for cutoff in (2024,2025,2026,2027):
        ny=cutoff-2022;a=table[:,:ny,0];dd=table[:,:ny,1];n=table[:,:ny,2]
        cagr=np.prod(1+a,axis=1)**(1/ny)-1
        # Favor sufficient observations and drawdown control rather than a
        # barely traded set with perfect accuracy.
        eligible=(n.min(1)>=8)&(n.sum(1)>=20*ny)&(a.min(1)>-.03)&(dd.min(1)>-.20)
        ids=np.flatnonzero(eligible)
        score=cagr+.5*a.min(1)+.5*dd.min(1)
        chosen=ids[np.argsort(-score[ids])][:15]
        picks=[dict(config=int(i),**rows[i],hold=r.EXITS[rows[i]['exit']][0],annual=table[i,:,0].tolist(),
                    drawdown=table[i,:,1].tolist(),n=table[i,:,2].tolist(),score=float(score[i])) for i in chosen]
        (OUT/f'picks_before_{cutoff}.json').write_text(json.dumps(picks,indent=2),encoding='utf-8')
        print(f'Before {cutoff}: {len(ids)} eligible',flush=True)
        for x in picks[:4]:print(json.dumps(x),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['build','search']);a=p.parse_args()
    build() if a.task=='build' else run_search()
