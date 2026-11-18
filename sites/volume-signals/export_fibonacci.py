"""Fibonacci ladder fills, price levels, cash returns and browser datasets."""
import json
import math
import gzip
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
import fibonacci_v7 as f

def main():
    report=json.loads((f.OUT/'report.json').read_text(encoding='utf-8'));report['audit']=json.loads((f.OUT/'audit.json').read_text(encoding='utf-8'))
    grid=pd.read_pickle(f.OUT/'evaluated.pkl');setups=pd.read_pickle(f.OUT/'setups.pkl').set_index('event_id')
    panel=pd.read_pickle(f.c.d.r.OUT/'panel.pkl');flags=pd.read_pickle(f.c.OUT/'status_maps.pkl');cal=pd.read_csv(f.c.d.r.OUT/'market.csv').date.to_numpy()
    rows={};aliases={};cache={};case_id=int(setups[(setups.code=='sz.300369')&setups.signal_date.eq(20260825)].index[0])
    def detail(event,mode,h,w,included,split):
        x=setups.loc[event];dates,raw,close,factor,pre,mi,*_=panel[x.code];i=int(x.stock_index)
        if x.code not in cache:
            st=flags[x.code];status=np.array([st.get(int(day),1) for day in dates],np.int8)
            rate=20 if x.code[3:6] in {'300','301','688'} else 10
            upper=raw[:,0]>=np.maximum(np.floor((pre*(100+rate)+50)/100),pre+1)
            locked=np.array([f.c.lower_locked(raw,pre,dates,st,x.code,z) for z in range(len(raw))])
            cache[x.code]=(raw,factor,mi,status,upper,locked)
        span=x.fib_high-x.fib_low;levels=np.array([x.fib_high-r*span for r in (.382,.5,.618)]+[x.fib_high-1.382*span if mode=='retracement' else x.fib_low+1.382*span])
        values=[];logs=[]
        for path in (0,1):
            v,log=f.simulate(*cache[x.code],i,levels,w,mode=='extension',h,path);values.append(v);logs.append(log)
        choice=1 if not np.isfinite(values[0][0]) or (np.isfinite(values[1][0]) and values[1][0]<values[0][0]) else 0
        v=values[choice];log=logs[choice];first=int(v[1]);last=int(v[2]);complete=(first>=0 and mi[first]+h-1<len(cal)) or (first<0 and mi[i]+20<len(cal))
        buys=[];sells=[];stop_levels=[]
        for z,price,q,side,leg,stop in log:
            z=int(z);adj=float(price*factor[z]/factor[i]/100)
            if side>0:
                buys.append(dict(date=int(dates[z]),raw_price=float(price/100),signal_price=adj,weight=float(w[int(leg)]),level=['0.382','0.5','0.618','1.382'][int(leg)],label='买'+str(len(buys)+1)))
                stop_levels.append(dict(date=int(dates[z]),price=float(stop*factor[z]/factor[i]/100)))
            else:sells.append(dict(date=int(dates[z]),raw_price=float(price/100),signal_price=adj,reason='full_hard15' if int(leg)==1 else 'expiry'))
        initial_ticks=[float((math.ceil(value/factor[i]-1e-7) if mode=='extension' and k==3 else math.floor(value/factor[i]+1e-7))/100) for k,value in enumerate(levels)]
        lines=[dict(price=value,label=label,start=int(dates[i]),color='#b18a31') for value,label in zip(initial_ticks,['0.382','0.5','0.618','1.382']) if value>0]
        # End-of-day weighted-cost stop segments, on the same chart price basis.
        stops={q['date']:q['price'] for q in stop_levels};ordered=sorted(stops)
        for j,date in enumerate(ordered):lines.append(dict(price=stops[date],label='15%止损',start=date,end=ordered[j+1] if j+1<len(ordered) else int(dates[last]) if last>=0 else int(dates[-1]),color='#bc4b76'))
        net=float(v[0]) if complete and np.isfinite(v[0]) else None
        row=dict(event_id=int(event),code=x.code,name=x['name'],signal_date=int(x.signal_date),signal_label='缩量/挂单',
            entry_date=int(dates[first]) if first>=0 else 0,exit_date=int(dates[last]) if last>=0 else 0,
            included=bool(included),split=split,horizon=h,weights=w.tolist(),net=net,spent=float(v[4]),stopped=bool(v[5]),
            status='未到期' if not complete else '未成交' if first<0 else '已退出' if np.isfinite(v[0]) else '退出未完成',
            path=choice,path_gap=float(abs(values[0][0]-values[1][0])) if np.isfinite(values[0][0]) and np.isfinite(values[1][0]) else None,
            floor_price=float(x.fib_low/factor[i]/100),peak_price=float(x.fib_high/factor[i]/100),levels=initial_ticks,
            buy_events=buys,sell_events=sells,price_lines=lines,horizons=[h],
            stage_events=[dict(date=int(x.floor_date),label='底部',color='#258b78'),dict(date=int(x.fib_high_date),label='第一波高点',color='#b87824')])
        row[f'exit_date_{h}']=row['exit_date'] if complete else 0
        return row
    for section in report['horizons']:
        h=section['horizon']
        for mode in f.MODES:
            for group in ('selected','equal4','equal3'):
                config=section['modes'][mode][group]
                if config is None:continue
                wi=config['weight_id'];w=np.array(config['weights']);key=f"{'common' if w[3]==0 else mode}:{h}:{wi}"
                aliases[f'{mode}:{h}:{group}']=key
                if key in rows:continue
                chosen=grid[(grid['mode']==mode)&(grid.horizon==h)&(grid.weight_id==wi)]
                records=[detail(int(x.event_id),mode,h,w,True,x.split) for x in chosen.itertuples()]
                if case_id not in chosen.event_id.values:records.append(detail(case_id,mode,h,w,False,'case'))
                rows[key]=records
                export=chosen.copy();export['weights']=' / '.join(f'{a:.4f}' for a in w)
                export.to_csv(DIST/f'fib-{key.replace(":","-")}.csv',index=False,encoding='utf-8-sig')
    payload=dict(report=report,rows=rows,aliases=aliases,case_id=case_id)
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    (DIST/'fibonacci-data.json.gz').write_bytes(gzip.compress(encoded,compresslevel=6,mtime=0))
    (DIST/'fibonacci-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (f.OUT/'case_browser.json').write_text(json.dumps({key:next(x for x in values if x['event_id']==case_id) for key,values in rows.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    print('fibonacci exported',len(rows),'datasets',sum(len(x) for x in rows.values()),'rows')

if __name__=='__main__':main()
