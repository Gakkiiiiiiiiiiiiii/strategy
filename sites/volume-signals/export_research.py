"""Publish audited research metrics and full OHLCV around selected signals."""
import json
import shutil
import sys
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
import backtest as bt

def export():
    src=ROOT/'results/research_v2'
    report=json.loads((src/'final_report.json').read_text(encoding='utf-8'))
    panel=pd.read_pickle(src/'panel.pkl');calendar=np.load(src/'outcomes.npz')['calendar']
    alltrades=pd.read_csv(src/'final_all_signal_trades.csv')
    actual=pd.read_csv(src/'final_portfolio_trades.csv')
    executed=set(actual.event_id)
    independent=set(alltrades.event_id)
    # A funded portfolio may skip an earlier signal and buy a later one that
    # was suppressed by the independent-signal overlap rule. Preserve both.
    alltrades=pd.concat([alltrades,actual.loc[~actual.event_id.isin(independent)]],ignore_index=True)
    alltrades['name']=alltrades['name'].fillna('缓存无名称')
    shards=[{} for _ in range(32)];rows=[]
    _,updates=bt.updated_bars(20261104,20261116,{})
    with ExitStack() as stack:
        _,read=bt.market_files(Path('D:/tdx/vipdoc'),stack)
        for code,group in alltrades.groupby('code'):
            dates,raw,c,factor,*_=panel[code]
            base=np.frombuffer(read(code.replace('.','')),bt.BAR_DTYPE)
            volumes=dict(zip(base['date'],base['volume']))
            if code in updates:
                extra,_=updates[code];volumes.update(dict(zip(extra['date'],extra['volume'])))
            for x in group.itertuples():
                i=int(x.stock_index);ci=int(x.signal_index)
                floor=i-60+int(np.argmin(c[i-60:i]));peak=i-120+int(np.argmax(c[i-120:floor+1]))
                start=min(peak,max(0,i-60));end=np.searchsorted(dates,calendar[min(ci+30,len(calendar)-1)],side='right')
                bars=np.column_stack((dates[start:end],np.rint(raw[start:end]*factor[start:end,None]/factor[i]),
                                      [volumes[int(day)] for day in dates[start:end]])).astype(np.int64).tolist()
                assert all(b[2]>=max(b[1],b[4]) and b[3]<=min(b[1],b[4]) for b in bars)
                shards[int(code.split('.')[1])%32][f'{code}-{x.signal_date}']=bars
                row={k:getattr(x,k) for k in ('code','name','signal_date','entry_date','exit_date','net','gross','buy_price','drawdown','floor_days','prior_from_floor','signal_return','volume5','volume10','daily','market20','amount20','event_id')}
                row.update(executed=x.event_id in executed,independent=x.event_id in independent,
                           pit_high_date=int(dates[peak]),candle_source='research')
                row.update({f'exit_date_{h}':int(calendar[ci+h]) for h in (5,10,20,30)})
                rows.append(row)
    for i,s in enumerate(shards):
        target=DIST/'research-chart-data';target.mkdir(exist_ok=True)
        (target/f'{i}.js').write_text(f'window.researchCandleRanges[{i}]='+json.dumps(s,separators=(',',':'))+';\n',encoding='utf-8')
    equity=pd.read_csv(src/'final_equity.csv')
    payload=dict(report=report,rows=sorted(rows,key=lambda r:(-r['signal_date'],r['code'])),equity=equity.to_numpy().tolist())
    assert sum(x['executed'] for x in rows)==len(executed)==report['selected']['fills']
    assert sum(x['independent'] for x in rows)==report['selected']['all_signals']['n']
    (DIST/'research-data.js').write_text('window.researchData='+json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    for filename in ('final_portfolio_trades.csv','final_all_signal_trades.csv','final_yearly.csv','final_report.json'):
        shutil.copyfile(src/filename,DIST/filename)
    print(f'Exported {len(rows)} signals, {len(executed)} funded fills, {sum(map(len,shards))} OHLCV windows')

if __name__=='__main__':export()
