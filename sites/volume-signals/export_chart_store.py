"""One full-history OHLCV store per stock, shared by all signal pages."""
import json
import sys
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
import backtest as bt

def export():
    codes=set(pd.read_csv(ROOT/'results/signals.csv').code)
    for src in ('research_v2/final_all_signal_trades.csv','research_v2/final_portfolio_trades.csv',
                'research_v3/selected_trades.csv','research_v3/all_signal_trades.csv',
                'research_v3/macd_trades.csv','research_v3/macd_all_signals.csv'):
        path=ROOT/'results'/src
        if path.exists():codes.update(pd.read_csv(path).code)
    panel=pd.read_pickle(ROOT/'results/research_v2/panel.pkl')
    _,updates=bt.updated_bars(20261104,20261116,{})
    followup=json.loads((HERE/'bear-followup.json').read_text(encoding='utf-8'))
    target=DIST/'ohlcv';target.mkdir(exist_ok=True)
    total=0
    with ExitStack() as stack:
        _,read=bt.market_files(Path('D:/tdx/vipdoc'),stack)
        for code in sorted(codes):
            dates,raw,c,factor,pre,mi,*_=panel[code]
            b=np.frombuffer(read(code.replace('.','')),bt.BAR_DTYPE)
            vols=dict(zip(b['date'],b['volume']))
            if code in updates:
                extra,_=updates[code];vols.update(dict(zip(extra['date'],extra['volume'])))
            rows=[[int(day),*[int(x) for x in raw[j]],int(vols[int(day)]),
                   round(float(raw[j,3]/pre[j]-1),10) if j else None,round(float(factor[j]),12)]
                  for j,day in enumerate(dates)]
            base_end=int(dates[-1])
            if code==bt.BEAR and followup['date']>dates[-1]:
                ohlc=[round(followup[k]*100) for k in ('open','high','low','close')]
                rows.append([followup['date'],*ohlc,int(followup['volume_shares']),
                             round(ohlc[-1]/raw[-1,3]-1,10),float(factor[-1])])
            assert all(x[2]>=max(x[1],x[4]) and x[3]<=min(x[1],x[4]) for x in rows)
            obj=dict(code=code,asof=int(rows[-1][0]),market_asof=20261116,rows=rows)
            text='window.fullCandleData['+json.dumps(code)+']='+json.dumps(obj,separators=(',',':'),allow_nan=False)+';\n'
            (target/(code.replace('.','')+'.js')).write_text(text,encoding='utf-8')
            total+=len(rows)
    print(f'Full history: {len(codes)} stocks, {total:,} daily bars; daily pct and adjustment factors included')

if __name__=='__main__':export()
