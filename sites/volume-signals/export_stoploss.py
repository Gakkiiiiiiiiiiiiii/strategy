"""Export the fixed-rule stop-loss comparison, actual cash flows and chart markers."""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
import research_v2 as r


def export():
    src=ROOT/'results/research_v4';report=json.loads((src/'report.json').read_text(encoding='utf-8'))
    report['audit']=json.loads((src/'audit.json').read_text(encoding='utf-8'))
    panel=pd.read_pickle(r.OUT/'panel.pkl');cal=np.load(r.OUT/'outcomes.npz')['calendar']
    rows={};curves={}
    for strategy in ('balanced','macd'):
        rows[strategy]={};curves[strategy]={}
        for variant in ('original','uncapped','original_stop','uncapped_stop'):
            stem=strategy+'_'+variant;trades=pd.read_csv(src/f'{stem}_trades.csv');records=[]
            for x in trades.to_dict('records'):
                x['name']=x['name'] if isinstance(x['name'],str) else '缓存无名称'
                x['actions']=json.loads(x['actions']);dates,raw,close,factor,*_=panel[x['code']]
                i=np.searchsorted(dates,x['signal_date']);market_i=np.searchsorted(cal,x['signal_date'])
                x['sell_events']=[{**a,'signal_price':a['raw_price']*factor[np.searchsorted(dates,a['date'])]/factor[i]} for a in x['actions']]
                x['stop_enabled']=variant.endswith('_stop');x['pit_high_date']=0
                x.update({f'exit_date_{h}':int(cal[market_i+h]) for h in (5,10,20,30)})
                records.append(x)
            rows[strategy][variant]=sorted(records,key=lambda x:(-x['signal_date'],x['code']))
            curves[strategy][variant]=pd.read_csv(src/f'{stem}_equity.csv').to_numpy().tolist()
            shutil.copyfile(src/f'{stem}_trades.csv',DIST/f'stops-{stem}.csv')
    (DIST/'stops-data.js').write_text('window.stopResearch='+json.dumps(dict(report=report,rows=rows,curves=curves),ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n',encoding='utf-8')
    (DIST/'stops-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename in ('index.html','research.html','multiframe.html','macd.html'):
        path=DIST/filename;text=path.read_text(encoding='utf-8')
        if 'href="stops.html"' not in text:
            text=text.replace('<nav aria-label="页面导航">','<nav aria-label="页面导航"><a href="stops.html">↘ 分级止损对照</a>',1)
            path.write_text(text,encoding='utf-8')
    print(json.dumps({strategy:{variant:len(a) for variant,a in data.items()} for strategy,data in rows.items()}))


if __name__=='__main__':export()
