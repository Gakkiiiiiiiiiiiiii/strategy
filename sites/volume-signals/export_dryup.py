"""Export frozen low-base dry-volume study and actual signal windows."""
import json
import sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
import dryup_v5 as d


def main():
    report=json.loads((d.OUT/'report.json').read_text(encoding='utf-8'))
    report['audit']=json.loads((d.OUT/'audit.json').read_text(encoding='utf-8'))
    rows={}
    for key in d.PROFILES:
        frame=pd.read_csv(d.OUT/f'{key}_signals.csv')
        frame=frame[frame.entry_status.isin(['filled','upper_open'])].copy()
        for h in d.HORIZONS:
            selected=pd.read_csv(d.OUT/f'{key}_{h}_trades.csv')
            frame[f'included_{h}']=frame.event_id.isin(selected.event_id)
            cols=['event_id','code','name','signal_date','entry_date','buy_price','floor_date','peak_date',
                  'rally','pullback','rally_volume_ratio','dry_ratio','dry_vs20',
                  f'exit_date_{h}',f'exit_price_{h}',f'net_{h}',f'benchmark_{h}',f'excess_{h}',f'mae_{h}']
            selected[cols].to_csv(DIST/f'dryup-{key}-{h}.csv',index=False,encoding='utf-8-sig')
        columns=['event_id','code','name','signal_date','entry_date','entry_status','buy_price',
            'floor_date','peak_date','floor_price','peak_price','signal_return','rally','pullback','retrace',
            'rally_volume_ratio','wash_volume_ratio','tail_volume_ratio','dry_ratio','dry_vs20']
        columns += [f'{field}_{h}' for h in d.HORIZONS for field in ('net','status','exit_date','benchmark','included')]
        values=json.loads(frame[columns].to_json(orient='records'))
        for row in values:
            row['horizons']=list(d.HORIZONS);row['signal_label']='缩量'
            row['stage_events']=[dict(date=row['floor_date'],label='底部',color='#258b78'),
                                 dict(date=row['peak_date'],label='放量峰',color='#b87824')]
        rows[key]=values
    payload=dict(report=report,rows=rows)
    (DIST/'dryup-data.js').write_text('window.dryupResearch='+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n',encoding='utf-8')
    (DIST/'dryup-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for filename in ('index.html','research.html','multiframe.html','macd.html','stops.html'):
        path=DIST/filename;text=path.read_text(encoding='utf-8')
        if 'href="dryup.html"' not in text:text=text.replace('</nav>','<a href="dryup.html">◌ 低位极致缩量</a></nav>',1)
        path.write_text(text,encoding='utf-8')
    print('dryup exported', {k:len(v) for k,v in rows.items()})


if __name__=='__main__':main()
