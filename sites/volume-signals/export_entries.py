"""Publish confirmation/stop controls without changing the archived v5 study."""
import json
import gzip
import sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
import confirm_v6 as c

def main():
    report=json.loads((c.OUT/'report.json').read_text(encoding='utf-8'))
    report['audit']=json.loads((c.OUT/'audit.json').read_text(encoding='utf-8'));rows={}
    for key in c.VARIANTS:
        frame=pd.read_csv(c.OUT/f'{key}_signals.csv');frame=frame[frame.entry_status.isin(['filled','upper_open'])].copy()
        for h in c.HORIZONS:
            selected=pd.read_csv(c.OUT/f'{key}_{h}_trades.csv');frame[f'included_{h}']=frame.event_id.isin(selected.event_id)
            selected.to_csv(DIST/f'entries-{key}-{h}.csv',index=False,encoding='utf-8-sig')
        columns=['event_id','code','name','signal_date','dry_date','entry_date','entry_status','buy_price',
            'floor_date','peak_date','floor_price','peak_price','signal_return','rally','pullback','retrace',
            'rally_volume_ratio','wash_volume_ratio','tail_volume_ratio','dry_ratio','dry_vs20']
        if key!='direct':columns+=['confirm_wait','volume5','volume10','stop_price','stop_signal_price']
        columns+=[f'{field}_{h}' for h in c.HORIZONS for field in ('net','status','exit_date','benchmark','included','reason')]
        if key!='direct':columns+=[f'sale_signal_price_{h}' for h in c.HORIZONS]
        values=json.loads(frame[columns].to_json(orient='records'))
        for row in values:
            row['horizons']=list(c.HORIZONS);row['signal_label']='缩量' if key=='direct' else '放量确认'
            row['stage_events']=[dict(date=row['floor_date'],label='底部',color='#258b78'),dict(date=row['peak_date'],label='放量峰',color='#b87824')]
            if key!='direct':row['stage_events'].append(dict(date=row['dry_date'],label='极致缩量',color='#6655c9'))
            row['stop_enabled']=key=='stop';row['stop_full_price']=row.get('stop_signal_price');row['stop_half_price']=None
        rows[key]=values
    payload=json.dumps(dict(report=report,rows=rows),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    (DIST/'entries-data.json.gz').write_bytes(gzip.compress(payload,compresslevel=6,mtime=0))
    (DIST/'entries-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for file in DIST.glob('*.html'):
        text=file.read_text(encoding='utf-8')
        if '<nav' in text and 'href="entries.html"' not in text:text=text.replace('</nav>','<a href="entries.html">↗ 放量确认 + 止损</a></nav>',1)
        if '<nav' in text and 'href="fibonacci.html"' not in text:text=text.replace('</nav>','<a href="fibonacci.html">⌁ 斐波那契分批</a></nav>',1)
        file.write_text(text,encoding='utf-8')
    print('entries exported', {k:len(v) for k,v in rows.items()})

if __name__=='__main__':main()
