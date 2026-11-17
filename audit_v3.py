"""Reconstruct fills and prefix-only indicator inputs independently of the search."""
import json
import numpy as np
import pandas as pd
import research_v2 as r
import research_v3 as v


def main():
    panel=pd.read_pickle(r.OUT/'panel.pkl')
    summary={}
    for filename,hold in [('selected_trades.csv',20),('macd_trades.csv',30)]:
        trades=pd.read_csv(v.OUT/filename);errors=[];weeks=0;months=0
        for x in trades.itertuples():
            dates,raw,close,factor,pre,mi,upper,locked=panel[x.code]
            i=np.searchsorted(dates,x.signal_date);j=np.searchsorted(dates,x.entry_date);end=np.searchsorted(dates,x.exit_date)
            assert dates[i]==x.signal_date and dates[j]==x.entry_date and dates[end]==x.exit_date
            assert mi[j]==mi[i]+1
            cap=np.floor(pre[j]*.98+1e-8);buy=raw[j,0] if raw[j,0]<=cap else cap
            assert raw[j,0]<upper[j] and (raw[j,0]<=cap or raw[j,2]<cap)
            assert not locked[end] and mi[end]-mi[j]+1>=hold
            sell=raw[end,3] if mi[end]-mi[j]+1==hold else raw[end,0]
            gross=sell*factor[end]/buy/factor[j]-1
            errors.append(abs(gross-x.gross))
            assert abs((1+gross)*(1-r.COST/2)/(1+r.COST/2)-1-x.net)<2e-7
            # Cut every series at the signal before aggregation or computing MACD.
            c=close[:i+1];dt=pd.to_datetime(dates[:i+1].astype(str),format='%Y%m%d')
            dif,dea,hist=v.macd(c)
            strict,near,_,p,q=v.divergence(c,dif,hist,len(c)-1)
            assert strict==x.divergence and near==x.near_divergence
            assert p<0 or p+2<i
            assert int(hist[-1]>hist[-2] and hist[-2]>=hist[-3])==x.hist_turn
            for freq,label in [('W-FRI','w'),('ME','m')]:
                frame=pd.Series(c,index=dt).resample(freq).last().dropna()
                frame=frame.loc[frame.index<=dt[-1]]
                assert int(frame.index[-1].strftime('%Y%m%d'))==getattr(x,label+'_bar_date')
                assert len(frame)==getattr(x,label+'_count')
                if len(frame)>=6:
                    assert int(frame.iloc[-1]/frame.iloc[-2]>frame.iloc[-2]/frame.iloc[-3])==getattr(x,label+'_mom_improving')
                    if label=='w':weeks+=1
                    else:months+=1
            # Recheck the volume ratio against the source bars exported for plotting.
            source=r.bt.ROOT/'sites/volume-signals/dist/ohlcv'/f'{x.code.replace(".","")}.js'
            rows=np.asarray(json.loads(source.read_text(encoding='utf-8').split('=',1)[1].rstrip(';\n'))['rows'],dtype=object)
            candles=rows[(rows[:,0]>=dates[i-40])&(rows[:,0]<x.signal_date)]
            assert len(candles)==40
            up=candles[:,4]>candles[:,1];down=candles[:,4]<candles[:,1]
            ratio=candles[up,5].astype(float).mean()/candles[down,5].astype(float).mean()
            assert abs(ratio-x.vol_mean_40)<1e-9
            assert abs(float(rows[rows[:,0]==x.signal_date][0,6])-x.signal_return)<1e-8
        assert max(errors)<2e-7
        summary[filename]=dict(trades=len(trades),hold=hold,max_gross_error=max(errors),
                              prefix_week_checks=weeks,prefix_month_checks=months,
                              prefix_macd_verified=True,chart_volume_and_daily_return_verified=True)
    (v.OUT/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
