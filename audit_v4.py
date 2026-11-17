"""Check actual sale ticks, T+1, share conservation and net proceeds from raw bars."""
import json
import numpy as np
import pandas as pd
import research_v2 as r
import stoploss_v4 as s


def main():
    panel=pd.read_pickle(r.OUT/'panel.pkl');summary={}
    for path in sorted(s.OUT.glob('*_stop_trades.csv')):
        t=pd.read_csv(path);errors=[];sale_count=0
        for x in t.itertuples():
            dates,raw,close,factor,pre,mi,upper,locked=panel[x.code]
            i=np.searchsorted(dates,x.signal_date);j=np.searchsorted(dates,x.entry_date)
            mid=(pre[i]+raw[i,3])/2*factor[i];full=pre[i]*factor[i]
            assert abs(x.stop_half_price-mid/factor[i]/100)<1e-8
            assert abs(x.stop_full_price-full/factor[i]/100)<1e-8
            cap=np.floor(pre[j]*.98+1e-8)
            buy=raw[j,0] if raw[j,0]<=cap else cap
            assert raw[j,0]<upper[j] and (raw[j,0]<=cap or raw[j,2]<cap)
            assert abs(buy/100-x.buy_price)<1e-8
            actions=json.loads(x.actions);quantity=0;proceeds=0.;lastday=0
            for a in actions:
                z=np.searchsorted(dates,a['date']);sale_count+=1;quantity+=a['quantity']
                assert dates[z]>dates[j] and not locked[z] and a['date']>=lastday
                lastday=a['date'];price=round(a['raw_price']*100)
                assert abs(price-a['raw_price']*100)<1e-6
                assert raw[z,2]-1e-6<=price<=raw[z,1]+1e-6
                assert a['quantity']%100==0
                reason=a['reason'];line=mid if reason.startswith('half_') else full
                if reason.endswith('_intraday'):
                    assert price==s.tick_below(line/factor[z])
                    assert raw[z,0]>price and raw[z,2]<=price
                elif reason.endswith('_gap'):
                    assert price==raw[z,0] and price<=s.tick_below(line/factor[z])
                elif reason.endswith('_queued'):
                    assert price==raw[z,0]
                    trigger=[k for k in range(j,z) if raw[k,2]<=s.tick_below(line/factor[k])]
                    assert trigger and any(k==j or locked[k] for k in trigger)
                elif reason=='expiry':assert mi[z]-mi[j]+1==x.hold and price==raw[z,3]
                elif reason=='expiry_delayed':assert mi[z]-mi[j]+1>x.hold and price==raw[z,0]
                else:raise ValueError(reason)
                assert abs(a['adjusted_price']-price*factor[z])<1e-7
                proceeds+=a['quantity']*price*factor[z]
            assert quantity==x.quantity and lastday==x.exit_date
            net=proceeds/(x.quantity*buy*factor[j])*(1-r.COST/2)/(1+r.COST/2)-1
            errors.append(abs(net-x.net))
            assert abs(mid/(buy*factor[j])-1-x.stop_half_distance)<1e-10
            assert abs(full/(buy*factor[j])-1-x.stop_full_distance)<1e-10
        assert max(errors)<1e-10
        summary[path.name]=dict(trades=len(t),sales=sale_count,max_net_error=max(errors),
                                t_plus_one=True,raw_ticks_verified=True,share_conservation=True)
    (s.OUT/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
