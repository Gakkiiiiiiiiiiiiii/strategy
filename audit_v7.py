"""Independent cash and T+1 accounting audit on selected Fibonacci ladders."""
import json
import numpy as np
import pandas as pd
import fibonacci_v7 as f

def main():
    report=json.loads((f.OUT/'report.json').read_text(encoding='utf-8'));grid=pd.read_pickle(f.OUT/'evaluated.pkl')
    setups=pd.read_pickle(f.OUT/'setups.pkl').set_index('event_id');panel=pd.read_pickle(f.c.d.r.OUT/'panel.pkl');flags=pd.read_pickle(f.c.OUT/'status_maps.pkl')
    cache={};checked=0;max_error=0.;prefixes=0
    for selected in report['selected']:
        # Selection is entirely determined by the training sample.
        candidates=[r for r in report['grid'] if r['mode']==selected['mode'] and r['horizon']==selected['horizon'] and sum(v>0 for v in r['weights'])>=2 and r['train']['filled']>=100 and r['train']['fill_rate']>=.5]
        assert selected['train']['mean']==max(r['train']['mean'] for r in candidates)
        chosen=grid[(grid['mode']==selected['mode'])&(grid.horizon==selected['horizon'])&(grid.weight_id==selected['weight_id'])]
        for row in chosen.itertuples():
            x=setups.loc[row.event_id];dates,raw,close,factor,pre,mi,*_=panel[row.code]
            if row.code not in cache:
                st=flags[row.code];statuses=np.array([st.get(int(day),1) for day in dates],np.int8)
                rate=20 if row.code[3:6] in {'300','301','688'} else 10
                upper=raw[:,0]>=np.maximum(np.floor((pre*(100+rate)+50)/100),pre+1)
                locked=np.array([f.c.lower_locked(raw,pre,dates,st,row.code,z) for z in range(len(raw))])
                cache[row.code]=(raw,factor,mi,statuses,upper,locked)
            span=x.fib_high-x.fib_low;levels=np.array([x.fib_high-v*span for v in (.382,.5,.618)]+[x.fib_high-1.382*span if row.mode=='retracement' else x.fib_low+1.382*span])
            v,log=f.simulate(*cache[row.code],int(x.stock_index),levels,np.array(selected['weights']),row.mode=='extension',row.horizon,row.path)
            cash=1.;buys=[];sold=0.;seen=set()
            for z,price,units,side,leg,stop in log:
                z=int(z);assert z>x.stock_index
                assert raw[z,2]-1e-7<=price<=raw[z,1]+1e-7
                if side>0:
                    assert mi[z]<=x.signal_index+20;assert int(leg) not in seen;seen.add(int(leg))
                    spend=units*price*factor[z]*(1+f.c.d.r.COST/2)
                    assert abs(spend-selected['weights'][int(leg)])<1e-10
                    cash-=spend;buys.append((z,units))
                else:
                    eligible=sum(q for day,q in buys if day<z)-sold
                    assert units<=eligible+1e-10
                    cash+=units*price*factor[z]*(1-f.c.d.r.COST/2);sold+=units
            assert abs(sum(q for _,q in buys)-sold)<1e-10
            error=abs(cash-1-row.net);max_error=max(max_error,error);assert error<1e-10
            if row.split=='train':assert row.observation_end<=20241231 and row.exit_date<=20241231
            if row.split=='validation':assert row.setup_date>=20250101
            if checked%100==0 and len(log):
                end=int(log[-1,0])+1;truncated=tuple(a[:end] for a in cache[row.code])
                prefix,_=f.simulate(*truncated,int(x.stock_index),levels,np.array(selected['weights']),row.mode=='extension',row.horizon,row.path)
                assert abs(prefix[0]-row.net)<1e-10;prefixes+=1
            checked+=1
    result=dict(selected_ladders=checked,max_cash_error=max_error,execution_prefix_checks=prefixes,
        t1_no_same_day_share_sales=True,training_only_selection=True,tests=9,
        caveat='Checks establish accounting under the stated daily path assumptions, not actual market fills.')
    (f.OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':main()
