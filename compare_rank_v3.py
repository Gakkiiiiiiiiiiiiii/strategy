"""Separate the effects of MACD/volume ranking from the 30-day holding period."""
import json
import numpy as np
import research_v3 as v
import research_v2 as r
import portfolio_search as ps
import finalize_v2 as f

e,d,c,order=v.prepare()
mask=np.asarray(v.base_rules(e)['original_market0.02'],bool)
results=[]
for hold,rank in [(20,'liquidity'),(20,'volume_macd'),(30,'liquidity'),(30,'volume'),(30,'macd'),(30,'volume_macd')]:
    k=r.EXITS.index((hold,0.,0.,0.))
    z=ps.simulate_port(mask,2,k,1.,10,**{**c,**order[rank]},cost=r.COST)
    s=dict(hold=hold,rank=rank,**f.summarize(z,c['calendar']))
    results.append(s)
    print(json.dumps({k:s[k] for k in ('hold','rank','cagr','max_drawdown','fills','trade_stats')},ensure_ascii=False),flush=True)
(v.OUT/'ranking_comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
