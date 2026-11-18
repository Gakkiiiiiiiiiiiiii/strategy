import unittest
import numpy as np
from fibonacci_v7 import simulate

class LadderTests(unittest.TestCase):
    def run_ladder(self,changes,levels=(90,80,70,60),weights=(.5,.5,0,0),extension=False,path=0,locked=None,upper=None):
        raw=np.tile([100,100,100,100],(40,1)).astype(float)
        for z,bar in changes.items():raw[z]=bar
        n=len(raw);locks=np.zeros(n,dtype=bool) if locked is None else locked;ups=np.zeros(n,dtype=bool) if upper is None else upper
        return simulate(raw,np.ones(n),np.arange(n),np.zeros(n,dtype=np.int8),ups,locks,10,np.array(levels,float),np.array(weights,float),extension,5,path,0.)
    def test_empty_orders_keep_entire_cash(self):
        v,log=self.run_ladder({});self.assertEqual(v[0],0);self.assertEqual(v[4],0);self.assertEqual(len(log),0)
    def test_gap_limit_fills_at_open_without_backdating(self):
        v,log=self.run_ladder({10:[70,70,70,70],11:[75,100,75,100]})
        self.assertEqual(v[1],11);self.assertEqual(v[3],3);self.assertTrue(np.all(log[:2,1]==75))
        self.assertAlmostEqual(v[0],100/75-1)
    def test_ladder_budget_and_weighted_returns(self):
        v,log=self.run_ladder({11:[100,100,85,95],12:[80,100,80,100]})
        self.assertEqual(v[4],1);self.assertAlmostEqual(v[0],(.5/90+.5/80)*100-1)
    def test_gap_stop_cancels_deeper_order_before_buy(self):
        v,log=self.run_ladder({11:[100,100,85,95],12:[70,100,70,100]})
        self.assertEqual(v[3],1);self.assertEqual(v[4],.5);self.assertEqual(v[5],1)
        self.assertAlmostEqual(v[0],.5+.5/90*70-1)
    def test_t1_new_leg_queues_while_old_shares_sell(self):
        v,log=self.run_ladder({11:[100,100,85,95],12:[85,85,70,75],13:[65,100,65,100]})
        sales=log[log[:,3]<0];self.assertEqual(len(sales),2);self.assertEqual(list(sales[:,0]),[12,13])
        self.assertEqual(list(sales[:,1]),[72,65])
    def test_extension_requires_base_position(self):
        v,log=self.run_ladder({11:[100,130,100,125]},levels=(90,80,70,120),weights=(.25,.25,.25,.25),extension=True)
        self.assertEqual(v[4],0);self.assertEqual(len(log),0)
    def test_two_daily_paths_expose_fill_order_ambiguity(self):
        kwargs=dict(changes={11:[100,130,80,110]},levels=(90,80,70,120),weights=(.25,.25,.25,.25),extension=True)
        a,_=self.run_ladder(**kwargs,path=0);b,_=self.run_ladder(**kwargs,path=1)
        self.assertNotEqual(a[3],b[3]);self.assertNotEqual(a[0],b[0])
    def test_locked_stop_remains_pending(self):
        lock=np.zeros(40,dtype=bool);lock[12]=True
        v,log=self.run_ladder({11:[100,100,85,95],12:[70,70,70,70],13:[60,100,60,100]},locked=lock)
        self.assertEqual(v[2],13);self.assertEqual(log[-1,1],60)
    def test_limit_up_open_skips_new_buys(self):
        upper=np.zeros(40,dtype=bool);upper[11]=True
        v,log=self.run_ladder({11:[75,75,75,75]},upper=upper)
        self.assertEqual(v[4],0)

if __name__=='__main__':unittest.main()
