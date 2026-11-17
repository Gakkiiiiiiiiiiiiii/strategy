"""Execution edge cases: gaps, T+1, price ticks, locked bars and share splits."""
import unittest
import numpy as np
from stoploss_v4 import staged_exit,tick_below,split_quantity


class StopExecutionTests(unittest.TestCase):
    def plan(self,tail,entry=(1078,1080,1060,1070),hold=4,factors=None,market=None,locked=None):
        raw=np.asarray([(1010,1100,1000,1100),entry,*tail],float);n=len(raw)
        return staged_exit(raw,np.ones(n) if factors is None else np.asarray(factors),
                           np.full(n,1000.),np.arange(n) if market is None else np.asarray(market),
                           np.zeros(n,bool) if locked is None else np.asarray(locked),0,hold)

    def test_half_then_full(self):
        p=self.plan([(1080,1090,1040,1070),(1040,1050,990,1000)])
        self.assertEqual([x['raw_price'] for x in p['legs']],[10.49,9.99])
        self.assertEqual([x['day'] for x in p['legs']],[2,3])

    def test_gap_below_both_sells_at_open(self):
        p=self.plan([(980,990,970,980)])
        self.assertEqual([x['raw_price'] for x in p['legs']],[9.8,9.8])
        self.assertTrue(all(x['reason']=='full_gap' for x in p['legs']))

    def test_two_crossings_same_bar(self):
        p=self.plan([(1080,1090,990,1010)])
        self.assertEqual([x['raw_price'] for x in p['legs']],[10.49,9.99])
        self.assertEqual([x['day'] for x in p['legs']],[2,2])

    def test_t1_latches_entry_day_breach(self):
        p=self.plan([(1090,1100,1080,1090)],entry=(1078,1080,980,1050))
        self.assertEqual(p['entry_breach'],2)
        self.assertEqual([x['raw_price'] for x in p['legs']],[10.9,10.9])
        self.assertTrue(all(x['day']==2 for x in p['legs']))

    def test_t1_half_even_if_recovered(self):
        p=self.plan([(1060,1070,1055,1065)],entry=(1078,1080,1040,1060),hold=2)
        self.assertEqual(p['legs'][0]['reason'],'half_queued')
        self.assertEqual(p['legs'][0]['raw_price'],10.6)
        self.assertEqual(p['legs'][1]['reason'],'expiry')

    def test_locked_limit_down_delays_execution(self):
        p=self.plan([(980,980,980,980),(960,970,950,960)],locked=[False,False,True,False])
        self.assertEqual(p['blocked_days'],1)
        self.assertEqual([x['day'] for x in p['legs']],[3,3])
        self.assertEqual([x['raw_price'] for x in p['legs']],[9.6,9.6])

    def test_equality_does_not_trigger(self):
        p=self.plan([(1080,1100,1050,1090)],hold=2)
        self.assertTrue(all(x['reason']=='expiry' for x in p['legs']))

    def test_adjustment_changes_raw_trigger(self):
        p=self.plan([(540,550,520,530)],hold=2,factors=[1.,1.,2.])
        self.assertEqual(p['legs'][0]['raw_price'],5.24)
        self.assertEqual(p['legs'][0]['price'],1048.)
        self.assertEqual(p['half_ref'],10.5)

    def test_suspension_past_expiry_uses_open(self):
        p=self.plan([(1070,1100,1060,1090)],hold=2,market=[0,1,5])
        self.assertTrue(all(x['reason']=='expiry_delayed' for x in p['legs']))
        self.assertEqual([x['raw_price'] for x in p['legs']],[10.7,10.7])

    def test_tick_and_share_rounding(self):
        self.assertEqual(tick_below(1050),1049)
        self.assertEqual(tick_below(1050.5),1050)
        self.assertEqual(split_quantity(300),(100,200))
        self.assertEqual(split_quantity(100),(0,100))
        self.assertEqual(split_quantity(1000),(500,500))


if __name__=='__main__':unittest.main()
