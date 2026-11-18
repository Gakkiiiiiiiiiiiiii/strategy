import unittest
import numpy as np
from confirm_v6 import hard_exit,confirmation

class ExecutionTests(unittest.TestCase):
    def run_exit(self,raw,mi=None,target=4,code='sz.300369',factor=None):
        raw=np.array(raw,float);n=len(raw);mi=np.arange(n) if mi is None else np.array(mi)
        factor=np.ones(n) if factor is None else np.array(factor)
        return hard_exit(raw,factor,mi,np.r_[100,raw[:-1,3]],20260101+mi,{},code,0,target,20260101+np.arange(20))
    def test_intraday_stop(self):
        x,s=self.run_exit([[100,101,95,100],[100,105,80,102]])
        self.assertEqual((s,x['price'],x['reason']),('complete',.85,'stop_intraday'))
    def test_gap_below_stop(self):
        x,s=self.run_exit([[100,101,95,100],[75,90,70,85]])
        self.assertEqual((x['price'],x['reason']),(.75,'stop_gap'))
    def test_t1_entry_breach_latched_even_after_rebound(self):
        x,s=self.run_exit([[100,101,80,100],[105,110,100,108]])
        self.assertEqual((x['z'],x['price'],x['reason']),(1,1.05,'stop_queued'))
    def test_locked_stop_waits_for_sellable_open(self):
        x,s=self.run_exit([[100,101,95,100],[80,80,80,80],[70,75,65,72]])
        self.assertEqual((x['z'],x['price'],x['reason']),(2,.70,'stop_queued'))
    def test_stop_is_adjusted_for_split(self):
        x,s=self.run_exit([[100,101,95,100],[50,52,43,50],[50,51,40,49]],factor=[1,2,2])
        self.assertEqual((x['z'],x['price']),(2,.42))
    def test_early_stop_does_not_fill_immature_horizon(self):
        raw=np.array([[100,101,95,100],[80,90,75,85]],float)
        x,s=hard_exit(raw,np.ones(2),np.arange(2),np.array([100,100]),np.array([20260101,20260102]),{},'sz.300369',0,119,np.array([20260101,20260102]))
        self.assertIsNone(x);self.assertEqual(s,'immature')
    def test_first_confirm_not_best_later_bar(self):
        raw=np.tile([100,110,90,100],(25,1)).astype(float);vol=np.ones(25)
        raw[13,3]=105;vol[13]=2;raw[15,3]=108;vol[15]=5
        self.assertEqual(confirmation(raw,raw[:,3],vol,np.arange(25),12,0),(13,'confirmed'))
    def test_floor_break_cancels_before_later_confirm(self):
        raw=np.tile([100,110,90,100],(25,1)).astype(float);vol=np.ones(25)
        raw[13,3]=99;raw[15,3]=108;vol[15]=5
        self.assertEqual(confirmation(raw,raw[:,3],vol,np.arange(25),12,0),(None,'floor_broken'))

if __name__=='__main__':unittest.main()
