import unittest
import numpy as np
from dryup_v5 import executable_exit,scan


class ExitTests(unittest.TestCase):
    def test_flat_base_without_rally_is_not_a_signal(self):
        close=np.r_[np.full(360,2000.),np.full(101,1000.)]
        found=scan(close,np.full(len(close),1e7),np.full(len(close),1e8),close)
        self.assertEqual(len(found),0)

    def test_immature_120_not_assigned_a_return(self):
        raw=np.array([[100,110,90,105],[105,110,100,107]],float)
        value,reason=executable_exit(raw,np.ones(2),np.arange(2),np.array([100,105]),{},'sz.300369',0,119,np.array([20260101,20260102]))
        self.assertIsNone(value);self.assertEqual(reason,'immature')

    def test_suspended_maturity_uses_next_open(self):
        raw=np.array([[100,110,90,105],[110,120,105,119]],float)
        value,reason=executable_exit(raw,np.ones(2),np.array([0,6]),np.array([100,105]),{},'sh.600000',0,4,np.arange(20260101,20260109))
        self.assertEqual(reason,'complete');self.assertEqual(value,(1,110.,1.1,2))

    def test_locked_down_maturity_delays(self):
        raw=np.array([[100,105,95,100],[90,90,90,90],[86,95,85,94]],float)
        value,reason=executable_exit(raw,np.ones(3),np.array([0,4,5]),np.array([100,100,90]),{},'sh.600000',0,4,np.arange(20260101,20260107))
        self.assertEqual(reason,'complete');self.assertEqual(value,(2,86.,.86,1))

    def test_locked_at_snapshot_is_unresolved(self):
        raw=np.array([[100,105,95,100],[90,90,90,90]],float)
        value,reason=executable_exit(raw,np.ones(2),np.array([0,4]),np.array([100,100]),{},'sh.600000',0,4,np.arange(20260101,20260106))
        self.assertIsNone(value);self.assertEqual(reason,'unresolved')


if __name__=='__main__':unittest.main()
