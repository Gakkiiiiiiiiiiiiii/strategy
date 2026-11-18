import unittest
import numpy as np
from macd_leaders_v9 import bottom,eligible


class LongDeclineTests(unittest.TestCase):
    def test_persistent_decline_near_floor_passes(self):
        c=np.linspace(120.,50.,160)
        self.assertTrue(eligible(bottom(c,len(c)),True))

    def test_flat_then_recent_crash_is_not_long_decline(self):
        c=np.r_[np.full(150,100.),np.linspace(100.,60.,10)]
        b=bottom(c,len(c))
        self.assertTrue(eligible(b))
        self.assertEqual(b['decline_days'],9)
        self.assertFalse(eligible(b,True))

    def test_already_far_above_floor_fails_core(self):
        c=np.linspace(120.,60.,160);c[-1]=70.
        self.assertFalse(eligible(bottom(c,len(c))))

    def test_old_bottom_fails_core_even_without_big_rebound(self):
        c=np.r_[np.linspace(120.,60.,131),np.full(29,62.)]
        b=bottom(c,len(c))
        self.assertLess(b['prior_from_floor'],.08)
        self.assertFalse(eligible(b))


if __name__=='__main__':unittest.main()
