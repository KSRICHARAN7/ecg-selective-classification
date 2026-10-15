import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_baseline import threshold_at_sensitivity,metrics


class TrainingHelpers(unittest.TestCase):
    def test_operating_threshold_preserves_required_positive_fraction(self):
        y=np.r_[np.ones(10),np.zeros(10)];p=np.r_[np.arange(10)/10,np.ones(10)*.3]
        threshold=threshold_at_sensitivity(y,p,.9)
        self.assertAlmostEqual(threshold,.1)
        self.assertGreaterEqual(metrics(y,p,threshold)['sensitivity'],.9)

    def test_threshold_uses_positive_cases_and_inclusive_ties(self):
        y=np.array([1,1,1,0]);p=np.array([.8,.8,.2,.99])
        self.assertEqual(threshold_at_sensitivity(y,p,2/3),.8)


if __name__=='__main__':unittest.main()
