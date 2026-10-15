import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_evaluation import selection_weights,metrics,bootstrap_primary
class EvaluationTests(unittest.TestCase):
 def test_ceil_and_lexical_tie(self):
  np.testing.assert_array_equal(selection_weights([.1,.1,.1],['2','10','3'],.34),[1,1,0])
 def test_multiplicity_equals_expanded_ranking(self):
  s=np.array([.2,.1,.1,.4]);ids=np.array(['5','8','7','3']);w=np.array([3,2,1,0]);k=4
  expanded=np.repeat(np.arange(4),w);order=np.lexsort((ids[expanded],s[expanded]));expected=np.bincount(expanded[order][:k],minlength=4)
  np.testing.assert_array_equal(selection_weights(s,ids,.6,w),expected)
 def test_absent_class_undefined(self):
  self.assertIsNone(metrics([1,0],[0,1],[1,0])['balanced_selective_error'])
 def test_retention_vs_sensitivity(self):
  m=metrics([1,1,0,0],[1,0,1,0],[1,0,1,1]);self.assertEqual(m['positive_retention'],.5);self.assertEqual(m['retained_sensitivity'],1);self.assertEqual(m['retained_true_positive_fraction_of_all_positives'],.5);self.assertEqual(m['balanced_selective_error'],.25)
 def test_invalid_inputs(self):
  for s,ids,w in [([np.nan],['a'],[1]),([1,2],['a','a'],[1,1]),([1],['a'],[.5])]:
   with self.assertRaises(ValueError):selection_weights(s,ids,.9,w)
 def test_full_coverage(self):
  np.testing.assert_array_equal(selection_weights([3,2,1],['a','b','c'],1,[0,3,2]),[0,3,2])
 def test_bootstrap_equal_methods_exact_zero(self):
  c=dict(ids=np.arange(20).astype(str),participant=np.repeat(np.arange(10),2),y=np.tile([0,1],10),pred=np.tile([1,1],10),combined=np.arange(20),confidence_adjusted=np.arange(20))
  summary,effects,_=bootstrap_primary([c]*4,30,3);np.testing.assert_array_equal(effects,0);self.assertEqual(summary['percentile95_CI'],[0,0])
 def test_bootstrap_repeatable_and_paired(self):
  c=dict(ids=np.arange(20).astype(str),participant=np.repeat(np.arange(10),2),y=np.tile([0,1],10),pred=np.tile([1,1],10),combined=np.arange(20),confidence_adjusted=-np.arange(20))
  a=bootstrap_primary([c]*4,30,3);b=bootstrap_primary([c]*4,30,3);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2][:,0],a[2][:,3])

class FinalNoiseNumerics(unittest.TestCase):
 def test_float32_input_snr_reference_matches_generator(self):
  from run_final_experiment import FinalNoise
  bank=object.__new__(FinalNoise)
  rng=np.random.default_rng(14);bank.bank={'bw':rng.normal(size=(2,58,1000))}
  signal=rng.normal(.323,.083,size=1000).astype('float32')
  _,audit=bank.pair(signal,'42','bw',24)
  self.assertLess(abs(audit['measured_snr_db']-24),1e-10)

if __name__=='__main__':unittest.main()
