import sys,unittest,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_tools import AblationRanker,score_ablation,has_native_artifact,group_masks,class_gate,flagging_metrics
class RobustnessTests(unittest.TestCase):
 def test_ablation_dimensions(self):
  expected={'confidence_no_class':2,'quality_no_class':14,'combined_no_class':16,'combined_no_amplitude':13,'combined_no_spectral':13,'combined_no_temporal':13,'combined_no_kurtosis':15}
  for name,n in expected.items():self.assertEqual(AblationRanker(name,.56).design(np.array([.1,.8]),np.ones((2,7)),fit=True).shape,(2,n))
 def test_ablation_json_roundtrip(self):
  rng=np.random.default_rng(4);p=rng.uniform(size=60);q=rng.normal(size=(60,7));q[0,2]=np.nan;y=np.tile([0,1],30)
  with tempfile.TemporaryDirectory()as t:
   path=Path(t)/'ranker.json'
   for name in ['confidence_no_class','quality_no_class','combined_no_class','combined_no_amplitude','combined_no_spectral','combined_no_temporal','combined_no_kurtosis']:
    r=AblationRanker(name,.56).fit(p,q,y,np.ones(60));r.save(path);np.testing.assert_allclose(r.score(p,q),score_ablation(path,p,q),atol=1e-12,rtol=0)
 def test_quality_without_class_has_no_probability_feature(self):
  r=AblationRanker('quality_no_class',.56);q=np.arange(21).reshape(3,7);a=r.design(np.array([.1,.2,.8]),q,fit=True);b=r.design(np.array([.9,.9,.1]),q);np.testing.assert_array_equal(a,b)
 def test_disease_metadata_is_not_noise(self):
  self.assertFalse(has_native_artifact({'extra_beats':'3','pacemaker':'True','static_noise':' '}));self.assertTrue(has_native_artifact({'static_noise':', V1'}))
 def test_earliest_tie_and_mixed_rules(self):
  common={'patient_id':'p','age':'40','sex':'0','recording_date':'1990-01-01 01:00:00','scp_codes':"{'AFIB':0}"};rows=[dict(common,ecg_id='2'),dict(common,ecg_id='10',scp_codes="{'AFIB':0,'SR':0}")];m=group_masks(['2','10'],rows,{'AFIB','SR'});np.testing.assert_array_equal(m['one_earliest_per_participant'],[False,True]);np.testing.assert_array_equal(m['exclude_mixed_rhythms'],[True,False])
 def test_class_gate_counts_people_not_variants(self):
  d=dict(y=np.tile([0,1],40),participant=np.repeat('one_person',80));g=class_gate(d,np.ones(80,bool));self.assertEqual(g['positive_participants'],1);self.assertFalse(g['numeric_subgroup_metrics_permitted'])
 def test_true_positive_deferral_differs_from_retained_sensitivity(self):
  d=dict(y=np.array([1,1,0,0]),pred=np.array([1,0,1,0]),p=np.array([.95,.01,.9,.01]));m=flagging_metrics(d,[0,1,0,1]);self.assertEqual(m['true_positives_deferred'],1);self.assertEqual(m['AF_records_deferred'],1);self.assertEqual(m['false_negatives_retained'],1);self.assertEqual(m['retained_sensitivity'],0);self.assertEqual(m['error_capture_fraction'],.5)
 def test_high_confidence_errors_count(self):
  d=dict(y=np.array([1,1,0,0]),pred=np.array([1,0,1,0]),p=np.array([.95,.01,.9,.01]));m=flagging_metrics(d,[0,1,0,1]);self.assertEqual(m['high_confidence_errors'],2);self.assertEqual(m['high_confidence_errors_flagged'],1)
if __name__=='__main__':unittest.main()
