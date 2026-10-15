import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reliability_methods import ErrorRanker,raw_confidence,weighted_quantile,policy_weights,score_saved
from run_reliability_development import PolicyValidationNoise


class ReliabilityMethodChecks(unittest.TestCase):
    def sample(self):
        generator=np.random.default_rng(53);p=generator.uniform(.02,.98,150);quality=generator.normal(size=(150,7));error=(np.arange(150)%3==0).astype(int)
        return p,quality,error,np.full(150,1/9)

    def test_hierarchical_policy_mixture(self):
        condition=['clean']+[f'{family}_{snr}'for family in ['bw','ma']for snr in [24,12,6,0]]
        w=policy_weights(condition)
        self.assertAlmostEqual(w.sum(),1);self.assertAlmostEqual(w[0],1/3);self.assertAlmostEqual(w[1:5].sum(),1/3);self.assertAlmostEqual(w[5:].sum(),1/3)

    def test_weighted_threshold_and_inclusive_ties(self):
        self.assertEqual(weighted_quantile([0,1,2],[8,1,1],.8),0)
        self.assertEqual(weighted_quantile([0,1,2],[8,1,1],.9),1)
        self.assertEqual(weighted_quantile([.2,.2,.9],[1,1,1],.5),.2)

    def test_raw_confidence_reference(self):
        self.assertTrue(np.allclose(raw_confidence([0,.2,.5,.8,1]),[0,.2,.5,.2,0]))

    def test_all_missing_feature_and_all_seven_indicators(self):
        p,q,error,w=self.sample();q[:,1]=np.nan;q[::4,2]=np.nan
        model=ErrorRanker('combined',.56).fit(p,q,error,w)
        self.assertEqual(model.medians[1],0);self.assertEqual(len(model.feature_names),17)
        self.assertTrue(np.isfinite(model.score(p,q)).all())

    def test_saved_json_reproduces_scores(self):
        p,q,error,w=self.sample()
        for kind in ['confidence_adjusted','quality_only','combined']:
            model=ErrorRanker(kind,.56).fit(p,q,error,w)
            with tempfile.TemporaryDirectory()as directory:
                path=Path(directory)/'model.json';model.save(path)
                self.assertTrue(np.allclose(score_saved(path,p,q),model.score(p,q),rtol=0,atol=1e-14))

    def test_scoring_does_not_refit_imputer_or_scaler(self):
        p,q,error,w=self.sample();model=ErrorRanker('combined',.56).fit(p,q,error,w)
        median=model.medians.copy();center=model.scaler.mean_.copy();scale=model.scaler.scale_.copy()
        q[:]=1000000;q[0,0]=np.nan;model.score(p,q)
        self.assertTrue(np.array_equal(median,model.medians));self.assertTrue(np.array_equal(center,model.scaler.mean_));self.assertTrue(np.array_equal(scale,model.scaler.scale_))

    def test_single_error_class_cannot_fit(self):
        p,q,error,w=self.sample()
        with self.assertRaises(ValueError):ErrorRanker('combined',.56).fit(p,q,np.zeros(len(p)),w)

    def test_invalid_probability_quality_and_weights_rejected(self):
        p,q,error,w=self.sample()
        with self.assertRaises(ValueError):raw_confidence([1.1])
        q[0,0]=np.inf
        with self.assertRaises(ValueError):ErrorRanker('quality_only',.56).fit(p,q,error,w)
        with self.assertRaises(ValueError):weighted_quantile([0,1],[1,-1])

    def test_policy_validation_decoder_refuses_final_test(self):
        model=PolicyValidationNoise.__new__(PolicyValidationNoise)
        with self.assertRaises(PermissionError):model.make_pair(np.arange(1000),'1','bw',6,'test')

    def test_unknown_policy_condition_rejected(self):
        with self.assertRaises(ValueError):policy_weights(['em_6'])


if __name__=='__main__':unittest.main()
