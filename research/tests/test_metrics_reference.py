import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metrics_reference import selective_metrics, retained_indices


class Metrics(unittest.TestCase):
    def test_dropping_all_positive_cases_cannot_be_success(self):
        result=selective_metrics([1,0,0,0],[0,0,0,0],[1,2,3])
        self.assertEqual(result['selective_error'],0)
        self.assertIsNone(result['balanced_selective_error'])
        self.assertEqual(result['positive_retention'],0)

    def test_class_balance_and_retention_have_distinct_denominators(self):
        result=selective_metrics([1,1,0,0],[1,0,1,0],[0,2,3])
        self.assertEqual(result['balanced_selective_error'],.25)
        self.assertEqual(result['positive_retention'],.5)
        self.assertEqual(result['retained_sensitivity'],1)
        self.assertEqual(result['retained_true_positive_fraction_of_all_positives'],.5)

    def test_fixed_budget_and_ties(self):
        self.assertEqual(retained_indices([.3,.1,.1],['c','b','a'],2/3),[2,1])

    def test_nonfinite_score_is_not_silently_retained(self):
        with self.assertRaises(ValueError):retained_indices([float('nan')],['a'])


if __name__=='__main__':unittest.main()
