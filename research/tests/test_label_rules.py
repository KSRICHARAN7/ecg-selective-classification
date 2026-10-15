import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_metadata import classify


class LabelRules(unittest.TestCase):
    def row(self, codes, **updates):
        return dict(age='60', validated_by_human='True', scp_codes=str(codes), **updates)

    def test_zero_likelihood_is_still_present_afib(self):
        self.assertEqual(classify(self.row({'AFIB': 0.0}), {'AFIB', 'SR'})[:2], ('eligible', 1))

    def test_unknown_rhythm_is_not_assumed_negative(self):
        self.assertEqual(classify(self.row({'MI': 100.0}), {'AFIB', 'SR'})[:2], ('no_rhythm_statement', None))

    def test_other_annotated_rhythm_is_negative(self):
        self.assertEqual(classify(self.row({'SR': 0.0, 'MI': 100.0}), {'AFIB', 'SR'})[:2], ('eligible', 0))

    def test_mixed_afib_retained_and_flaggable(self):
        result = classify(self.row({'AFIB': 0.0, 'AFLT': 0.0}), {'AFIB', 'AFLT'})
        self.assertEqual(result, ('eligible', 1, {'AFIB', 'AFLT'}))

    def test_adult_and_review_eligibility(self):
        row = self.row({'AFIB': 0.0}); row['age'] = '17'
        self.assertEqual(classify(row, {'AFIB'})[0], 'under_18')
        row['age'] = '60'; row['validated_by_human'] = 'False'
        self.assertEqual(classify(row, {'AFIB'})[0], 'not_human_reviewed')
        self.assertEqual(classify(row, {'AFIB'}, False)[1], 1)


if __name__ == '__main__':
    unittest.main()
