"""Offline verification of release integrity and saved aggregate arithmetic.

No external packages, source ECGs, individual predictions, or network required.
This checks internal consistency, not independent replication of predictions.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, context):
    if expected is None:
        assert actual in ('', 'nan', 'None'), (context, actual)
    else:
        assert math.isclose(float(actual), expected, abs_tol=1e-12, rel_tol=1e-12), (context, actual, expected)


def main():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        assert path.is_file(), f'Missing release file: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Changed release file: {name}'
    primary = json.loads((ROOT / 'research/outputs/final_experiment/primary_result.json').read_text())
    checked = 0
    for path in (ROOT / 'research/outputs').rglob('*.csv'):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {'TP', 'FN', 'TN', 'FP', 'retained', 'N'} <= rows[0].keys():
            continue
        for r in rows:
            tp, fn, tn, fp, kept, total = (float(r[k]) for k in ['TP', 'FN', 'TN', 'FP', 'retained', 'N'])
            assert min(tp, fn, tn, fp) >= 0
            close(kept, tp+fn+tn+fp, path.name)
            assert 0 <= kept <= total
            close(r['coverage'], kept/total, path.name)
            close(r['selective_error'], (fn+fp)/kept if kept else None, path.name)
            if 'balanced_selective_error' in r:
                close(r['balanced_selective_error'], .5*(fn/(tp+fn)+fp/(tn+fp)) if tp+fn and tn+fp else None, path.name)
            if 'positive_retained' in r:
                close(r['positive_retained'], tp+fn, path.name)
                close(r['negative_retained'], tn+fp, path.name)
            checked += 1
    rows = list(csv.DictReader((ROOT / 'research/outputs/final_experiment/matched_coverage_metrics.csv').open()))
    differences = []
    for snr in [24, 12, 6, 0]:
        methods = primary['primary_conditions'][str(snr)]
        for name, expected in methods.items():
            row = next(r for r in rows if r['condition'] == f'em_{snr}' and r['method'] == name and float(r['target_coverage']) == .9)
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    close(row[key], value, f'{snr}/{name}/{key}')
        differences.append(methods['combined']['balanced_selective_error']-methods['confidence_adjusted']['balanced_selective_error'])
    close(primary['effect_combined_minus_confidence_adjusted'], sum(differences)/4, 'primary effect')
    assert not primary['qualified_superiority_supported']
    assert not primary['AF_retention_guardrail_passed']
    print(json.dumps({'release_files_verified': len(manifest['sha256']), 'aggregate_metric_rows_checked': checked, 'primary_method_condition_cells_checked': 16, 'superiority_supported': False, 'scope': 'saved-evidence integrity and arithmetic; not prediction replication'}, indent=2))


if __name__ == '__main__':
    main()
