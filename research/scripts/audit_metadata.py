"""Step 2 metadata audit. No model fitting or test-outcome summaries.

Run from any directory: python3 scripts/audit_metadata.py
Only aggregate identifiers/split integrity are examined outside development folds.
"""
import ast
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def classify(row, rhythm_codes, require_human=True):
    """AFIB statement presence, not numeric likelihood, determines annotation.

    Return reason, label or None, rhythm statements. Mixed labels are retained
    and flagged; their removal is a planned sensitivity analysis.
    """
    if not row.get('age'):
        return 'missing_age', None, set()
    if float(row['age']) < 18:
        return 'under_18', None, set()
    if require_human and row.get('validated_by_human') != 'True':
        return 'not_human_reviewed', None, set()
    statements = ast.literal_eval(row['scp_codes'])
    if not isinstance(statements, dict):
        raise ValueError('scp_codes must be a dictionary')
    rhythms = set(statements) & rhythm_codes
    if not rhythms:
        return 'no_rhythm_statement', None, rhythms
    return 'eligible', int('AFIB' in rhythms), rhythms


def dump(name, obj):
    (ROOT / 'outputs' / name).write_text(
        json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def run():
    config = json.loads((ROOT / 'config/task_definition.json').read_text())
    for name, expected in config['source_sha256'].items():
        data = (ROOT / 'data/source' / name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f'Unexpected source hash: {name}')
    rows = read_csv(ROOT / 'data/source/ptbxl_database.csv')
    statements = read_csv(ROOT / 'data/source/scp_statements.csv')
    rhythm_codes = {r[''] for r in statements if r['rhythm'] == '1.0'}
    groups = {'development': set(), 'validation': set(), 'test': set()}
    for row in rows:
        fold = int(row['strat_fold'])
        group = 'development' if fold <= 8 else ('validation' if fold == 9 else 'test')
        if not 1 <= fold <= 10:
            raise ValueError('Invalid fold')
        groups[group].add(row['patient_id'])
    overlaps = {f'{a}_{b}': len(groups[a] & groups[b])
                for a, b in [('development', 'validation'), ('development', 'test'), ('validation', 'test')]}
    duplicates = len(rows) - len({r['ecg_id'] for r in rows})
    if any(overlaps.values()) or duplicates or any(not r['patient_id'] for r in rows):
        raise ValueError('Recording ID or participant partition integrity failure')
    dump('source_integrity.json', {
        'dataset_version': '1.0.3', 'recordings': len(rows),
        'participants': len(set.union(*groups.values())),
        'patient_overlap': overlaps, 'duplicate_ecg_ids': duplicates,
        'source_sha256': config['source_sha256'],
        'scope': 'Identifiers and split integrity only outside folds 1 to 8',
    })
    dev = [r for r in rows if int(r['strat_fold']) <= 8]
    exclusion, eligible = Counter(), []
    rhythm_counts, likelihoods, mixed = Counter(), Counter(), Counter()
    for row in dev:
        reason, label, rhythms = classify(row, rhythm_codes)
        exclusion[reason] += 1
        if reason != 'eligible':
            continue
        code_values = ast.literal_eval(row['scp_codes'])
        if label:
            likelihoods[str(code_values['AFIB'])] += 1
            mixed.update(rhythms - {'AFIB'})
        rhythm_counts.update(rhythms)
        eligible.append({
            'ecg_id': row['ecg_id'], 'patient_id': row['patient_id'],
            'strat_fold': row['strat_fold'], 'target_afib_annotation': label,
            'rhythm_codes': '|'.join(sorted(rhythms)),
            'mixed_rhythm_annotation': int(len(rhythms) > 1),
            'filename_hr': row['filename_hr'], 'filename_lr': row['filename_lr'],
        })
    positives = [r for r in eligible if r['target_afib_annotation']]
    negative = [r for r in eligible if not r['target_afib_annotation']]
    summary = {
        'scope': 'Development folds 1 to 8 only; no model or test evaluation',
        'development_recordings_before_exclusions': len(dev),
        'exclusion_flow_sequential': dict(exclusion),
        'eligible_recordings': len(eligible),
        'eligible_participants': len({r['patient_id'] for r in eligible}),
        'positive_recordings': len(positives),
        'positive_participants': len({r['patient_id'] for r in positives}),
        'negative_recordings': len(negative),
        'positive_fraction': len(positives) / len(eligible),
        'positive_mixed_rhythm_recordings': sum(r['mixed_rhythm_annotation'] for r in positives),
        'positive_coannotated_rhythms': dict(mixed),
        'positive_likelihood_values': dict(likelihoods),
        'rhythm_statement_counts_not_mutually_exclusive': dict(rhythm_counts),
        'candidate_all_review_status_recordings': sum(classify(r, rhythm_codes, False)[0] == 'eligible' for r in dev),
        'candidate_all_review_status_positive_recordings': sum(classify(r, rhythm_codes, False)[1] == 1 for r in dev),
    }
    dump('step_2_development_cohort.json', summary)
    with (ROOT / 'outputs/development_cohort_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(eligible[0]))
        writer.writeheader(); writer.writerows(eligible)
    with (ROOT / 'outputs/development_fold_summary.csv').open('w', newline='') as f:
        fields = ['fold', 'recordings', 'participants', 'positive_recordings', 'negative_recordings']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for fold in range(1, 9):
            subset = [r for r in eligible if int(r['strat_fold']) == fold]
            pos = sum(r['target_afib_annotation'] for r in subset)
            writer.writerow(dict(zip(fields, [fold, len(subset), len({r['patient_id'] for r in subset}), pos, len(subset)-pos])))
    dump('execution_environment.json', {'python': platform.python_version(), 'dependencies': 'Python standard library only'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
