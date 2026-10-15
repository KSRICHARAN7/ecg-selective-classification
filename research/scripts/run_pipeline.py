"""Stage runner; baseline fitting requires every declared source record."""
import argparse,subprocess,sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['metadata','download-pilot','pilot','partitions','loader-check','train-download-plan','archive-cache-plan','baseline','baseline-figures','fault-library','fault-previews','step8-download','reliability-development','policy-figures','step9-download','final-experiment','final-audit','final-figures','external-download','generalization','generalization-audit','generalization-figures','seed-sensitivities','robustness-checks','robustness-audit','robustness-figures','tests','verify'],required=True)
args=p.parse_args()
commands={
 'metadata':[sys.executable,str(root/'scripts/audit_metadata.py')],
 'download-pilot':[sys.executable,str(root/'scripts/download_pilot.py')],
 'pilot':[sys.executable,str(root/'scripts/pilot_check.py')],
 'partitions':[sys.executable,str(root/'scripts/build_partitions.py')],
 'loader-check':[sys.executable,str(root/'scripts/cohort_loader.py')],
 'train-download-plan':[sys.executable,str(root/'scripts/download_cohort.py'),'--role','train'],
 'archive-cache-plan':[sys.executable,str(root/'scripts/fetch_archive_roles.py'),'--offline','--plan-only','--roles','train','checkpoint'],
 'baseline':[sys.executable,str(root/'scripts/train_baseline.py')],
 'baseline-figures':[sys.executable,str(root/'scripts/plot_baseline.py')],
 'fault-library':[sys.executable,str(root/'scripts/verify_fault_library.py')],
 'fault-previews':[sys.executable,str(root/'scripts/plot_fault_library.py')],
 'step8-download':[sys.executable,str(root/'scripts/download_step8_sources.py')],
 'reliability-development':[sys.executable,str(root/'scripts/run_reliability_development.py')],
 'policy-figures':[sys.executable,str(root/'scripts/plot_reliability_policies.py')],
 'step9-download':[sys.executable,str(root/'scripts/download_final_sources.py')],
 'final-experiment':[sys.executable,str(root/'scripts/run_final_experiment.py')],
 'final-audit':[sys.executable,str(root/'scripts/audit_final_experiment.py')],
 'final-figures':[sys.executable,str(root/'scripts/plot_final_experiment.py')],
 'external-download':[sys.executable,str(root/'scripts/external_ecg.py')],
 'generalization':[sys.executable,str(root/'scripts/run_generalization.py')],
 'generalization-audit':[sys.executable,str(root/'scripts/audit_generalization.py')],
 'generalization-figures':[sys.executable,str(root/'scripts/plot_generalization.py')],
 'seed-sensitivities':[sys.executable,str(root/'scripts/run_seed_sensitivities.py')],
 'robustness-checks':[sys.executable,str(root/'scripts/run_robustness_checks.py')],
 'robustness-audit':[sys.executable,str(root/'scripts/audit_robustness.py')],
 'robustness-figures':[sys.executable,str(root/'scripts/plot_robustness.py')],
 'tests':[sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],
 'verify':[sys.executable,str(root/'scripts/verify_package.py')],
}
subprocess.run(commands[args.stage],cwd=root,check=True)
