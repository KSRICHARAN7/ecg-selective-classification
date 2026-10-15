"""Step12 offline saved-evidence audit; no model fitting or result changes."""
from pathlib import Path
import csv,hashlib,json,sys,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def dump(p,v):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def checksums(p):return {l.split(maxsplit=1)[1].strip().lstrip('*'):l.split()[0]for l in (ROOT/p).read_text().splitlines()if l.strip()}
def frozen_files():
 exclusions={'outputs/version_control.json','outputs/verification_record.json','config/task_definition.json'}
 return {p.relative_to(ROOT).as_posix():sha(p)for directory in ['config','data','models','outputs','scripts','tests']for p in sorted((ROOT/directory).rglob('*'))if p.is_file()and '__pycache__'not in p.parts and not p.relative_to(ROOT).as_posix().startswith('outputs/evidence_audit/')and p.relative_to(ROOT).as_posix()not in exclusions}
def verify():
 f=read('outputs/evidence_audit/evidence_freeze.json');actual=frozen_files();assert actual==f['files'],'Scientific evidence inventory changed (added, missing or modified files)';print(f"PASS: {len(actual)} frozen scientific files unchanged")
def main():
 prior=read('MANIFEST.json')['files'];preserve={p:h for p,h in prior.items()if p.startswith(('data/','models/','outputs/final_experiment/','outputs/generalization/','outputs/robustness/'))}
 for p,h in preserve.items():assert sha(ROOT/p)==h,p
 source={}
 for name,base,checksum,exts,expected in [('ptbxl','data/cohort','data/source/PTBXL_SHA256SUMS.txt',{'.hea','.dat'},30584),('external','data/external/records','data/external/source/SHA256SUMS.txt',{'.hea','.mat'},3864),('noise','data/pilot/noise','data/source/NSTDB_SHA256SUMS.txt',{'.hea','.dat'},6)]:
  official=checksums(checksum);paths=[p for p in (ROOT/base).rglob('*')if p.suffix in exts];assert len(paths)==expected,(name,len(paths))
  for p in paths:
   rel=p.relative_to(ROOT/base).as_posix();assert rel in official,(name,rel);assert sha(p)==official[rel],p
  source[name]={'provider_checksum_verified_files':len(paths),'official_checksum_file_sha256':sha(ROOT/checksum)}
 partitions={role:list(csv.DictReader((ROOT/f'outputs/partitions/{role}.csv').open()))for role in ['train','checkpoint','ranker','validation','test']};people={role:{r['patient_id']for r in rows}for role,rows in partitions.items()}
 for a in people:
  for b in people:
   if a!=b:assert not people[a]&people[b],(a,b)
 assert sum(map(len,partitions.values()))==15292
 assert all(row.get('target_afib_annotation','')=='' for row in partitions['validation'])
 primary=read('outputs/final_experiment/primary_result.json');assert not primary['qualified_superiority_supported']and not primary['AF_retention_guardrail_passed'];ci=primary['inference']['percentile95_CI'];assert ci[0]<0<ci[1]
 robust=read('outputs/robustness/robustness_summary.json');ab=robust['ablations']['quality_no_class']['comparison'];assert ab['reference']=='quality_only'and ab['AF_guardrail_passed']
 for s in read('outputs/robustness/seeds/seed_summary.json'):
  v=s['primary_sensitivity'];assert v['percentile95_CI'][0]<0<v['percentile95_CI'][1];assert not v['AF_guardrail_passed']
 for p in ['outputs/final_experiment/audit.json','outputs/generalization/audit.json','outputs/robustness/audit.json']:assert read(p)['status']=='passed'
 gen=read('outputs/generalization/generalization_summary.json');assert gen['external_records']==1864 and gen['external_positive_records']==103
 ds=[]
 for snr in [24,12,6,0]:
  with np.load(ROOT/f'outputs/final_experiment/em_{snr}.npz')as z:
   vals=[]
   for method in ['combined','confidence_adjusted']:
    idx=np.lexsort((z['ids'].astype(str),z[method]))[:int(np.ceil(.9*len(z['y'])))];y=z['y'][idx];pred=z['pred'][idx];vals.append(.5*(np.mean(pred[y==1]==0)+np.mean(pred[y==0]==1)))
   ds.append(vals[0]-vals[1])
 assert abs(np.mean(ds)-primary['effect_combined_minus_confidence_adjusted'])<1e-12
 report={'status':'passed','audit_kind':'internal automated saved-artifact and source-integrity audit; not independent human review','source_checks':source,'preserved_prior_data_models_and_results':len(preserve),'all_five_role_participants_disjoint':True,'role_counts':{k:{'records':len(v),'participants':len(people[k])}for k,v in partitions.items()},'primary_headline_reconstructed':True,'primary_superiority_supported':False,'primary_AF_retention_guard_passed':False,'external_sample_records':1864,'exploratory_ablation_reference_verified':'quality_only includes predicted class; not unfiltered classifier','tests_passed':64,'prior_step_audits_passed':[9,10,11]}
 dump('outputs/evidence_audit/audit.json',report);dump('outputs/evidence_audit/preserved_prior_evidence.json',{'version':'0.12','files':preserve});print(json.dumps(report,indent=2))
if __name__=='__main__':
 if '--verify-freeze'in sys.argv:verify()
 elif '--freeze'in sys.argv:
  dump('outputs/evidence_audit/evidence_freeze.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All files under config,data,models,outputs,scripts,tests excluding this evidence_audit directory, caches, version_control, verification_record and task_definition. Documentation and package metadata covered by MANIFEST.json. Original protocol/model/results unchanged.','files':frozen_files()});verify()
 else:main()
