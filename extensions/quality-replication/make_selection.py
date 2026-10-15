"""Reconstruct the fixed sample from the provider inventory without header labels."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('inventory',type=Path,help='ecg-arrhythmia1.0.0 SHA256SUMS.txt');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'PRIVATE_selection.json');args=p.parse_args();here=Path(__file__).resolve().parent;plan=json.loads((here/'PLAN.json').read_text());headers=[]
for line in args.inventory.read_text().splitlines():
 _,name=line.split(maxsplit=1);name=name.strip().lstrip('./*')
 if name.startswith('WFDBRecords/')and name.endswith('.hea'):headers.append(name)
assert len(headers)==45152
old=set(sorted(headers,key=lambda n:hashlib.sha256(f'20261019|{n}'.encode()).digest())[:2000])
selected=sorted([n for n in headers if n not in old],key=lambda n:hashlib.sha256(f"{plan['external']['sample_seed']}|{n}".encode()).digest())[:plan['external']['sample_headers']]
digest=hashlib.sha256('\n'.join(selected).encode()).hexdigest();assert digest==plan['external']['selected_paths_sha256'];args.output.write_text(json.dumps({'selected':selected,'sha256':digest},indent=2)+'\n');print('Reconstructed fixed selection; digest matches locked plan.')
