"""Maintainer utility; regenerate only after intentional release changes."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=json.loads((root/'SOURCE_PROVENANCE.json').read_text())
paths=[root/'research'/name for name in source['source_files_sha256']]
for pattern in ['*.md','*.txt','LICENSE','.gitignore','.gitattributes','SOURCE_*.json','tools/*.py','docs/*.md','docs/verification/*.json','.github/workflows/*.yml','provider-checksums/NSTDB_SHA256SUMS.txt','research/outputs/final_experiment/*.svg']:
    paths.extend(root.glob(pattern))
for p in (root/'extensions/quality-replication').rglob('*'):
    if p.is_file() and p.suffix in ['.md','.json','.csv','.py','.svg','.pt'] and not any(part.startswith('PRIVATE') or part=='__pycache__' for part in p.relative_to(root).parts):
        paths.append(p)
paths=sorted({p for p in paths if p.is_file()})
manifest={'hash_algorithm':'sha256','scope':'public code, model and aggregate evidence release; no cryptographic signature','sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
(root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Manifest files:',len(paths))
