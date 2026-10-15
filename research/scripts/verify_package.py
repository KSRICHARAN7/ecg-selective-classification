"""Verify the file inventory and every SHA-256 digest in this archive."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'MANIFEST.json').read_text())
actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
          if p.is_file() and not any(x in p.parts for x in ['__pycache__','.git','.venv']) and p.name != 'MANIFEST.json'}
assert actual == set(manifest['files']), 'Unexpected or missing package files'
for name, digest in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print(f'PASS: {len(actual)} package files match the manifest')
