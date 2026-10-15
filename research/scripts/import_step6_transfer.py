"""Verify the data-only Colab transfer against local roles and provider hashes."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from cohort_loader import ROOT,manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);args=parser.parse_args()
    checks={}
    checksum_source=ROOT/'data/source/PTBXL_SHA256SUMS.txt'
    for line in checksum_source.read_text().splitlines():
        digest,name=line.split(maxsplit=1);checks[name.strip().lstrip('./*')]=digest
    rows=[row for role in ['train','checkpoint']for row in manifest(role)]
    wanted={row['filename_lr']+suffix for row in rows for suffix in ['.hea','.dat']}
    assert len(rows)==9711 and len(wanted)==19422
    with zipfile.ZipFile(args.archive)as z:
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('Duplicate archive member')
        if set(names)!=wanted|{'LICENSE.txt','acquisition_summary.json','selected_source_checksums.json'}:raise ValueError('Unexpected or missing source archive member')
        if z.testzip()is not None:raise ValueError('ZIP CRC failure')
        log=[]
        for n,rel in enumerate(sorted(wanted),1):
            data=z.read(rel);digest=hashlib.sha256(data).hexdigest()
            if digest!=checks[rel]:raise ValueError('Provider SHA256 mismatch: '+rel)
            path=ROOT/'data/cohort'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            log.append({'path':rel,'bytes':len(data),'sha256':digest,'provider_sha256_match':True})
            if n%2000==0:print(f'Imported and checksum-verified {n}/{len(wanted)} files',flush=True)
        original_license=(ROOT/'data/source/PTBXL_LICENSE.txt').read_bytes()
        if hashlib.sha256(original_license).hexdigest()!=checks['LICENSE.txt']:raise ValueError('Original provider license mismatch')
        # The data-only notebook read/wrote license text, normalizing CRLF.
        if z.read('LICENSE.txt')!=original_license.replace(b'\r\n',b'\n'):raise ValueError('Transferred license text mismatch')
        for name in ['acquisition_summary.json','selected_source_checksums.json']:
            (ROOT/'data/source'/('colab_'+name)).write_bytes(z.read(name))
    report={'roles':['train','checkpoint'],'records':len(rows),'source_files':len(wanted),'all_requested_members_provider_sha256_verified':True,'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),'archive_bytes':args.archive.stat().st_size,'Drive_file_id':'10chVGS2flN8SpFuJTEjIwPA9O-7XjqLz','provider_checksum_file_sha256':hashlib.sha256(checksum_source.read_bytes()).hexdigest(),'test_members_extracted':False,'validation_members_extracted':False,'source':'User-executed data-only notebook retrieving the official PhysioNet public AWS mirror','license_note':'Transfer license text has only CRLF-to-LF normalization; original provider license bytes preserved separately and hash verified','metadata_note':'Raw transfer sidecars retained as received; acquisition claims are independently checked, not trusted from sidecars.'}
    (ROOT/'outputs/archive_acquisition_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'outputs/archive_acquisition.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in log))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
