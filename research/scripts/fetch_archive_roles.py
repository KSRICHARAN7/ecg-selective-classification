"""Retrieve declared roles from the official ZIP with cached HTTP ranges.

Only requested role members are extracted. Compressed range blocks can contain
unselected members; these remain undecoded in the external transport cache.
No test member is extracted without the existing protocol-lock guard.
"""
import argparse,concurrent.futures,csv,hashlib,io,json,threading,time,zipfile
from pathlib import Path
import requests
from cohort_loader import ROOT,manifest,require_protocol_lock

URL='https://physionet.org/content/ptb-xl/get-zip/1.0.3/'


class RangeFile(io.RawIOBase):
    def __init__(self,url,cache,chunk=1024*1024,offline=False):
        self.url=url;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.offline=offline
        if offline:
            state=json.loads((self.cache/'source.json').read_text())
            if state['url']!=url or state['chunk_bytes']!=chunk:raise ValueError('Unexpected cached archive identity')
            self.size=state['bytes'];self.chunk=chunk;self.pos=0;self.local=threading.local()
            return
        response=requests.head(url,allow_redirects=True,timeout=(15,60));response.raise_for_status()
        self.size=int(response.headers['Content-Length']);self.chunk=chunk;self.pos=0;self.local=threading.local()
        state={'url':url,'bytes':self.size,'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified'),'chunk_bytes':chunk}
        state_path=self.cache/'source.json'
        if state_path.exists()and json.loads(state_path.read_text())!=state:raise ValueError('Remote archive changed; do not reuse transport cache')
        state_path.write_text(json.dumps(state,indent=2)+'\n')

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative offset')
        return self.pos

    def get_chunk(self,index):
        start=index*self.chunk;end=min(self.size,start+self.chunk)-1
        path=self.cache/f'{index:08d}.bin'
        if path.exists():
            data=path.read_bytes()
            if len(data)!=end-start+1:raise ValueError('Truncated transport cache')
            return data
        if self.offline:raise FileNotFoundError(f'Offline cache lacks archive block {index}')
        session=getattr(self.local,'session',None)
        if session is None:self.local.session=requests.Session();session=self.local.session
        for attempt in range(12):
            try:
                with session.get(self.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,60))as r:
                    if r.status_code!=206:raise ValueError(f'Expected206 range response, received{r.status_code}')
                    if r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('Unexpected Content-Range')
                    data=r.content
                if len(data)!=end-start+1:raise ValueError('Truncated range response')
                temp=path.with_suffix('.'+str(threading.get_ident())+'.part');temp.write_bytes(data);temp.replace(path)
                return data
            except Exception:
                if attempt==11:raise
                time.sleep(min(20,2**attempt))
        raise RuntimeError('Unreachable')

    def read(self,n=-1):
        if n<0:
            n=self.size-self.pos
            if n>32*1024*1024:raise ValueError('Whole-archive reads are disabled')
        n=min(n,self.size-self.pos);pieces=[]
        while n>0:
            index=self.pos//self.chunk;offset=self.pos%self.chunk
            data=self.get_chunk(index);take=min(n,len(data)-offset)
            pieces.append(data[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(pieces)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--roles',nargs='+',default=['train','checkpoint'],choices=['train','checkpoint','ranker','validation','test']);parser.add_argument('--cache',default=str(ROOT/'data/transport_cache'));parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--archive',type=Path,help='Local official PTB-XL1.0.3 ZIP; no network access')
    parser.add_argument('--offline',action='store_true',help='Use cached ranges only; fail if a required range is missing')
    parser.add_argument('--plan-only',action='store_true',help='Inspect needed versus cached blocks without retrieving waveforms')
    args=parser.parse_args()
    if not 1<=args.workers<=16:raise ValueError('Use at most16 archive-range workers')
    if 'test'in args.roles:require_protocol_lock()
    rows=[r for role in args.roles for r in manifest(role)]
    source=args.archive if args.archive else RangeFile(URL,args.cache,offline=args.offline)
    print('Reading official archive index',flush=True)
    with zipfile.ZipFile(source)as archive:
        names=archive.namelist();metadata_name=next(n for n in names if n.endswith('/ptbxl_database.csv')or n=='ptbxl_database.csv')
        prefix=metadata_name[:-len('ptbxl_database.csv')]
        if args.archive and hashlib.sha256(archive.read(metadata_name)).hexdigest()!=hashlib.sha256((ROOT/'data/source/ptbxl_database.csv').read_bytes()).hexdigest():raise ValueError('Local archive metadata differs from the verified source version')
        checksum_bytes=archive.read(prefix+'SHA256SUMS.txt')
        checksum_path=ROOT/'data/source/PTBXL_SHA256SUMS.txt';checksum_path.write_bytes(checksum_bytes)
        checksums={}
        for line in checksum_bytes.decode().splitlines():
            digest,name=line.split(maxsplit=1);checksums[name.strip().lstrip('./*')]=digest
        (ROOT/'data/source/PTBXL_LICENSE.txt').write_bytes(archive.read(prefix+'LICENSE.txt'))
        wanted=[]
        for row in rows:
            for suffix in ['.hea','.dat']:
                rel=row['filename_lr']+suffix;path=ROOT/'data/cohort'/rel
                if path.exists()and hashlib.sha256(path.read_bytes()).hexdigest()==checksums[rel]:continue
                wanted.append((rel,path,archive.getinfo(prefix+rel)))
        chunks=set()
        for rel,path,info in wanted:
            if args.archive:continue
            # The extra1KiB conservatively covers the local file header.
            first=info.header_offset//source.chunk
            last=(info.header_offset+30+len(info.filename.encode())+len(info.extra)+1024+info.compress_size)//source.chunk
            chunks.update(range(first,last+1))
        if not args.archive:chunks={i for i in chunks if i*source.chunk<source.size}
        plan={'roles':args.roles,'records':len(rows),'members_to_extract':len(wanted),'range_chunks':len(chunks),'archive_bytes':args.archive.stat().st_size if args.archive else source.size}
        if not args.archive:
            plan['required_blocks_already_cached']=sum((source.cache/f'{i:08d}.bin').exists()for i in chunks)
        print(json.dumps(plan),flush=True)
        if args.plan_only:
            (ROOT/'outputs/archive_download_plan.json').write_text(json.dumps(plan,indent=2)+'\n');return
        if not args.archive:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:
                for n,_ in enumerate(pool.map(source.get_chunk,sorted(chunks)),1):
                    if n%10==0 or n==len(chunks):print(f'Cached archive blocks{n}/{len(chunks)}',flush=True)
        out=ROOT/'outputs/archive_acquisition.jsonl'
        with out.open('w')as log:
            for n,(rel,path,info)in enumerate(wanted,1):
                data=archive.read(info)  # zipfile also checks the ZIP CRC.
                digest=hashlib.sha256(data).hexdigest()
                if digest!=checksums[rel]:raise ValueError('Provider checksum mismatch:'+rel)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                log.write(json.dumps({'path':rel,'bytes':len(data),'sha256':digest,'provider_sha256_match':True})+'\n')
                if n%1000==0:print(f'Extracted and verified{n}/{len(wanted)} members',flush=True)
        report={'roles':args.roles,'records':len(rows),'archive_url':URL,'archive_bytes':plan['archive_bytes'],'local_archive_used':bool(args.archive),'provider_checksum_file_sha256':hashlib.sha256(checksum_bytes).hexdigest(),'all_requested_members_provider_sha256_verified':True,'test_members_extracted':'test'in args.roles,'transport_note':'Cached compressed blocks can contain unselected bytes; only listed role members are decompressed or materialized.'}
        (ROOT/'outputs/archive_acquisition_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
