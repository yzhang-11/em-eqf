"""Retrieve exact author-published benchmark files from a pinned Git commit."""
import concurrent.futures as cf
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
STORE=Path('/root/autodl-fs/em_eqf_catalogs/earthquakenpp_20261106')
OUT=ROOT/'experiments/revision/catalog_benchmarks_20261106'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retrieve(item,repo,commit):
    relative=item['path'];path=STORE/'author_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/{repo}/{commit}/{relative}'
    if not path.exists():
        response=requests.get(url,timeout=(10,90));response.raise_for_status()
        data=response.content
        if data.startswith(b'version https://git-lfs'):raise ValueError('Git LFS pointer is not a dataset')
        temporary=path.with_name(path.name+'.part');temporary.write_bytes(data);temporary.replace(path)
    data=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob!=item['git_blob_sha'] or len(data)!=item['size']:raise ValueError(f'Pinned Git object mismatch: {relative}')
    meta=dict(path=str(path),repository_path=relative,url=url,git_blob_sha=blob,sha256=sha(path),
        bytes=len(data),commit=commit,retrieved_utc=datetime.now(timezone.utc).isoformat())
    if path.suffix=='.csv':
        f=pd.read_csv(path);meta.update(records=len(f),columns=list(f),head=f.head(2).to_dict('records'))
    print('PINNED',relative,len(data),meta.get('records',''),flush=True)
    return meta


def main():
    spec=json.loads((OUT/'enpp_sources.json').read_text());OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for item in pool.map(lambda item:retrieve(item,spec['repository'],spec['commit']),spec['files']):results.append(item)
    result=dict(repository=spec['repository'],commit=spec['commit'],files=results)
    (OUT/'author_snapshot_manifest.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print('AUTHOR_SNAPSHOT_READY',len(results),flush=True)


if __name__=='__main__':main()
