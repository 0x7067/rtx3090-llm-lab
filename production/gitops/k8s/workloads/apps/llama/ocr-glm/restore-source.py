"""Restore an inventoried source release from this repository's inactive snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath


def relative(value):
    path=PurePosixPath(value)
    if not isinstance(value,str) or value!=path.as_posix() or path.is_absolute() or '..' in path.parts or value in ['', '.'] or '\\' in value:
        raise ValueError('Invalid snapshot path')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args()
    target=args.target
    if not target.is_absolute() or target.exists() or target.is_symlink():raise ValueError('Target must be a new absolute directory')
    snapshot=Path(__file__).resolve().parent/'source'
    record=json.loads((snapshot/'source-tree.json').read_text())
    mapping=json.loads((snapshot/'source-path-map.json').read_text())
    files=record['files']
    canonical=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    if record['algorithm']!='sha256-path-content-v1' or hashlib.sha256(canonical).hexdigest()!=record['sha256'] or set(files)!=set(mapping):
        raise ValueError('Invalid source inventory')
    verified=[]
    for name,digest in files.items():
        relative(name);relative(mapping[name])
        source=snapshot/mapping[name]
        if not source.resolve().is_relative_to(snapshot.resolve()) or not source.is_file() or source.is_symlink():raise ValueError('Invalid stored source')
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Stored source differs from inventory')
        verified.append((name,data))
    target.mkdir(mode=0o750,parents=True)
    for name,data in verified:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as output:output.write(data)
        destination.chmod(0o644)
    (target/'source-tree.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    (target/'node_modules').symlink_to('/opt/finance-ocr/node_modules',target_is_directory=True)
    print(json.dumps({'sourceSha256':record['sha256'],'restoredFiles':len(files),'readOnlyMountRequired':True}))


if __name__=='__main__':main()
