"""Explicitly download the pinned public checkpoint and verify every SHA-256."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',required=True);args=p.parse_args()
 manifest=json.loads(Path(__file__).with_name('MODEL_MANIFEST.json').read_text());dest=Path(args.destination);dest.mkdir(parents=True,exist_ok=True)
 for item in manifest['files']:
  name=item['name']
  if Path(name).name!=name:raise ValueError('Unexpected nested filename')
  target=dest/name
  if not target.exists():
   partial=dest/(name+'.partial');urllib.request.urlretrieve(item['url'],partial)
   if hashlib.sha256(partial.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Downloaded checksum mismatch: '+name)
   partial.replace(target)
  if target.stat().st_size!=item['size'] or hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Cached checksum mismatch: '+name)
 (dest/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Verified',len(manifest['files']),'files')
if __name__=='__main__':main()
