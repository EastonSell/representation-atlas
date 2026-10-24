"""CLI for rebuilding the measured narration plan before rendering."""
import argparse
import json
from pathlib import Path
from timeline import build_timeline

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--timings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.resolve() in (args.manifest.resolve(),args.timings.resolve()):
        raise ValueError('Output must not overwrite an input')
    result=build_timeline(json.loads(args.manifest.read_text()),json.loads(args.timings.read_text()),30)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
