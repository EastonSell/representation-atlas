"""Compatibility entrypoint: renders with the real Manim package, never Pillow.

Run: python source/render.py --scene coordinate-readout --out .
Use --dry-run to print the exact Manim command without rendering.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


def main():
    here=Path(__file__).resolve().parent
    stories=json.loads((here/'storyboards.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True,choices=stories)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--workdir',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    import manim
    from manim import Scene
    if manim.__version__!='0.22.0' or Scene.__module__!='manim.scene.scene':
        raise RuntimeError('Actual Manim Community 0.22.0 is required.')
    out=args.out.resolve();work=(args.workdir or out/'.render-work').resolve()
    command=[sys.executable,'-m','manim','render','--renderer','cairo','--fps','15',
        '--resolution','1280,720','--media_dir',str(work),'--disable_caching',
        '--verbosity','WARNING','-o',args.scene,str(here/'scenes.py'),stories[args.scene]['class']]
    if args.dry_run:
        print(json.dumps({'renderer':'actual Manim Community','version':manim.__version__,'command':command},indent=2))
        return
    out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    (work/'render-command.json').write_text(json.dumps(command,indent=2)+'\n')
    subprocess.run(command,check=True,env=env)
    raw=list((work/'videos').rglob(args.scene+'.mp4'))
    if len(raw)!=1:raise RuntimeError(f'Expected one Manim MP4, found {raw}')
    subprocess.run([sys.executable,str(here/'finalize.py'),'--scene',args.scene,
        '--input',str(raw[0]),'--out',str(out)],check=True,env=env)


if __name__=='__main__':main()
