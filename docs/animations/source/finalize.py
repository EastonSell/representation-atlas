"""Package an ACTUAL Manim render; ffmpeg only remuxes/extracts/creates GIF preview.

Never creates scientific frames. Render the scene with Manim before using this.
"""
from pathlib import Path
import argparse
import hashlib
import html
import importlib.metadata
import json
import shutil
import subprocess
import sys


# Prefer the same locked environment's media binaries over incidental host tools.
FFMPEG=str(Path(sys.executable).parent/'ffmpeg') if (Path(sys.executable).parent/'ffmpeg').exists() else shutil.which('ffmpeg')
FFPROBE=str(Path(sys.executable).parent/'ffprobe') if (Path(sys.executable).parent/'ffprobe').exists() else shutil.which('ffprobe')

def run(command):subprocess.run(command,check=True)

def probe(path):return json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(path)]))

def stamp(seconds):return f'00:00:{seconds:02d}.000'


def package(slug,source,out):
    import manim
    assert manim.__version__=='0.22.0'
    story=json.loads((Path(__file__).parent/'storyboards.json').read_text())[slug]
    out.mkdir(parents=True,exist_ok=True);qa=out/'qa';qa.mkdir(exist_ok=True)
    original=probe(source);video=next(x for x in original['streams'] if x['codec_type']=='video')
    assert video['codec_name']=='h264'
    assert not any(x['codec_type']=='audio' for x in original['streams'])
    assert (video['width'],video['height'])==(1280,720)
    assert 23.9<=float(original['format']['duration'])<=24.15
    assert video['r_frame_rate']=='15/1'
    target=out/'concept.mp4'
    remux=[FFMPEG,'-v','error','-y','-i',str(source),'-map','0:v:0','-c:v','copy','-an','-movflags','+faststart',str(target)]
    run(remux)
    for filename,t in [('poster.png',23),('qa/start.png',0),('qa/middle.png',12),('qa/end.png',23),('qa/phase-1.png',6),('qa/phase-2.png',14),('qa/phase-3.png',22)]:
        run([FFMPEG,'-v','error','-y','-ss',str(t),'-i',str(target),'-frames:v','1','-threads','1',str(out/filename)])
    gif=[FFMPEG,'-v','error','-y','-i',str(target),'-filter_complex',
        'fps=6,scale=768:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=none',
        '-threads','2','-filter_complex_threads','1','-loop','0',str(out/'concept.gif')]
    run(gif)
    transcript=[story['title'],'Rendered with actual Manim Community 0.22.0 (Cairo).',story['badge'],'',
                '24 seconds, silent. All meaning is available without motion in this complete transcript.','']
    captions=['WEBVTT','']
    for i,shot in enumerate(story['scenes']):
        transcript += [f"{shot['start']:02d}–{shot['end']:02d} seconds",shot['caption'],shot['description'],'']
        captions += [str(i+1),f"{stamp(shot['start'])} --> {stamp(shot['end'])}",shot['caption'],'']
    (out/'transcript.txt').write_text('\n'.join(transcript),encoding='utf8')
    (out/'captions.vtt').write_text('\n'.join(captions),encoding='utf8')
    title=html.escape(story['title']);alt=html.escape(story['poster_alt'],quote=True)
    sections=''.join(f'<section><h2>{s["start"]:02d}–{s["end"]:02d} seconds</h2><p>{html.escape(s["caption"])}</p><p>{html.escape(s["description"])}</p></section>' for s in story['scenes'])
    viewer=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>:root{{color-scheme:dark}}body{{max-width:1050px;margin:auto;padding:24px;font:18px/1.6 system-ui;background:#0b0f15;color:#f4f5f7}}a{{color:#5ad8df}}img,video{{width:100%;height:auto}}button{{font:inherit;padding:10px 16px;background:#18202b;color:inherit;border:1px solid #5ad8df;border-radius:6px}}[hidden]{{display:none!important}}h1{{font-size:36px;line-height:1.2}}h2{{font-size:23px}}:focus-visible{{outline:3px solid #ffd27c;outline-offset:4px}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important;scroll-behavior:auto!important}}}}</style>
<main><h1>{title}</h1><p>Actual Manim Community 0.22.0. {html.escape(story['badge'])}.</p><p>Nothing plays automatically. The static poster and complete transcript are the reduced-motion path.</p>
<img id="poster" src="poster.png" alt="Final Manim scene: {alt}"><video id="clip" controls playsinline preload="none" poster="poster.png" hidden aria-label="{title}"><source src="concept.mp4" type="video/mp4"><track kind="captions" src="captions.vtt" srclang="en" label="English" default>Use the transcript below if video is unavailable.</video>
<p><button id="play" type="button">Choose to play animation</button> <button id="still" type="button" hidden>Return to static poster</button></p><p id="note"></p>
<p><a href="concept.mp4">MP4</a> · <a href="concept.gif">Optional GIF</a> · <a href="poster.png">Static PNG</a> · <a href="transcript.txt">Text transcript</a> · <a href="captions.vtt">Captions</a></p><h2>Complete transcript</h2>{sections}</main>
<script>const p=document.querySelector('#poster'),v=document.querySelector('#clip'),play=document.querySelector('#play'),still=document.querySelector('#still'),n=document.querySelector('#note'),q=matchMedia('(prefers-reduced-motion: reduce)');function note(){{n.textContent=q.matches?'Reduced motion is requested. Keep this static view, or explicitly choose to play.':'Static by default; playback is your choice.'}}function stop(){{v.pause();v.currentTime=0;v.hidden=true;p.hidden=false;play.hidden=false;still.hidden=true}}note();q.addEventListener?.('change',()=>{{stop();note()}});play.addEventListener('click',()=>{{p.hidden=true;v.hidden=false;play.hidden=true;still.hidden=false;v.play().catch(()=>{{n.textContent='Use the video controls to play, or return to the poster.'}})}});still.addEventListener('click',()=>{{stop();play.focus()}});</script></html>'''
    (out/'index.html').write_text(viewer)
    snippet=f"[![{story['poster_alt']}](docs/animations/poster.png)](docs/animations/concept.mp4)\n\n[24-second Manim MP4](docs/animations/concept.mp4) · [Optional GIF](docs/animations/concept.gif) · [No-motion transcript](docs/animations/transcript.txt) · [Captions](docs/animations/captions.vtt)\n\nActual Manim Community renderer. {story['badge']}. No audio. Poster and transcript provide a no-motion alternative.\n"
    (out/'README-SNIPPET.md').write_text(snippet)
    src=out/'source';src.mkdir(exist_ok=True)
    for name in ['scenes.py','models.py','test_concepts.py','test_manim.py','concepts.py','render.py','storyboards.json','finalize.py']:
        original_source=Path(__file__).parent/name
        if original_source.resolve()!=(src/name).resolve():shutil.copy2(original_source,src/name)
    final_probe=probe(target)
    (qa/'ffprobe.json').write_text(json.dumps(final_probe,indent=2)+'\n')
    manifest={'renderer':'Manim Community','manim_version':manim.__version__,'manim_module':manim.__file__,
        'scene_class':story['class'],'manim_source_sha256':hashlib.sha256((src/'scenes.py').read_bytes()).hexdigest(),
        'raw_render_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'packaged_video_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'remux_command':remux,'gif_command':gif,'python':sys.version,
        'media_tools':{'ffmpeg':subprocess.check_output([FFMPEG,'-version'],text=True).splitlines()[0],'ffprobe':subprocess.check_output([FFPROBE,'-version'],text=True).splitlines()[0]},
        'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    (qa/'render-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    run([FFMPEG,'-v','error','-i',str(target),'-f','null','-'])
    # Verify remuxing did not change a single decoded frame.
    for label_,path in [('raw',source),('packaged',target)]:
        run([FFMPEG,'-v','error','-y','-i',str(path),'-f','framemd5',str(qa/f'{label_}-frames.md5')])
    raw_rows=[x for x in (qa/'raw-frames.md5').read_text().splitlines() if not x.startswith('#')]
    packaged_rows=[x for x in (qa/'packaged-frames.md5').read_text().splitlines() if not x.startswith('#')]
    assert raw_rows==packaged_rows
    print(json.dumps({'scene':slug,'output':str(out),'actual_manim':manim.__version__,'frames_unchanged_by_remux':True,'duration':final_probe['format']['duration']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',required=True);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();package(args.scene,args.input,args.out)
