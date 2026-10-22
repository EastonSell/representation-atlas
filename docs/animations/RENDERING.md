# Reproduce this concept animation

This is an original, deterministic educational illustration, not empirical research evidence.
It uses the repository's learning-demo values and rules recorded in SOURCE.json.
No model execution, network access, training, device discovery or external assets are used.

## Existing rendering environment

- CPython 3.12.14
- Pillow 12.3.0
- ffmpeg / ffprobe 7.1.5, existing libx264 and GIF support
- DejaVu Sans regular/bold and DejaVu Sans Mono from /usr/share/fonts/truetype/dejavu/
- Linux x86_64

Manim was checked but was not available. No packages were installed.
The renderer requires the existing tools above and fails rather than installing anything.
Exact encoder arguments are retained in source/render.py. A fresh render writes its
commands, environment, font hashes and encoded media metadata to its generated qa/ directory.
The original QA receipts and example README snippet are not shipped in this public package.
MP4 encoding is bounded to two threads; rendering is sequential and 1280 × 720 at 12 fps.

From this directory (for example, docs/animations):

```sh
python -B -m unittest discover -s source -v
python -B source/render.py --scene coordinate-readout --out .
```

The rendering command overwrites its named media, poster, viewer, transcript and captions,
and creates an example README-SNIPPET.md and qa/ diagnostics. Those generated QA outputs
and snippet are not prerequisites for viewing or reproducing the shipped animation.
After a rerender or source edit, regenerate SHA256SUMS for the intended public file set.
The renderer does not refresh that integrity file automatically.
Use a new --out directory to retain an old render. Source functions are pure teaching calculations;
the toy service in source/concepts.py is intentionally not a production API implementation.
The unit suite checks all five educational concepts, even when rendering only this one.

## Files

- concept.mp4: 24-second silent H.264, 1280 × 720, 12 fps, broadly playable yuv420p, fast-start MP4
- concept.gif: optional animated preview, 768 × 432 at 6 fps; it loops, so do not embed it as the default reduced-motion path
- poster.png: no-motion summary, 1280 × 720
- captions.vtt: scene-aligned captions, supplementing the on-screen captions
- transcript.txt: full explanation and visual descriptions, including caveats
- index.html: local/static-host viewer, no autoplay, static by default, explicit play and return-to-poster controls
- SOURCE.json: historical teaching-demo provenance, retained unchanged
- source/: self-contained teaching calculations, renderer and unit tests
- SHA256SUMS: integrity hashes for the files shipped in this animation directory

Open index.html locally or serve it with any existing static server. GitHub's source-file view
does not run HTML as a page; the README links directly to the MP4 and text alternatives.
No hosting, public upload, analytics or external requests are included.

## Interpretation

EXACT GEOMETRY ILLUSTRATION · NOT A MODEL EXPERIMENT.
All mathematical values shown are deterministic constructed examples. Progress bars and markers
show lesson progression or dependency order, not measured training speed or network packet timing.
The poster and complete transcript carry the content without requiring playback.

## Retained provenance and fonts

SOURCE.json uses the historical repository name `representation-atlas`. This repository is now
`EastonSell/what-activations-do`. Its source-file path and SHA-256 refer to the unchanged teaching demo.
The shipped media are retained byte-for-byte; this integration does not claim a rerender.

Original render font SHA-256 values (font files are system dependencies, not bundled assets):

- DejaVuSans-Bold.ttf: `a4c5bc453ca281d90ea079e596da7ae0dfeb5777497c29ec254e76d97ff6f890`
- DejaVuSans.ttf: `57f73e11f51999432bf7ab22ce55b6f945d5eca1bf824404cfa9ec2e3718c84e`
- DejaVuSansMono.ttf: `54bf827eb99404e8f430c330ad30f063334f637eba0109b6a18d4f566a8e9dd8`
