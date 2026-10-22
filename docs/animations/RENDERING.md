# Reproduce this actual Manim animation

Renderer: **Manim Community 0.22.0**, Cairo. Python **3.12.15**. Pycairo 1.29.2, Cairo 1.18.6, ManimPango 0.7.0, Pango 1.58.2, PyAV 19.0.2. Packaging uses the locked environment's FFmpeg/ffprobe **9.0.2**. DejaVu Sans text avoids a LaTeX dependency.

The video is **24 seconds**, 1280×720, 15 fps, H.264/yuv420p, with no audio. Every scientific frame comes from real Manim. FFmpeg only remuxes the resulting video without video re-encoding, extracts PNGs, and makes an optional 768×432 GIF. These are illustrative calculations, not model or hardware measurements.

## Environment

The official [Manim conda installation guide](https://docs.manim.community/en/stable/installation/conda.html) supports conda-forge for native dependencies. Use conda/mamba/micromamba from its official distribution; [official micromamba instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). The original workspace-only bootstrap and package archive hashes are in `dependencies/`.

From the repository root, with micromamba available, create the exact Linux x86-64 environment without shell initialization or system changes:

```sh
RUNTIME="$PWD/.manim-runtime"
mkdir -p "$RUNTIME/home" "$RUNTIME/cache" "$RUNTIME/pkgs"
HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" \
  CONDA_PKGS_DIRS="$RUNTIME/pkgs" MAMBA_ROOT_PREFIX="$RUNTIME/mamba" \
  micromamba --no-rc create --yes --prefix "$RUNTIME/env" \
    --file docs/animations/dependencies/linux-64.lock
```

`dependencies/linux-64.lock` contains exact package URLs and MD5 checksums, with SHA-256 checksums and build metadata in `package-provenance.json`. The YAML is a less-strict cross-platform starting point, not a byte-for-byte environment lock. Do not commit `.manim-runtime/` or `.render-work/` build files.

## Render and verify

```sh
HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$RUNTIME/env/bin/python" docs/animations/source/render.py \
    --scene coordinate-readout --out docs/animations
"$RUNTIME/env/bin/python" -m unittest discover \
  -s docs/animations/source -p 'test_*.py' -v
```

The compatibility entrypoint `source/render.py` now calls `python -m manim render --renderer cairo --fps 15 --resolution 1280,720 --disable_caching -o coordinate-readout source/scenes.py CoordinateReadout` before packaging. It cannot invoke the previous raster renderer. `source/concepts.py` only re-exports pure mathematical models; `test_concepts.py` tests those models. `test_manim.py` verifies actual Manim, scene inheritance, equal geometric unit scales, discrete stage accounting, and whole-frame timing.

Use `--dry-run` to inspect the exact render command. To invoke Manim directly, run that command from this animation directory with `--media_dir .render-work`, then run `source/finalize.py --scene coordinate-readout --input RAW_MANIM_RENDER.mp4 --out .` using the same environment. Packaging does not modify scientific claims or execute a pretrained model.

## Accessibility and verification

`index.html` starts with a static poster, never autoplay. An explicit button enables the video; a second button restores the still. Reduced-motion changes stop playback. A complete transcript, WebVTT captions, and PNG poster are supplied. The GIF is opt-in through a link, not embedded as autoplay content. The standalone viewer's HTML/JavaScript was structurally checked; browser playback was not exercised here.

`qa/` contains actual decoded start/middle/end and three phase frames, ffprobe metadata, raw/packaged frame checksums, math tests and Manim contract tests. Full video decoding and exact decoded-frame equality after remuxing were checked. Scientific numbers were independently compared with the repository learning demos. Relevant scene/helper AST equivalence is recorded when the shared source module changed only in another scene after a render. Bitwise video identity across different platforms, codecs or font stacks is not promised.
