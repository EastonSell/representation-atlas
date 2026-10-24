# Reproduce the narrated version

This additive package preserves the original silent animation and its historical QA. It explains the same toy or calculation, not a new experimental result.

Use an isolated Python environment with Manim 0.22.0, Matplotlib, Kokoro-ONNX 0.6.1, SoundFile, FFmpeg and ffprobe. Native Cairo/Pango dependencies are required; consult [official Manim installation guidance](https://docs.manim.community/en/stable/installation.html). `requirements-recorded.txt` records the tested Python environment, not a cross-platform native lock. Formula SVGs use Computer Modern mathtext because the recorded system LaTeX lacked required formats.

Obtain the model and stock voice assets from the [official Kokoro-ONNX release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0), subject to the [model card and license](https://huggingface.co/hexgrad/Kokoro-82M). Put `kokoro-v1.0.onnx` and `voices-v1.0.bin` in a local model directory; hashes are recorded in SOURCE.json. Models are not bundled. No API key is required.

From this directory, using the isolated environment's Python and a fresh work directory:

```sh
python source/synthesize_narration.py narration.json --model-dir "$MODEL_DIR" --output-dir "$WORK/speech" --voice am_michael --speed 0.95 --tail-out 0.5
python source/prepare_timeline.py --manifest narration.json --timings "$WORK/speech/timings.json" --output "$WORK/measured-plan.json"
export NARRATED_PLAN="$WORK/measured-plan.json"
export NARRATED_WIDTH=1280 NARRATED_HEIGHT=720
python -m manim render --renderer cairo --fps 30 --resolution 1280,720 --disable_caching --media_dir "$WORK/media" -o visual-master source/narrated_scenes.py NarratedGeometry
python source/package_video.py --video "$WORK/media/videos/narrated_scenes/720p30/visual-master.mp4" --manifest "$WORK/measured-plan.json" --timings "$WORK/speech/timings.json" --work-dir "$WORK/assembly" --output "$WORK/narrated.mp4" --fps 30 --lead-in 0 --tail-out 0 --final-tail-out 0 --target-tp -2 --subtitles
python -m unittest discover -s tests -v
ffprobe -v error -show_streams -show_format "$WORK/narrated.mp4"
ffmpeg -v error -i "$WORK/narrated.mp4" -f null -
```

Before a master, render the complete preview with `NARRATED_WIDTH=640 NARRATED_HEIGHT=360` and `--resolution 640,360`. The actual measured speech controls scene timing. The `base_scenes.py` file is a narrated-path adaptation of the original source, with explicit transition-state labels and delayed spoken reveals; the old source remains unchanged one directory above this package.

Speech uses a distinct stock synthetic voice, not an imitation of any person. Canonical `text` remains separate from pronunciation-only `tts_text`. Captions are sentence-duration estimates, not word-aligned subtitles. The packager normalizes the full track once, encodes AAC once, checks A/V endpoints and preserves the input video. It may add a frame of rounding padding. Do not route this audio through the old silent-only finalizer.

QA includes independent arithmetic checks, full-media decode, actual-frame review, technical audio measurements, and independent automatic speech recognition. No listening/prosody certification is claimed. Any changed speech requires regeneration, timing rebuild, rerender and fresh checks. Browser playback is not independently exercised; the static-first viewer includes native controls and a complete no-motion transcript.
