#!/usr/bin/env python3
"""Retime a complete Manim movie to validated narration and export H.264/AAC.

Requires Python 3.10+ and ffmpeg/ffprobe on PATH; no Python packages.
Manifest: {"title": "...", "segments": [{"id": "01", "start": 0,
"end": 4, "text": "Canonical narration.", "chapter": "Optional chapter"}]}.
Timings: {"segments": [{"id": "01", "text": "Canonical narration.",
"audio_path": "audio/01.wav", "duration": 3.2, "lead_in": 0, "tail_out": 0}]}.
Optional tts_text must match in both documents when different from text.
WAVs must be nonempty mono PCM16, with one common AAC-compatible sample rate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
import wave


def run(args, log=None):
    result = subprocess.run([str(x) for x in args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"{args[0]} failed: {result.stderr[-2500:]}")
    return result


def number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def records(document, label):
    if not isinstance(document, dict) or not isinstance(document.get("segments"), list) or not document["segments"]:
        raise ValueError(f"{label} needs a nonempty segments list")
    indexed = {}
    for item in document["segments"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ValueError(f"{label} segment ids must be nonempty strings")
        if item["id"] in indexed:
            raise ValueError(f"duplicate id in {label}: {item['id']}")
        indexed[item["id"]] = item
    return indexed


def probe(args, path):
    return json.loads(run([args.ffprobe, "-v", "error", "-show_streams",
                           "-show_format", "-of", "json", path]).stdout)


def validate(args):
    for name in ("video", "manifest", "timings", "work_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    inputs = {args.video, args.manifest, args.timings}
    if args.output in inputs:
        raise ValueError("output must differ from every input; the source is preserved")
    if args.output.exists() and not args.overwrite:
        raise ValueError("output exists; choose another path or pass --overwrite")
    if args.output.is_dir():
        raise ValueError("--output must name a file, not a directory")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    for key in ("lead_in", "tail_out", "final_tail_out"):
        number(getattr(args, key), key)
    if not all(math.isfinite(x) for x in (args.target_i, args.target_tp, args.target_lra)):
        raise ValueError("loudness targets must be finite")
    if not (-70 <= args.target_i <= -5 and -9 <= args.target_tp <= 0 and 1 <= args.target_lra <= 50):
        raise ValueError("loudnorm targets require I in [-70,-5], TP in [-9,0], LRA in [1,50]")
    plan = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    timing = json.loads(args.timings.read_text(encoding="utf-8-sig"))
    planned, spoken = records(plan, "manifest"), records(timing, "timings")
    if planned.keys() != spoken.keys():
        raise ValueError("manifest and timings must contain exactly the same ids")
    info = probe(args, args.video)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("input has no video stream")
    rate = video.get("avg_frame_rate", "0/0")
    source_fps = Fraction(rate if rate != "0/0" else video.get("r_frame_rate", "0/0"))
    fps = Fraction(args.fps) if args.fps else source_fps
    if not (0 < fps <= 240) or fps.numerator > 1_000_000:
        raise ValueError("fps must be positive, at most 240, and have a practical rational timebase")
    duration = number(float(video.get("duration", info["format"]["duration"])), "video duration", True)
    dimensions = (int(video["width"]), int(video["height"]))
    if any(x <= 0 or x % 2 for x in dimensions):
        raise ValueError("H.264 yuv420p needs positive even source dimensions")
    timeline, wavs, prior, cursor, sample_cursor, sample_rate = [], [], 0.0, 0, 0, None
    for index, seg in enumerate(planned.values()):
        sid, meta = seg["id"], spoken[seg["id"]]
        start = number(seg.get("start"), f"{sid} start")
        end = number(seg.get("end"), f"{sid} end", True)
        if end <= start:
            raise ValueError(f"{sid}: zero-length or reversed source interval")
        if abs(start - prior) > 1e-6:
            raise ValueError(f"{sid}: source gap/overlap; segments must be contiguous from zero")
        final = index == len(planned) - 1
        tolerance = max(.001, .5 / float(source_fps))
        if end > duration + (tolerance if final else 1e-6):
            raise ValueError(f"{sid}: source interval out of range")
        if final:
            if abs(end - duration) > tolerance:
                raise ValueError("manifest must cover the entire input movie")
            end = duration  # Canonicalize an endpoint rounded to source-frame precision.
        text = seg.get("text")
        if not isinstance(text, str) or not text.strip() or meta.get("text") != text:
            raise ValueError(f"{sid}: missing or stale canonical text")
        if seg.get("tts_text", text) != meta.get("tts_text", meta["text"]):
            raise ValueError(f"{sid}: stale tts_text")
        if "chapter" in seg and not isinstance(seg["chapter"], str):
            raise ValueError(f"{sid}: chapter must be text")
        for key in ("duration", "start", "end", "lead_in", "tail_out"):
            if key in meta:
                number(meta[key], f"{sid} audio {key}", key == "duration")
        audio = Path(meta["audio_path"])
        audio = (args.timings.parent / audio).resolve() if not audio.is_absolute() else audio.resolve()
        inputs.add(audio)
        with wave.open(str(audio), "rb") as wav:
            sr, count = wav.getframerate(), wav.getnframes()
            if (wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (1, 2, "NONE"):
                raise ValueError(f"{sid}: WAV must be mono uncompressed PCM16")
            if sr not in (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000):
                raise ValueError(f"{sid}: WAV sample rate is not AAC-compatible")
            raw = wav.readframes(count)
        if not count or len(raw) != count * 2 or not any(raw):
            raise ValueError(f"{sid}: WAV is empty, truncated, or entirely silent")
        if sample_rate not in (None, sr):
            raise ValueError("all WAVs must have the same sample rate")
        sample_rate = sr
        speech_duration = count / sr
        if abs(number(meta.get("duration"), f"{sid} duration", True) - speech_duration) > max(1 / sr, 1e-6):
            raise ValueError(f"{sid}: timing duration disagrees with WAV samples")
        lead, tail = meta.get("lead_in", 0), meta.get("tail_out", 0)
        if lead + tail >= speech_duration:
            raise ValueError(f"{sid}: embedded lead/tail consume the speech window")
        if "start" in meta and "end" in meta and (meta["end"] <= meta["start"] or abs(meta["end"] - meta["start"] - speech_duration) > max(1 / sr, 1e-6)):
            raise ValueError(f"{sid}: inconsistent audio start/end")
        padding = args.final_tail_out if final else args.tail_out
        frames = math.ceil(max(end - start, speech_duration + args.lead_in + padding) * fps)
        sample_end = round(Fraction(cursor + frames, 1) * sr / fps)
        samples = sample_end - sample_cursor
        offset = round(args.lead_in * sr)
        if offset + count > samples:
            raise ValueError(f"{sid}: speech does not fit its planned samples")
        timeline.append({**seg, "start": start, "end": end, "audio_path": str(audio),
                         "speech_duration": speech_duration, "lead_in": lead, "tail_out": tail,
                         "frame_start": cursor, "frames": frames, "sample_start": sample_cursor,
                         "samples": samples, "audio_offset": offset / sr,
                         "target_start": float(cursor / fps), "target_end": float((cursor + frames) / fps)})
        wavs.append((raw, offset, samples))
        prior, cursor, sample_cursor = end, cursor + frames, sample_end
    reserved = {args.work_dir / n for n in ("timeline.json", "narration.md", "narration.approx.srt")}
    if args.output in inputs or reserved & inputs or args.output in reserved:
        raise ValueError("input/output paths collide with packaging artifacts")
    return plan, timeline, wavs, fps, sample_rate, dimensions, cursor, sample_cursor


def escaped(text):
    return str(text).replace("\\", "\\\\").replace("\n", " ").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")


def timestamp(value):
    ms = round(value * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def package(args):
    plan, timeline, wavs, fps, sr, dimensions, total_frames, total_samples = validate(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    stage = args.work_dir / ("run_" + uuid.uuid4().hex[:12]); stage.mkdir()
    title = plan.get("title", args.video.stem)
    data = {"title": title, "source_video": str(args.video), "fps": str(fps),
            "width": dimensions[0], "height": dimensions[1], "sample_rate": sr,
            "frames": total_frames, "samples": total_samples, "duration": float(total_frames / fps),
            "caption_timing": "approximate; not ASR aligned" if args.subtitles else None,
            "segments": timeline}
    (args.work_dir / "timeline.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", f"Duration: {data['duration']:.3f} seconds.", ""]
    for s in timeline:
        lines += [f"## {timestamp(s['target_start'])} — {s.get('chapter', s['id'])}", "", s["text"], ""]
    (args.work_dir / "narration.md").write_text("\n".join(lines), encoding="utf-8")
    master = stage / "narration_raw.wav"
    with wave.open(str(master), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sr)
        for raw, offset, samples in wavs:
            wav.writeframesraw(bytes(offset * 2) + raw + bytes((samples - offset - len(raw) // 2) * 2))

    def encode(index):
        s = timeline[index]; target = stage / f"segment_{index:04d}.mp4"
        scale = float(Fraction(s["frames"], 1) / fps) / (s["end"] - s["start"])
        # Trim source time BEFORE scaling timestamps; output -t would trim retimed time.
        filt = (f"trim=start={s['start']:.12f}:end={s['end']:.12f},setpts=(PTS-STARTPTS)*{scale:.15f},"
                f"fps={fps},tpad=stop_mode=clone:stop={s['frames']},trim=end_frame={s['frames']},setpts=PTS-STARTPTS")
        run([args.ffmpeg, "-y", "-v", "error", "-threads", "2", "-i", args.video, "-an", "-vf", filt,
             "-frames:v", s["frames"], "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-threads", "2", "-video_track_timescale", fps.numerator, target], stage / f"encode_{index:04d}.log")
        stream = probe(args, target)["streams"][0]
        if int(stream["nb_frames"]) != s["frames"] or (stream["width"], stream["height"]) != dimensions:
            raise ValueError(f"{s['id']}: encoded frames or dimensions differ from plan")
        return target

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        videos = list(pool.map(encode, range(len(timeline))))
    concat = stage / "concat.txt"
    concat.write_text("ffconcat version 1.0\n" + "".join("file '" + p.as_posix().replace("'", "'\\''") + "'\n" for p in videos), encoding="utf-8")
    silent = stage / "retimed_silent.mp4"
    run([args.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat, "-an", "-c:v", "copy", silent])
    base = f"loudnorm=I={args.target_i}:TP={args.target_tp}:LRA={args.target_lra}"
    measurement = run([args.ffmpeg, "-v", "info", "-i", master, "-af", base + ":print_format=json", "-f", "null", "-"], stage / "loudness_measure.log")
    loud = json.loads(re.findall(r'\{\s*"input_i"[\s\S]*?\}', measurement.stderr)[-1])
    if not all(math.isfinite(float(loud[k])) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
        raise ValueError("whole-master loudness measurement was not finite")
    (stage / "loudness_measure.json").write_text(json.dumps(loud, indent=2), encoding="utf-8")
    normal = base + ":" + ":".join(f"{key}={loud[value]}" for key, value in
        [("measured_I", "input_i"), ("measured_TP", "input_tp"), ("measured_LRA", "input_lra"), ("measured_thresh", "input_thresh"), ("offset", "target_offset")])
    normal += f":linear=true,aresample={sr},apad=whole_len={total_samples},atrim=end_sample={total_samples},asetpts=N/SR/TB"
    metadata = [";FFMETADATA1", "title=" + escaped(title)]
    if any(s.get("chapter") for s in timeline):
        chapters = []
        for s in timeline:
            chapter = s.get("chapter") or (chapters[-1][1] if chapters else title)
            if not chapters or chapters[-1][1] != chapter:
                chapters.append((s["frame_start"], chapter))
        for i, (start, chapter) in enumerate(chapters):
            end = chapters[i + 1][0] if i + 1 < len(chapters) else total_frames
            metadata += ["[CHAPTER]", f"TIMEBASE={fps.denominator}/{fps.numerator}", f"START={start}", f"END={end}", "title=" + escaped(chapter)]
    meta_path = stage / "metadata.txt"; meta_path.write_text("\n".join(metadata) + "\n", encoding="utf-8")
    command = [args.ffmpeg, "-y", "-v", "error", "-i", silent, "-i", master, "-f", "ffmetadata", "-i", meta_path]
    if args.subtitles:
        cues = []
        for s in timeline:
            sentences = re.split(r"(?<=[.!?])\s+", s["text"].strip())
            start = s["target_start"] + s["audio_offset"] + s["lead_in"]
            window = s["speech_duration"] - s["lead_in"] - s["tail_out"]
            weight = sum(len(x) for x in sentences)
            for sentence in sentences:
                end = start + window * len(sentence) / weight
                cues.append(f"{len(cues)+1}\n{timestamp(start)} --> {timestamp(end)}\n" + "\n".join(textwrap.wrap(sentence, 56)))
                start = end
        subtitle = args.work_dir / "narration.approx.srt"
        subtitle.write_text("\n\n".join(cues) + "\n", encoding="utf-8")
        command += ["-f", "srt", "-i", subtitle]
    command += ["-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "2", "-map_chapters", "2", "-c:v", "copy", "-af", normal, "-c:a", "aac", "-b:a", "192k", "-ar", sr]
    if args.subtitles:
        command += ["-map", "3:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "title=Approximate narration timing (not ASR aligned)", "-disposition:s:0", "0"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.stem + "." + uuid.uuid4().hex[:12] + ".tmp.mp4")
    try:
        run(command + ["-movflags", "+faststart", "-movie_timescale", sr, temporary], stage / "master_mux.log")
        final = probe(args, temporary)
        v = next(s for s in final["streams"] if s["codec_type"] == "video")
        a = next(s for s in final["streams"] if s["codec_type"] == "audio")
        audio_samples = Fraction(a["duration_ts"]) * Fraction(a["time_base"]) * sr
        if int(v["nb_frames"]) != total_frames or audio_samples != total_samples:
            raise ValueError("final frame count/audio duration differs from timeline")
        if args.overwrite:
            os.replace(temporary, args.output)
        else:
            with args.output.open("xb") as destination, temporary.open("rb") as source:
                shutil.copyfileobj(source, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Packaged {len(timeline)} segments, {total_frames} frames, {total_samples} samples: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("video", "manifest", "timings", "work-dir", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fps", help="Output FPS, e.g. 30 or 30000/1001; default: source average FPS")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subtitles", action="store_true", help="Add explicitly approximate sentence-weighted SRT and subtitle track; no ASR")
    for key, default in (("lead-in", .2), ("tail-out", .15), ("final-tail-out", .8), ("target-i", -16), ("target-tp", -1.5), ("target-lra", 8)):
        parser.add_argument("--" + key, type=float, default=default)
    parser.add_argument("--ffmpeg", default="ffmpeg"); parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        package(args)
    except (ValueError, OSError, KeyError, TypeError, StopIteration, ZeroDivisionError, wave.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
