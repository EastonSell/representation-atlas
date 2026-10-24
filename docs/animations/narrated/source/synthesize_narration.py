#!/usr/bin/env python3
"""Synthesize measured, resumable stock-voice narration with local Kokoro ONNX.

Example:
  python synthesize_narration.py narration.json --model-dir ./models \
      --output-dir ./audio --voice am_michael --speed 0.90

The manifest is a nonempty list, or {"segments": [...]}. Every segment needs
unique string "id" and clean transcript "text". Optional "tts_text" controls
pronunciation while preserving the transcript. Segment overrides: voice,
language, speed (0.5..2.0), lead_in, tail_out (nonnegative seconds).

Models are never downloaded. Expected local files: kokoro-v1.0.onnx and
voices-v1.0.bin; their names can be overridden. Dependencies are imported only
when synthesis starts, so --help and manifest validation use the stdlib.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time


CACHE_VERSION = 1
WINDOWS_RESERVED_IDS = {"con", "prn", "aux", "nul", "clock$"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    tts_text: str
    voice: str
    language: str
    speed: float
    lead_in: float
    tail_out: float


def number(value, label: str, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{label} must be finite and {limit}")
    return value


def nonempty_string(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def validate_manifest(data, args) -> list[Segment]:
    """Validate the entire input before creating an output directory or file."""
    rows = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty list or an object with a nonempty segments list")
    number(args.speed, "--speed", minimum=0.5, maximum=2.0)
    number(args.lead_in, "--lead-in", minimum=0)
    number(args.tail_out, "--tail-out", minimum=0)
    if args.threads < 1:
        raise ValueError("--threads must be a positive integer")
    seen = set()
    result = []
    for index, row in enumerate(rows, 1):
        label = f"Segment {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        sid = nonempty_string(row.get("id"), f"{label} id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", sid) or sid.casefold() in WINDOWS_RESERVED_IDS:
            raise ValueError(f"{label} has an unsafe id {sid!r}; use 1..80 letters, digits, underscores or hyphens")
        if sid.casefold() in seen:
            raise ValueError(f"Duplicate segment id {sid!r} (ids must also differ on case-insensitive filesystems)")
        seen.add(sid.casefold())
        text = nonempty_string(row.get("text"), f"{label} text")
        tts_text = nonempty_string(row.get("tts_text", text), f"{label} tts_text")
        voice = nonempty_string(row.get("voice", args.voice), f"{label} voice")
        language = nonempty_string(row.get("language", args.language), f"{label} language")
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*", language):
            raise ValueError(f"{label} language must be an eSpeak language code, such as en-us")
        result.append(Segment(
            sid, text, tts_text, voice, language,
            number(row.get("speed", args.speed), f"{label} speed", minimum=0.5, maximum=2.0),
            number(row.get("lead_in", args.lead_in), f"{label} lead_in", minimum=0),
            number(row.get("tail_out", args.tail_out), f"{label} tail_out", minimum=0),
        ))
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_cache(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def run(args) -> None:
    data = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    segments = validate_manifest(data, args)
    model = (args.model_dir / args.model_file).resolve()
    voices = (args.model_dir / args.voices_file).resolve()
    for path in (model, voices):
        if not path.is_file():
            raise ValueError(f"Local model asset not found: {path}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {args.output_dir}")
    try:
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
        from kokoro_onnx import Kokoro
    except ImportError as exc:
        raise RuntimeError("Install synthesis dependencies in a dedicated environment: "
                           "python -m pip install kokoro-onnx soundfile") from exc
    fingerprint = {
        "model_sha256": sha256_file(model),
        "voices_sha256": sha256_file(voices),
        "kokoro_onnx_version": importlib.metadata.version("kokoro-onnx"),
        "cache_version": CACHE_VERSION,
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    engine = Kokoro.from_session(session, str(voices))
    available = set(engine.get_voices())
    missing = sorted({segment.voice for segment in segments} - available)
    if missing:
        raise ValueError(f"Unknown voice(s) {', '.join(missing)}; available: {', '.join(sorted(available))}")
    # All manifest and voice validation has finished before any output writes.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    cursor = 0.0
    for segment in segments:
        target = args.output_dir / f"{segment.id}.wav"
        sidecar = args.output_dir / f"{segment.id}.cache.json"
        params = {"tts_text": segment.tts_text, "voice": segment.voice,
                  "language": segment.language, "speed": segment.speed,
                  "lead_in": segment.lead_in, "tail_out": segment.tail_out}
        key = hashlib.sha256(json.dumps({"parameters": params, "engine": fingerprint},
                                       sort_keys=True).encode("utf-8")).hexdigest()
        cached = read_cache(sidecar)
        reuse = (not args.overwrite and target.is_file() and cached.get("cache_key") == key
                 and cached.get("audio_sha256") == sha256_file(target))
        started = time.perf_counter()
        if not reuse:
            samples, sample_rate = engine.create(segment.tts_text, voice=segment.voice,
                                                  speed=segment.speed, lang=segment.language, trim=True)
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            if not samples.size or not np.isfinite(samples).all():
                raise RuntimeError(f"Segment {segment.id}: synthesis returned empty or nonfinite audio")
            fade = min(round(0.008 * sample_rate), len(samples) // 2)
            if fade:
                samples[:fade] *= np.linspace(0, 1, fade)
                samples[-fade:] *= np.linspace(1, 0, fade)
            samples = np.concatenate((np.zeros(round(segment.lead_in * sample_rate), np.float32),
                                      samples, np.zeros(round(segment.tail_out * sample_rate), np.float32)))
            peak = float(np.max(np.abs(samples)))
            if peak > 0.98:
                samples *= 0.98 / peak
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{segment.id}.",
                                                 suffix=".wav", delete=False) as handle:
                    temporary = Path(handle.name)
                sf.write(temporary, samples, sample_rate, format="WAV", subtype="PCM_16")
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            atomic_json(sidecar, {"cache_key": key, "audio_sha256": sha256_file(target),
                                  "parameters": params, "engine": fingerprint})
        info = sf.info(target)
        item = {"id": segment.id, "text": segment.text, "tts_text": segment.tts_text,
                "audio_path": str(target.resolve()), "audio_sha256": sha256_file(target),
                "duration": info.duration, "start": cursor, "end": cursor + info.duration,
                "sample_rate": info.samplerate, "voice": segment.voice, "language": segment.language,
                "speed": segment.speed, "lead_in": segment.lead_in, "tail_out": segment.tail_out,
                "status": "cached" if reuse else "generated"}
        results.append(item)
        cursor += info.duration
        atomic_json(args.output_dir / "timings.json", {"engine": fingerprint, "segments": results,
                                                      "total_duration": cursor})
        print(f"{segment.id}: {info.duration:.3f} s ({item['status']}; "
              f"{time.perf_counter() - started:.2f} s elapsed)", flush=True)
    print(f"Total: {cursor:.3f} s. Timings: {(args.output_dir / 'timings.json').resolve()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="JSON transcript and synthesis parameters")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory of already-downloaded ONNX and voice assets")
    parser.add_argument("--output-dir", type=Path, required=True, help="WAVs, cache sidecars and measured timings")
    parser.add_argument("--model-file", default="kokoro-v1.0.onnx")
    parser.add_argument("--voices-file", default="voices-v1.0.bin")
    parser.add_argument("--voice", default="am_michael")
    parser.add_argument("--language", default="en-us")
    parser.add_argument("--speed", type=float, default=0.90)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lead-in", type=float, default=0.12)
    parser.add_argument("--tail-out", type=float, default=0.28)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even when a matching cache exists")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
