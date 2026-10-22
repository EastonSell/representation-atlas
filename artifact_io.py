"""Exclusive creation of new result files; never replace an archived artifact."""
import json
import os
from pathlib import Path


def check_fresh_output(path):
    path = Path(path)
    # lexists catches dangling symlinks as well as ordinary existing paths.
    if os.path.lexists(path):
        raise FileExistsError(f'Output already exists: {path}')
    return path


def write_new_json(path, value):
    path = check_fresh_output(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a check/write race from replacing a file.
    with path.open('x') as stream:
        stream.write(payload)
