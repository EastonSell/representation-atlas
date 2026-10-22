"""Durable per-case records; incomplete runs never masquerade as complete."""
import hashlib, json, os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def reserve(path):
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    atomic(path / 'status.json', {'status': 'running'})
    return path


def check_coverage(rows, expected):
    keys = [(r['group_id'], r['register']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('Incomplete or duplicate validation coverage')


def summarize(rows):
    groups = {}
    for r in rows: groups.setdefault(r['group_id'], []).append(r)
    valid = [g for g in groups.values() if all(r['prediction'] is not None for r in g)]
    return {
        'records':len(groups), 'responses':len(rows),
        'strict_correct':sum(r['prediction'] == r['answer'] for r in rows),
        'format_coverage':sum(r['prediction'] is not None for r in rows),
        'exact_target_template_match':sum(r['generated'].strip() == r['target'] for r in rows),
        'all_registers_correct':sum(all(r['prediction'] == r['answer'] for r in g) for g in groups.values()),
        'valid_groups':len(valid), 'invalid_groups':len(groups)-len(valid),
        'valid_group_answer_change_rate':sum(len({r['prediction'] for r in g}) > 1 for g in valid)/len(valid) if valid else None,
        'answer_or_invalid_status_change_count':sum(len({r['prediction'] for r in g}) > 1 for g in groups.values())}


def verify_freeze(root):
    frozen = json.loads((root / 'FREEZE.json').read_text())
    for name, expected in frozen.items():
        if digest(root/name) != expected: raise ValueError('Source freeze mismatch: '+name)
    return frozen
