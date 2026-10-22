"""Frozen offline supplied-answer evaluation; no downloads or causal claims."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from arithmetic_benchmark import audit
from geometry import direction, fit_transform, transform

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'milestones/unordered-arithmetic-benchmark/dataset.json'
DATASET_SHA256 = '396381dcd0472c6f27a45d97732a758da0a6433c7e9100c42b5fb89ec36fe4b6'
MODEL_MANIFEST = ROOT / 'milestones/smollm-activations/MODEL_MANIFEST.json'
MODEL_MANIFEST_SHA256 = '34f007987ac97042a0f1f8278050af521edb868cec1ac083fe4c77ea93bf3ce2'
COMPONENTS = (0, 1, 2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_frozen_dataset():
    if digest(DATASET) != DATASET_SHA256:
        raise ValueError('Frozen dataset hash mismatch')
    data = json.loads(DATASET.read_text())
    audit(data)
    return data


def verify_model(path):
    if digest(MODEL_MANIFEST) != MODEL_MANIFEST_SHA256:
        raise ValueError('Pinned model manifest hash mismatch')
    manifest = json.loads(MODEL_MANIFEST.read_text())
    for item in manifest['files']:
        file = Path(path) / item['name']
        if file.stat().st_size != item['size'] or digest(file) != item['sha256']:
            raise ValueError('Pinned model file mismatch: ' + item['name'])
    return manifest


def validate_records(rows, records, activations=False):
    if len(records) != len(rows) or [r['id'] for r in records] != [r['id'] for r in rows]:
        raise ValueError('Backend must return every requested ID once, in order')
    width = None
    for record in records:
        for key in ('correct_log_likelihood', 'incorrect_log_likelihood'):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value > 1e-6:
                raise ValueError('Expected finite nonpositive candidate log likelihoods')
        if activations:
            positive = np.asarray(record.get('positive'), dtype=float)
            negative = np.asarray(record.get('negative'), dtype=float)
            if (positive.ndim != 1 or positive.size < 3 or positive.shape != negative.shape
                    or not np.isfinite(positive).all() or not np.isfinite(negative).all()):
                raise ValueError('Expected matched finite activation vectors of dimension at least three')
            if width is not None and positive.size != width:
                raise ValueError('Inconsistent activation dimensions')
            width = positive.size
    return records


def capability_gate(rows, records):
    validate_records(rows, records)
    if len(rows) != 60 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Gate must use only the frozen 60 training rows')
    groups = defaultdict(list)
    for row, record in zip(rows, records):
        groups[tuple(sorted((row['a'], row['b'])))].append(
            record['correct_log_likelihood'] > record['incorrect_log_likelihood'])
    if len(groups) != 33:
        raise ValueError('Expected 33 training unordered problems')
    correct = sum(sum(values) for values in groups.values())
    complete = sum(all(values) for values in groups.values())
    return {'passed': correct >= 45 and complete >= 24, 'strict_correct_rows': correct,
            'total_rows': 60, 'all_correct_unordered_groups': complete, 'total_groups': 33,
            'required_rows': 45, 'required_groups': 24, 'ties_count_as_failures': True,
            'scope': 'Supplied-answer conditional likelihood discrimination, not free generation'}


def arrays(records):
    return np.asarray([r['positive'] for r in records]), np.asarray([r['negative'] for r in records])


def margin_summary(margins):
    margins = np.asarray(margins)
    return {'wins': int(np.sum(margins > 0)), 'ties': int(np.sum(margins == 0)),
            'losses': int(np.sum(margins < 0)),
            'accuracy_half_ties': float(np.mean((margins > 0) + .5 * (margins == 0)))}


def evidence_records(records, margins=None):
    result = []
    for i, record in enumerate(records):
        item = {k: v for k, v in record.items() if k not in ('positive', 'negative')}
        item['likelihood_margin'] = record['correct_log_likelihood'] - record['incorrect_log_likelihood']
        if margins is not None:
            item['selected_geometry_margin'] = float(margins[i])
        result.append(item)
    return result


def run_evaluation(data, backend, output):
    """Gate first; test inference is impossible until validation selection is saved."""
    audit(data)
    if data != load_frozen_dataset():
        raise ValueError('Only the exact frozen dataset is accepted')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stage = 'training_gate'
    started = time.monotonic()
    try:
        rows = {s: [r for r in data['rows'] if r['split'] == s] for s in ('train', 'validation', 'test')}
        gate_records = validate_records(rows['train'], backend.evaluate(rows['train'], activations=False))
        gate = capability_gate(rows['train'], gate_records)
        save_json(output / 'gate.json', {**gate, 'records': evidence_records(gate_records)})
        if not gate['passed']:
            result = {'status': 'gate_failed', 'gate': gate, 'dataset_sha256': DATASET_SHA256,
                      'validation_evaluated': False, 'test_evaluated': False,
                      'geometry_fitted': False, 'causal_interventions': False}
        else:
            stage = 'training_and_validation'
            train = validate_records(rows['train'], backend.evaluate(rows['train'], activations=True), True)
            validation = validate_records(rows['validation'], backend.evaluate(rows['validation'], activations=True), True)
            for a, b in zip(gate_records, train):
                if any(abs(a[k] - b[k]) > 1e-4 for k in ('correct_log_likelihood', 'incorrect_log_likelihood')):
                    raise ValueError('Training likelihood changed between gate and extraction')
            tp, tn = arrays(train)
            vp, vn = arrays(validation)
            candidates = []
            states = {}
            for k in COMPONENTS:
                state = fit_transform(np.concatenate([tp, tn]), k)
                axis = direction(transform(tp, state), transform(tn, state))
                margins = (transform(vp, state) - transform(vn, state)) @ axis
                states[k] = (state, axis)
                candidates.append({'remove_top_pcs': k, **margin_summary(margins),
                                   'records': [{'id': r['id'], 'geometry_margin': float(m)} for r, m in zip(validation, margins)]})
            selected = max(candidates, key=lambda c: (c['accuracy_half_ties'], -c['remove_top_pcs']))['remove_top_pcs']
            state, axis = states[selected]
            selection = {'selected_remove_top_pcs': selected, 'tie_break': 'smallest component count',
                         'validation_candidates': candidates, 'mean': state[0].tolist(),
                         'removed_pcs': state[1].tolist(), 'direction': axis.tolist(),
                         'train_ids': [r['id'] for r in train], 'validation_records': evidence_records(validation)}
            save_json(output / 'selection.json', selection)
            np.savez(output / 'training_validation_activations.npz', train_positive=tp, train_negative=tn,
                     validation_positive=vp, validation_negative=vn)
            stage = 'selected_test_evaluation'
            test = validate_records(rows['test'], backend.evaluate(rows['test'], activations=True), True)
            ep, en = arrays(test)
            test_margins = (transform(ep, state) - transform(en, state)) @ axis
            train_margins = (transform(tp, state) - transform(tn, state)) @ axis
            np.savez(output / 'activations.npz', train_positive=tp, train_negative=tn,
                     validation_positive=vp, validation_negative=vn, test_positive=ep, test_negative=en)
            result = {'status': 'completed', 'dataset_sha256': DATASET_SHA256, 'gate': gate,
                      'selected_remove_top_pcs': selected, 'test_geometry': margin_summary(test_margins),
                      'test_likelihood': margin_summary([r['correct_log_likelihood'] - r['incorrect_log_likelihood'] for r in test]),
                      'train_records': evidence_records(train, train_margins),
                      'test_records': evidence_records(test, test_margins), 'causal_interventions': False}
        save_json(output / 'results.json', result)
        save_json(output / 'runtime.json', {
            'backend_type': type(backend).__name__, 'versions': getattr(backend, 'versions', {}),
            'device': 'cpu', 'threads': 2, 'attention_implementation': 'eager',
            'model_manifest_sha256': MODEL_MANIFEST_SHA256,
            'runner_sha256': digest(Path(__file__)), 'geometry_sha256': digest(ROOT / 'geometry.py'),
            'elapsed_evaluation_seconds': time.monotonic() - started})
        save_json(output / 'artifact_manifest.json', {'files': [
            {'path': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(output.iterdir()) if p.is_file()]})
        return result
    except Exception as exc:
        save_json(output / 'failure.json', {'status': 'incomplete', 'stage': stage, 'exception_type': type(exc).__name__})
        raise


def encode_candidates(tokenizer, rows):
    encoded = {}
    for row in rows:
        prefix = tokenizer.encode(row['prompt'], add_special_tokens=False)
        pair = [tokenizer.encode(row['prompt'] + ' ' + row[k], add_special_tokens=False) for k in ('correct', 'incorrect')]
        if (not prefix or len(pair[0]) != len(pair[1]) or len(pair[0]) > 256
                or any(len(ids) <= len(prefix) or ids[:len(prefix)] != prefix for ids in pair)):
            raise ValueError('Frozen dataset tokenizer mismatch at ' + row['id'])
        encoded[row['id']] = (prefix, pair)
    return encoded


class LocalModelBackend:
    """Pinned CPU adapter; tokenizer failures reject the dataset, never filter it."""
    def __init__(self, path, rows):
        self.manifest = verify_model(path)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        import transformers
        if torch.__version__.split('+')[0] != '2.5.1' or transformers.__version__ != '4.46.3':
            raise ValueError('Frozen runtime requires torch 2.5.1 and transformers 4.46.3')
        torch.set_num_threads(2)
        self.torch = torch
        self.versions = {'python': platform.python_version(), 'numpy': np.__version__,
                         'torch': torch.__version__, 'transformers': transformers.__version__}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, token=False)
        self.encoded = encode_candidates(self.tokenizer, rows)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, token=False,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager')
        self.model.eval()
        self.started = time.monotonic()

    def evaluate(self, rows, activations=False):
        records = []
        with self.torch.inference_mode():
            for row in rows:
                if time.monotonic() - self.started > 3600:
                    raise TimeoutError('One-hour inference budget exhausted')
                prefix, pair = self.encoded[row['id']]
                ids = self.torch.tensor(pair, dtype=self.torch.long, device='cpu')
                out = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                                 output_hidden_states=activations, use_cache=False, return_dict=True)
                logp = out.logits.log_softmax(-1)
                likelihood = [float(sum(logp[i, j-1, ids[i, j]].item() for j in range(len(prefix), len(pair[i])))) for i in range(2)]
                record = {'id': row['id'], 'correct_log_likelihood': likelihood[0], 'incorrect_log_likelihood': likelihood[1],
                          'prefix_token_ids': prefix, 'correct_suffix_token_ids': pair[0][len(prefix):],
                          'incorrect_suffix_token_ids': pair[1][len(prefix):]}
                if activations:
                    record['positive'] = out.hidden_states[-1][0, -1].cpu().numpy().copy().tolist()
                    record['negative'] = out.hidden_states[-1][1, -1].cpu().numpy().copy().tolist()
                records.append(record)
                del out, logp
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not exist; no overwrite/resume')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Output directory already exists')
    data = load_frozen_dataset()
    backend = LocalModelBackend(args.model_path, data['rows'])
    result = run_evaluation(data, backend, args.output_dir)
    print(result['status'])


if __name__ == '__main__':
    main()
