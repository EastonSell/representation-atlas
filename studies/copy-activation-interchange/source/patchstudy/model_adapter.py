"""Exact raw decoder-output interventions; imports never load model weights.

Each public forward starts from the supplied original prefix, without KV caches.
The optional loader is deliberately separate from the synthetic-test adapter.
"""
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import json
import hashlib
import numbers
import numpy as np
import torch

BLOCKS = (0, 5, 10, 14, 19, 24, 29)
REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
MODEL_REPO = 'HuggingFaceTB/SmolLM2-135M-Instruct'
PINNED_VERSIONS = {'torch': '2.5.1+cpu', 'transformers': '4.46.3', 'numpy': '2.3.5'}


def _architecture(config):
    for name, expected in {'num_hidden_layers':30, 'hidden_size':576,
                           'vocab_size':49152, 'model_type':'llama'}.items():
        actual = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
        if actual != expected:
            raise ValueError(f'Unsupported architecture {name}={actual!r}; expected {expected!r}')


def validate_snapshot(path):
    """Verify pinned receipt and every local file without importing Transformers.

    Receipt integrity is bound externally by the reviewed model manifest. A
    matching self-authored receipt alone cannot authenticate a remote revision.
    """
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('A local model snapshot directory is required')
    receipt = json.loads((root / 'download-receipt.json').read_text())
    if receipt.get('model_id') != MODEL_REPO or receipt.get('revision') != REVISION:
        raise ValueError('Receipt does not identify the pinned model and revision')
    entries = receipt.get('files')
    if not isinstance(entries,dict) or 'config.json' not in entries or 'model.safetensors' not in entries:
        raise ValueError('Receipt must bind configuration and model safetensors')
    for name, expected in entries.items():
        rel = Path(name)
        target = (root / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or root not in target.parents or not target.is_file():
            raise ValueError('Invalid receipt file path: ' + name)
        if target.stat().st_size != expected['bytes']:
            raise ValueError('Receipt byte mismatch: ' + name)
        digest = hashlib.sha256()
        with target.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError('Receipt digest mismatch: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'download-receipt.json'}
    if actual != set(entries):
        raise ValueError('Snapshot contains unbound or missing files')
    config = json.loads((root / 'config.json').read_text())
    _architecture(config)
    if config.get('architectures') != ['LlamaForCausalLM']:
        raise ValueError('Expected LlamaForCausalLM')
    return root


def load_model(path):
    """Explicit, offline, safetensors-only loader; never called on import.

    The caller must freeze and verify the receipt itself against its reviewed
    provenance manifest. Validation does not perform an inference forward.
    """
    for package, expected in PINNED_VERSIONS.items():
        if version(package) != expected:
            raise RuntimeError(f'{package} must be pinned to {expected}')
    root = validate_snapshot(path)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32,
        attn_implementation='eager',
    )
    if model.__class__.__name__ != 'LlamaForCausalLM':
        raise ValueError('Loaded unexpected model class')
    model.eval()
    Adapter(model)  # validate CPU/f32, modules and configuration; no forward
    return model


class Adapter:
    """Batch-one CPU float32 adapter, also usable with architecture-shaped toys."""
    def __init__(self, model):
        self.model = model
        self._validate_model()
        self.model.eval()
        self._active = False

    def _validate_model(self):
        _architecture(self.model.config)
        if len(self.model.model.layers) != 30:
            raise ValueError('Expected exactly 30 decoder blocks')
        for tensor in (*self.model.parameters(), *self.model.buffers()):
            if tensor.device.type != 'cpu' or (tensor.is_floating_point() and tensor.dtype != torch.float32):
                raise ValueError('Model must already be CPU float32')

    @staticmethod
    def _ids(ids):
        a = np.asarray(ids)
        if a.ndim != 1 or not 1 <= len(a) <= 32 or a.dtype.kind not in 'iu':
            raise ValueError('Expected 1..32 unpadded integer token IDs')
        # np.asarray would otherwise silently coerce mixed booleans to integers.
        if any(isinstance(i, (bool, np.bool_)) for i in ids):
            raise ValueError('Boolean token IDs are invalid')
        if np.any(a < 0) or np.any(a >= 49152):
            raise ValueError('Token ID out of vocabulary')
        return torch.tensor(a.astype(np.int64), dtype=torch.long).unsqueeze(0)

    @staticmethod
    def _raw(output, seq):
        if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
            raise RuntimeError('Expected decoder tuple with raw residual at element zero')
        raw = output[0]
        if raw.shape != (1,seq,576) or raw.dtype != torch.float32 or raw.device.type != 'cpu':
            raise RuntimeError('Unexpected raw decoder output shape/device/dtype')
        if not torch.isfinite(raw).all():
            raise RuntimeError('Nonfinite raw decoder output')
        return raw

    @contextmanager
    def _hooks(self, callbacks):
        handles = []
        counts = {block:0 for block in callbacks}
        try:
            for block, callback in callbacks.items():
                def hook(module, args, output, block=block, callback=callback):
                    counts[block] += 1
                    if counts[block] != 1:
                        raise RuntimeError(f'Block {block} invoked more than once')
                    return callback(output)
                handles.append(self.model.model.layers[block].register_forward_hook(hook))
            yield
            if any(n != 1 for n in counts.values()):
                raise RuntimeError('Each selected block must execute exactly once')
        finally:
            for handle in handles:
                handle.remove()

    def _forward(self, input_ids, callbacks):
        self._validate_model()
        if self._active:
            raise RuntimeError('Adapter does not permit nested forwards')
        if self.model.training:
            raise ValueError('Model must remain in evaluation mode')
        self._active = True
        try:
            with torch.inference_mode(), self._hooks(callbacks):
                output = self.model(input_ids=input_ids, use_cache=False,
                                    output_hidden_states=False, output_attentions=False,
                                    return_dict=True)
                logits = output.logits
                if logits.shape != (1,input_ids.shape[1],49152) or logits.dtype != torch.float32 or logits.device.type != 'cpu':
                    raise RuntimeError('Unexpected logits shape/device/dtype')
                final = logits[0,-1].detach().clone().numpy().copy()
                if not np.isfinite(final).all():
                    raise RuntimeError('Nonfinite final logits')
            return final
        finally:
            self._active = False

    def plain(self, ids):
        return self._forward(self._ids(ids), {})

    def capture(self, ids):
        input_ids = self._ids(ids)
        seq = input_ids.shape[1]
        states = {}
        def capture_block(block):
            def capture(output):
                states[block] = self._raw(output,seq)[0].detach().clone().numpy().copy()
                return None  # preserve output object exactly, including tuple tail
            return capture
        logits = self._forward(input_ids, {b:capture_block(b) for b in BLOCKS})
        return logits, states

    def patch(self, ids, block, positions, replacement):
        input_ids = self._ids(ids)
        seq = input_ids.shape[1]
        if isinstance(block, (bool,np.bool_)) or not isinstance(block,numbers.Integral) or block not in BLOCKS:
            raise ValueError('Block must be one of the frozen intervention blocks')
        if isinstance(positions,str) and positions == 'all':
            selected = list(range(seq))
        elif isinstance(positions,list) and positions and all(isinstance(p,numbers.Integral) and not isinstance(p,(bool,np.bool_)) for p in positions):
            selected = positions.copy()
        else:
            raise ValueError("Positions must be a nonempty integer list or 'all'")
        if len(set(selected)) != len(selected) or min(selected)<0 or max(selected)>=seq:
            raise ValueError('Invalid or repeated patch positions')
        replacement = np.asarray(replacement)
        if replacement.shape != (seq,576) or replacement.dtype != np.float32 or not np.isfinite(replacement).all():
            raise ValueError('Replacement must be a finite full float32 baseline state [seq,576]')
        donor = torch.from_numpy(replacement.copy())
        self.last_patch_diagnostics = None
        def replace(output):
            raw = self._raw(output,seq)
            patched = raw.clone()
            patched[0,selected,:] = donor[selected,:]
            actual = patched[0,selected,:].double().numpy()-raw[0,selected,:].double().numpy()
            self.last_patch_diagnostics = dict(realized_residual_delta_norm=float(np.linalg.norm(actual)))
            return (patched,) + output[1:]
        return self._forward(input_ids, {int(block):replace})
