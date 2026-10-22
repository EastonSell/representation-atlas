"""Offline synthetic tests: never instantiate or execute pretrained weights."""
import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch

BLOCKS = (0, 5, 10, 14, 19, 24, 29)

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tail = object()
        self.last = None
    def forward(self, x):
        y = x + 0.015 * x.cumsum(1) + 0.002
        self.last = y
        return (y, self.tail)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=30, hidden_size=576,
                                      vocab_size=49152, model_type='llama')
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(Block() for _ in range(30))
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.break_at = None
        self.repeat = False
        self.skip = False
        self.normed = None
    def forward(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs, torch.is_grad_enabled()))
        width = torch.arange(576, dtype=torch.float32)
        x = torch.sin(input_ids[..., None].float() * (width + 1) * .013)
        for i, block in enumerate(self.model.layers):
            if self.break_at == i:
                raise RuntimeError('synthetic failure')
            if self.skip and i == 14:
                continue
            out = block(x)
            assert out[1] is block.tail
            x = out[0]
            if self.repeat and i == 14:
                x = block(x)[0]
        self.normed = x / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-5)
        logits = self.normed[..., torch.arange(49152) % 576]
        return SimpleNamespace(logits=logits)

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def setUp(self):
        self.module = importlib.import_module('patchstudy.model_adapter')
        self.model = Toy()
        self.adapter = self.module.Adapter(self.model)
        self.a, self.b = [2, 3, 4, 5], [2, 9, 4, 5]
    def clean(self):
        self.assertTrue(all(not b._forward_hooks for b in self.model.model.layers))
    def test_capture_is_raw_not_final_normalized_and_readonly(self):
        plain = self.adapter.plain(self.a)
        logits, states = self.adapter.capture(self.a)
        np.testing.assert_array_equal(plain, logits)
        self.assertEqual(logits.shape, (49152,))
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(set(states), set(BLOCKS))
        for state in states.values():
            self.assertEqual(state.shape, (4,576))
            self.assertEqual(state.dtype, np.float32)
        np.testing.assert_array_equal(states[29], self.model.model.layers[29].last[0].numpy())
        self.assertFalse(np.allclose(states[29], self.model.normed[0].numpy()))
        states[29][:] = 0
        self.assertFalse(torch.all(self.model.model.layers[29].last == 0))
        for ids, kw, grad in self.model.calls:
            self.assertEqual(tuple(ids.shape), (1,4))
            self.assertFalse(kw['use_cache'])
            self.assertFalse(grad)
        self.clean()
    def test_all_controls_and_original_prefix(self):
        za, sa = self.adapter.capture(self.a)
        zb, sb = self.adapter.capture(self.b)
        for block in BLOCKS:
            np.testing.assert_allclose(self.adapter.patch(self.a,block,[1],sa[block]),za,rtol=0,atol=1e-6)
            np.testing.assert_allclose(self.adapter.patch(self.a,block,[0],sb[block]),za,rtol=0,atol=1e-6)
            np.testing.assert_allclose(self.adapter.patch(self.a,block,'all',sb[block]),zb,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(self.model.calls[-1][0].numpy(),[self.a])
        np.testing.assert_allclose(self.adapter.patch(self.a,29,[3],sb[29]),zb,rtol=0,atol=1e-6)
        np.testing.assert_allclose(self.adapter.patch(self.a,29,[1],sb[29]),za,rtol=0,atol=1e-6)
        self.assertFalse(np.allclose(self.adapter.patch(self.a,0,[1],sb[0]),za))
        self.assertFalse(np.allclose(self.adapter.patch(self.a,0,[2],sb[0]),zb))
        self.assertFalse(np.allclose(self.adapter.patch(self.a,5,'all',sb[0]),zb))
        self.clean()
    def test_realized_norm_is_observed_at_hook(self):
        _, sa = self.adapter.capture(self.a)
        _, sb = self.adapter.capture(self.b)
        self.adapter.patch(self.a,14,[1],sb[14])
        expected=float(np.linalg.norm(sb[14][[1]].astype('float64')-sa[14][[1]].astype('float64')))
        self.assertAlmostEqual(self.adapter.last_patch_diagnostics['realized_residual_delta_norm'],expected,places=10)
    def test_patch_clones_and_preserves_unselected_positions(self):
        _, sb = self.adapter.capture(self.b)
        donor = sb[14].copy()
        seen = []
        original = self.model.model.layers[15].forward
        def inspect(x):
            seen.append(x.clone())
            return original(x)
        self.model.model.layers[15].forward = inspect
        self.adapter.patch(self.a,14,[1],donor)
        raw = self.model.model.layers[14].last[0].numpy()
        np.testing.assert_array_equal(seen[0][0, [0,2,3]].numpy(),raw[[0,2,3]])
        np.testing.assert_array_equal(seen[0][0,1].numpy(),donor[1])
        self.assertFalse(np.array_equal(raw[1],donor[1]))
        np.testing.assert_array_equal(donor,sb[14])
        self.clean()
    def test_hook_errors_cleanup_and_exactly_once(self):
        _, states = self.adapter.capture(self.a)
        self.model.break_at=19
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            self.adapter.capture(self.a)
        self.clean()
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            self.adapter.patch(self.a,14,[1],states[14])
        self.clean()
        self.model.break_at=None
        for attribute in ('repeat','skip'):
            setattr(self.model,attribute,True)
            with self.assertRaises(RuntimeError): self.adapter.capture(self.a)
            self.clean()
            with self.assertRaises(RuntimeError): self.adapter.patch(self.a,14,[1],states[14])
            self.clean()
            setattr(self.model,attribute,False)
    def test_reject_bad_inputs_before_forward(self):
        _, states = self.adapter.capture(self.a)
        before=len(self.model.calls)
        for ids in ([], [[1,2]], [True,2], [1.5,2], [-1,2], [49152], list(range(33))):
            with self.assertRaises((ValueError,TypeError)): self.adapter.plain(ids)
        for block, pos, replacement in ((1,[1],states[14]),(14,[4],states[14]),(14,[],states[14]),(14,[1,1],states[14]),(14,[True],states[14]),(14,[1],states[14][:2]),(14,[1],states[14].astype('float64')),(14,[1],np.full((4,576),np.nan,dtype='float32'))):
            with self.assertRaises((ValueError,TypeError)): self.adapter.patch(self.a,block,pos,replacement)
        self.assertEqual(len(self.model.calls),before)
        self.clean()
    def test_cpu_float32_architecture_rejected(self):
        self.model.double()
        with self.assertRaises(ValueError): self.module.Adapter(self.model)
        self.model.float(); self.model.config.hidden_size=12
        with self.assertRaises(ValueError): self.module.Adapter(self.model)

class SnapshotTests(unittest.TestCase):
    def test_receipt_hashes_architecture_and_revision_without_loading(self):
        from patchstudy import model_adapter as m
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                          model_type='llama', architectures=['LlamaForCausalLM'])
            (root/'config.json').write_text(json.dumps(config))
            (root/'model.safetensors').write_bytes(b'FAKE: never load these synthetic bytes')
            receipt = dict(model_id=m.MODEL_REPO, revision=m.REVISION, files={})
            for p in root.iterdir():
                receipt['files'][p.name] = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            rp = root/'download-receipt.json'
            def save(): rp.write_text(json.dumps(receipt))
            save()
            self.assertEqual(m.validate_snapshot(root),root.resolve())
            receipt['revision']='incorrect'; save()
            with self.assertRaises(ValueError): m.validate_snapshot(root)
            receipt['revision']=m.REVISION; save()
            receipt['model_id']='another/model'; save()
            with self.assertRaises(ValueError): m.validate_snapshot(root)
            receipt['model_id']=m.MODEL_REPO; save()
            (root/'model.safetensors').write_bytes(b'tamper')
            with self.assertRaises(ValueError): m.validate_snapshot(root)
    def test_receipt_rejects_unbound_files_escape_and_wrong_architecture(self):
        from patchstudy import model_adapter as m
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def freeze(config):
                (root/'config.json').write_text(json.dumps(config))
                (root/'model.safetensors').write_bytes(b'fake')
                files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.iterdir() if p.name!='download-receipt.json'}
                return dict(model_id=m.MODEL_REPO,revision=m.REVISION,files=files)
            config=dict(num_hidden_layers=30,hidden_size=576,vocab_size=49152,model_type='llama',architectures=['LlamaForCausalLM'])
            receipt=freeze(config)
            rp=root/'download-receipt.json'
            rp.write_text(json.dumps(receipt))
            (root/'model.safetensors.index.json').write_text('{}')
            with self.assertRaises(ValueError): m.validate_snapshot(root)
            (root/'model.safetensors.index.json').unlink()
            receipt['files']['../escape']=dict(bytes=0,sha256='0'*64)
            rp.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): m.validate_snapshot(root)
            config['hidden_size']=12
            rp.write_text(json.dumps(freeze(config)))
            with self.assertRaises(ValueError): m.validate_snapshot(root)

if __name__ == '__main__': unittest.main()
