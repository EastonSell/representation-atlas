"""Fresh output and explicit aggregate-check coverage contracts."""
import json
import hashlib
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_experiment
import run_digits
import run_checks
import artifact_io

ROOT = Path(__file__).resolve().parents[1]


class ReleaseSafetyTests(unittest.TestCase):
    def test_both_runners_refuse_existing_before_any_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'existing.json'
            out.write_text('keep me')
            with patch.object(run_experiment, 'paired', side_effect=AssertionError('must not compute')):
                with self.assertRaises(FileExistsError):
                    run_experiment.run(out)
            with patch.object(run_digits, 'load_digits', side_effect=AssertionError('must not load')):
                with self.assertRaises(FileExistsError):
                    run_digits.run(out)
            self.assertEqual(out.read_text(), 'keep me')

    def test_both_cli_require_explicit_out_and_leave_archives_unchanged(self):
        paths = [ROOT/'results/experiment.json', ROOT/'results/digits.json']
        before = [p.read_bytes() for p in paths]
        for script in ('run_experiment.py', 'run_digits.py'):
            run = subprocess.run([sys.executable, str(ROOT/script)], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn(b'--out', run.stderr)
        self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_small_synthetic_run_writes_only_fresh_target(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=Path(d)/'config.json'
            cfg.write_text(json.dumps({'seeds':[1], 'dimensions':[4], 'regimes':['low_signal_variance'], 'train_pairs':8, 'test_pairs':6, 'remove_top_pcs':[0,1]}))
            out=Path(d)/'nested/new.json'
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment.run(out, config_path=cfg)
            result=json.loads(out.read_text())
            self.assertEqual(len(result['rows']),2)
            with self.assertRaises(FileExistsError):run_experiment.run(out, config_path=cfg)

    def test_small_digits_run_writes_only_fresh_target(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=Path(d)/'config.json'
            original=json.loads((ROOT/'digits_config.json').read_text())
            original.update(seeds=[1],hidden_units=4,max_iter=1,remove_top_pcs=[0])
            cfg.write_text(json.dumps(original))
            out=Path(d)/'nested/digits.json'
            with contextlib.redirect_stdout(io.StringIO()):
                run_digits.run(out, config_path=cfg)
            self.assertEqual(len(json.loads(out.read_text())['rows']),1)

    def test_historical_runner_bytes_preserved_under_old_manifest_hashes(self):
        manifest=json.loads((ROOT/'MANIFEST.json').read_text())
        for name in ('run_experiment.py','run_digits.py'):
            expected=next(r['sha256'] for r in manifest['files'] if r['path']==name)
            historical=ROOT/'milestones/release-safety/historical'/name
            self.assertEqual(hashlib.sha256(historical.read_bytes()).hexdigest(),expected)

    def test_aggregate_plan_reports_optional_model_omission(self):
        plan=run_checks.build_plan()
        self.assertEqual([x['name'] for x in plan],['root','pretrained_artifacts','preanswer_core'])
        self.assertTrue(all(x['cwd'] for x in plan))
        self.assertEqual(run_checks.coverage_status(False, True),'core_passed_optional_not_run')
        self.assertEqual(run_checks.coverage_status(True, True),'all_requested_passed')
        self.assertEqual(run_checks.coverage_status(True, False),'failed')

    def test_requested_model_plan_verifies_hashes_before_tokenizer_suite(self):
        plan=run_checks.build_plan(model_path=Path('/example/model'), model_python='python-model')
        self.assertEqual([x['name'] for x in plan],['root','pretrained_artifacts','model_hashes','preanswer_full'])
        self.assertEqual(plan[-1]['command'][0],'python-model')
        self.assertIn('preanswer-capability',str(plan[-1]['cwd']))

    def test_model_interpreter_symlink_is_not_dereferenced(self):
        with tempfile.TemporaryDirectory() as d:
            link=Path(d)/'venv-python'
            link.symlink_to(sys.executable)
            result=subprocess.run([sys.executable,str(ROOT/'run_checks.py'),'--plan','--model-path','model','--model-python','./venv-python'],cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads(result.stdout)['plan']
            self.assertEqual(plan[-1]['command'][0],str(link))

    def test_exclusive_write_does_not_replace_racing_target(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'race.json'
            out.write_text('original')
            with patch.object(artifact_io,'check_fresh_output',return_value=out):
                with self.assertRaises(FileExistsError):artifact_io.write_new_json(out,{'new':True})
            self.assertEqual(out.read_text(),'original')

    def test_dangling_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'dangling.json'
            out.symlink_to(Path(d)/'absent')
            with self.assertRaises(FileExistsError):run_experiment.run(out)
            self.assertTrue(out.is_symlink())
            self.assertFalse((Path(d)/'absent').exists())

    def test_aggregate_missing_required_model_fails(self):
        run=subprocess.run([sys.executable,str(ROOT/'run_checks.py'),'--require-model','--plan'],capture_output=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn(b'--model-path',run.stderr)


if __name__=='__main__':unittest.main()
