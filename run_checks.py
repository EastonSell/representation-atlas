"""Aggregate offline checks with explicit optional tokenizer-verification coverage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from artifact_io import check_fresh_output, write_new_json

ROOT = Path(__file__).resolve().parent


def build_plan(model_path=None, model_python=None):
    plan = [
        {'name': 'root', 'cwd': str(ROOT), 'command': [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']},
        {'name': 'pretrained_artifacts', 'cwd': str(ROOT/'milestones/smollm-activations'),
         'command': [sys.executable, '-m', 'unittest', 'discover', '-v']},
    ]
    directory = str(ROOT/'milestones/preanswer-capability')
    if model_path is None:
        plan.append({'name': 'preanswer_core', 'cwd': directory,
                     'command': [sys.executable, '-m', 'unittest', '-v', 'test_gate', 'test_repairs']})
    else:
        python = model_python or sys.executable
        plan.extend([
            {'name': 'model_hashes', 'cwd': str(ROOT), 'command': [python, '-c',
                'import sys; from pathlib import Path; from benchmark_runner import verify_model; verify_model(Path(sys.argv[1]))', str(model_path)]},
            {'name': 'preanswer_full', 'cwd': directory, 'command': [python, '-m', 'unittest', 'discover', '-v']},
        ])
    return plan


def coverage_status(model_requested, passed):
    if not passed:
        return 'failed'
    return 'all_requested_passed' if model_requested else 'core_passed_optional_not_run'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, help='Enable pinned-file verification and all tokenizer-dependent tests; no generation')
    parser.add_argument('--model-python', help='Optional separate interpreter containing the model runtime')
    parser.add_argument('--require-model', action='store_true', help='Fail argument validation unless --model-path is provided')
    parser.add_argument('--plan', action='store_true', help='Print commands/coverage without executing tests')
    parser.add_argument('--report', type=Path, help='Fresh JSON summary file; never overwritten')
    args = parser.parse_args()
    if (args.require_model or args.model_python) and args.model_path is None:
        parser.error('--model-path is required with --require-model or --model-python')
    if args.report:
        check_fresh_output(args.report)
    model_path = args.model_path.resolve() if args.model_path else None
    model_python = args.model_python
    if model_python and ("/" in model_python or "\\" in model_python):
        model_python = str(Path(model_python).absolute())
    plan = build_plan(model_path, model_python)
    optional = ('requested: all 11 tokenizer-dependent evidence methods, preceded by pinned model-file hashes'
                if model_path else 'NOT RUN: 11 optional tokenizer-dependent evidence methods; supply --model-path to include them')
    if args.plan:
        print(json.dumps({'plan': plan, 'optional_model_verification': optional, 'tests_executed': False}, indent=2))
        return 0
    print(optional, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if model_path:
        env['SMOLLM_MODEL_PATH'] = str(model_path)
    results = []
    passed = True
    for check in plan:
        if not passed:
            results.append({'name': check['name'], 'status': 'not_run_due_to_prior_failure'})
            continue
        print('\nCHECK: ' + check['name'], flush=True)
        try:
            code = subprocess.run(check['command'], cwd=check['cwd'], env=env).returncode
        except OSError as exc:
            results.append({'name': check['name'], 'status': 'failed_to_start', 'error_type': type(exc).__name__})
            passed = False
            continue
        results.append({'name': check['name'], 'status': 'passed' if code == 0 else 'failed', 'returncode': code})
        passed = code == 0
    report = {'status': coverage_status(model_path is not None, passed),
              'optional_model_verification': optional, 'checks': results,
              'model_generation_performed': False}
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        write_new_json(args.report, report)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
