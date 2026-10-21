"""Portable post-result independent evidence audit; tokenizer only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MODEL = Path(os.environ['SMOLLM_MODEL_PATH']) if os.environ.get('SMOLLM_MODEL_PATH') else None
FAMILIES = [('color', 'red', 'blue'), ('temperature', 'hot', 'cold'),
            ('state', 'open', 'closed'), ('size', 'large', 'small'),
            ('moisture', 'wet', 'dry'), ('shade', 'light', 'dark')]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def normalize(text):
    # Independent regular-expression implementation of frozen scoring semantics.
    return re.sub(r'\.+$', '', text.strip()).strip().lower()


def tokenizer():
    if MODEL is None: raise ValueError('Set SMOLLM_MODEL_PATH or use --model-path')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                         trust_remote_code=False)


def audit(data, tok):
    rows = data['rows']
    require(len(rows) == 24, 'Expected exactly 24 development prompts')
    groups = defaultdict(list)
    outcomes = []
    for row in rows:
        groups[row['pair']].append(row)
        rendered = tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                          tokenize=False, add_generation_prompt=True)
        require(row['chat_prompt'] == rendered, 'Chat prompt mismatch or answer appended')
        require(rendered.endswith('<|im_start|>assistant\n'), 'Wrong extraction prefix')
        prompt_ids = tok.encode(rendered, add_special_tokens=False)
        require(row['prompt_token_ids'] == prompt_ids, 'Prompt token IDs mismatch')
        candidate_ids = [tok.encode(value, add_special_tokens=False)
                         for value in row['candidates']]
        require(all(len(ids) == 1 for ids in candidate_ids), 'Candidate not one token')
        require(row['candidate_ids'] == candidate_ids, 'Candidate token IDs mismatch')
        for value, value_ids in zip(row['candidates'], candidate_ids):
            require(tok.encode(rendered + value, add_special_tokens=False) == prompt_ids + value_ids,
                    'Answer continuation changes prefix tokenization')
        logits = row['logits']
        require(len(logits) == 2 and all(math.isfinite(x) for x in logits), 'Invalid logits')
        chosen = row['candidates'][0 if logits[0] >= logits[1] else 1]
        correct = chosen == row['answer']
        require(row['constrained_answer'] == chosen, 'Saved choice disagrees with logits')
        require(row['constrained_correct'] == correct, 'Saved constrained correctness wrong')
        gen = row['generation_token_ids']
        require(0 < len(gen) <= 12, 'Generation exceeds frozen token budget')
        if tok.eos_token_id in gen:
            require(gen[-1] == tok.eos_token_id and gen.count(tok.eos_token_id) == 1,
                    'Generation continues after EOS')
        truncated = tok.eos_token_id not in gen
        decoded = tok.decode(gen, skip_special_tokens=True)
        require(decoded == row['generation'], 'Generation token IDs do not decode to text')
        require(truncated == row['truncated'], 'Truncation flag inconsistent with EOS evidence')
        free_exact = not truncated and normalize(decoded) == row['answer']
        require(row['free_exact'] == free_exact, 'Saved unrestricted exactness wrong')
        norms = row['preanswer_residual_norms']
        require(set(norms) == {'9', '19', '29'}, 'Wrong residual sites')
        require(all(math.isfinite(x) and x > 0 for x in norms.values()), 'Invalid norms')
        outcomes.append({'pair': row['pair'], 'answer': row['answer'], 'chosen': chosen,
                         'constrained_correct': correct, 'free_exact': free_exact,
                         'generation_tokens': len(gen), 'truncated': truncated,
                         'target_statement_correct': decoded ==
                         f"The {row['family']} of {row['target']} is {row['answer']}.",
                         'first_prompt_candidate_chosen':
                         chosen == row['candidates'][row['template']]})

    require(len(groups) == 12, 'Expected exactly 12 pairs')
    used_entities = []
    for fi, (family, a, b) in enumerate(FAMILIES):
        for order in (0, 1):
            members = groups[f'dev-{fi}-{order}']
            require(len(members) == 2, 'Pair is incomplete or duplicated')
            require(Counter(x['answer'] for x in members) == Counter([a, b]),
                    'Counterfactual answer values are not balanced')
            target = f'item {fi * 4 + order * 2 + 1}'
            other = f'item {fi * 4 + order * 2 + 2}'
            used_entities.extend([target, other])
            skeletons = []
            for row in members:
                require((row['family'], row['template'], row['target'], row['candidates']) ==
                        (family, order, target, [a, b]), 'Unexpected dataset metadata')
                fact = f'The {family} of {target} is {row["answer"]}.'
                distractor = f'The {family} of {other} is {b}.'
                require(row['prompt'].count(fact) == 1 and row['prompt'].count(distractor) == 1,
                        'Incorrect target fact or non-fixed distractor')
                skeletons.append(row['prompt'].replace(fact, '<TARGET_FACT>', 1))
                if order == 0:
                    expected = (fact + ' ' + distractor + f' What is the {family} of {target}? '
                                f'Answer with exactly one word: {a} or {b}.')
                else:
                    expected = ('Facts: ' + distractor + ' ' + fact +
                                f' Using these facts, give the {family} of {target}. Reply only {b} or {a}.')
                require(row['prompt'] == expected, 'Wording, fact order or candidate order mismatch')
            require(skeletons[0] == skeletons[1], 'Counterfactual changes more than target value')
    require(len(set(used_entities)) == 24, 'Entity pair reuse')
    by_pair = defaultdict(list)
    for row in outcomes:
        by_pair[row['pair']].append(row['constrained_correct'])
    constrained = sum(x['constrained_correct'] for x in outcomes)
    free = sum(x['free_exact'] for x in outcomes)
    both = sum(len(v) == 2 and all(v) for v in by_pair.values())
    passed = constrained >= 20 and free >= 15 and both >= 6
    require(data['tokenizer_valid'] is True, 'Tokenizer validity summary wrong')
    for key, value in [('constrained_correct', constrained), ('free_exact', free),
                       ('both_pair_correct', both), ('gate_pass', passed)]:
        require(data[key] == value, f'Saved {key} disagrees with independent recomputation')
    require(data['architecture']['python_indices'] == [9, 19, 29], 'Off-by-one layer index')
    require(data['architecture']['one_based_sites'] == [10, 20, 30], 'Wrong layer labels')
    config = json.loads((MODEL / 'config.json').read_text())
    require(config['num_hidden_layers'] == data['architecture']['blocks'] == 30, 'Block count wrong')
    require(config['hidden_size'] == data['architecture']['hidden_size'] == 576, 'Width wrong')
    require(data['peak_rss_bytes'] < 3 * 1024**3, 'Recorded peak RSS exceeds budget')
    require(data['elapsed_seconds'] < 3600, 'Recorded duration exceeds budget')
    require('NOT RUN' in data['causal_endpoints'], 'Causal endpoints not clearly unmeasured')
    return {'constrained_correct': constrained, 'constrained_denominator': 24,
            'free_exact': free, 'free_denominator': 24, 'both_pair_correct': both,
            'pair_denominator': 12, 'gate_pass': passed,
            'truncated_generations': sum(x['truncated'] for x in outcomes),
            'generation_length_counts': dict(Counter(x['generation_tokens'] for x in outcomes)),
            'correct_target_statements_descriptive_only': sum(x['target_statement_correct'] for x in outcomes),
            'first_prompt_option_chosen': sum(x['first_prompt_candidate_chosen'] for x in outcomes),
            'raw_row_recomputation': outcomes}


def main():
    global MODEL
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-path',default=os.environ.get('SMOLLM_MODEL_PATH'))
    parser.add_argument('--results',type=Path,default=ROOT/'results.json')
    args=parser.parse_args()
    if not args.model_path:parser.error('--model-path or SMOLLM_MODEL_PATH is required')
    MODEL=Path(args.model_path)
    data=json.loads(args.results.read_text())
    manifest=json.loads((ROOT/'MODEL_MANIFEST.json').read_text())
    require(data['model_manifest']==manifest,'Model manifest mismatch')
    for item in manifest['files']:
        require(digest(MODEL/item['name'])==item['sha256'],'Model hash mismatch: '+item['name'])
    summary=audit(data,tokenizer())
    print(json.dumps({k:v for k,v in summary.items() if k!='raw_row_recomputation'},indent=2))
if __name__=='__main__':main()
