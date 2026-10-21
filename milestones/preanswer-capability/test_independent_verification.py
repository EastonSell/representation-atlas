"""Evidence corruption tests for the portable release; no expected failures."""
import ast
import copy
import json
import unittest
from verify_gate_independent import ROOT, audit, normalize, tokenizer


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'results.json').read_text())
        cls.tok = tokenizer()

    def check_mutation_rejected(self, mutate):
        data = copy.deepcopy(self.data)
        mutate(data)
        with self.assertRaises(AssertionError):
            audit(data, self.tok)

    def test_saved_evidence_recomputes(self):
        result = audit(self.data, self.tok)
        self.assertEqual((result['constrained_correct'], result['free_exact'],
                          result['both_pair_correct'], result['gate_pass']), (14, 0, 2, False))

    def test_corrupt_logits_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'][0].update(logits=[-1000, 1000]))

    def test_nonfinite_logits_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'][0].update(logits=[float('nan'), 1]))

    def test_answer_appended_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'][0].update(chat_prompt=d['rows'][0]['chat_prompt'] + 'red'))

    def test_eos_missing_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'][0].update(generation_token_ids=d['rows'][0]['generation_token_ids'][:-1]))

    def test_tokens_after_eos_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'][0].update(generation_token_ids=d['rows'][0]['generation_token_ids'] + [504]))

    def test_cap_exceeded_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'][0].update(generation_token_ids=[504] * 13))

    def test_wrong_layer_rejected(self):
        self.check_mutation_rejected(lambda d: d['architecture'].update(python_indices=[10, 20, 30]))

    def test_duplicate_pair_member_rejected(self):
        self.check_mutation_rejected(lambda d: d['rows'].__setitem__(1, copy.deepcopy(d['rows'][0])))

    def test_fabricated_success_rejected(self):
        self.check_mutation_rejected(lambda d: d.update(gate_pass=True))

    def test_parser_boundary_cases(self):
        for good in ['red', ' Red. ', 'RED...', '\nred\t']:
            self.assertEqual(normalize(good), 'red')
        for bad in ['red or blue', 'red!', 'red?', 'red,', 'The answer is red.', 'red\nblue']:
            self.assertNotEqual(normalize(bad), 'red')


if __name__=='__main__':unittest.main()
