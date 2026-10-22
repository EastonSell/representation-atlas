import unittest
import numpy as np
from patchstudy.audit import independent_score, audit_archive
from patchstudy.metrics import score_logits
import tempfile

class AuditTests(unittest.TestCase):
    def test_independent_score_agrees_and_full_vector_sensitive(self):
        a=np.zeros(49152,dtype=np.float32); a[1]=5
        b=np.zeros_like(a); b[2]=5
        p=b.copy(); p[40000]=2
        expected=score_logits(p,a,b,1,2); actual=independent_score(p,a,b,1,2)
        for key in expected:
            self.assertAlmostEqual(actual[key],expected[key],places=12)
        self.assertNotEqual(independent_score(b,a,b,1,2)['donor_probability'],actual['donor_probability'])
    def test_missing_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            result=audit_archive(d)
            self.assertFalse(result['passed']); self.assertTrue(result['errors'])
if __name__=='__main__': unittest.main()

class AdversarialAuditTests(unittest.TestCase):
    def setUp(self):
        from patchstudy.audit import audit_records
        self.audit_records=audit_records
        self.rows=[dict(row_id=i,pair_id=i//2,orientation=i%2,donor_row_id=i^1,recipient_word=str(i),recipient_token_id=i+1,donor_token_id=(i^1)+1) for i in range(64)]
        self.base={}
        for i in range(64):
            z=np.zeros(49152,dtype=np.float32); z[i+1]=5; self.base[i]=z
    def test_nontarget_control_change_is_caught(self):
        a=self.base[0]; d=self.base[1]; p=a.copy(); p[40000]=.01
        rec=dict(row_id=0,block=0,condition='identity_word',logits=p,expected_identity='recipient',identity_max_abs_error=.01,**independent_score(p,a,d,1,2))
        result=self.audit_records(self.rows,self.base,[rec])
        self.assertTrue(any('Full-vocabulary' in e for e in result['errors']))
    def test_scalar_corruption_and_duplicate(self):
        a=self.base[0]; d=self.base[1]
        rec=dict(row_id=0,block=0,condition='donor_word',logits=d,expected_identity='none',identity_max_abs_error=None,**independent_score(d,a,d,1,2))
        rec['raw_transfer_delta']=123
        result=self.audit_records(self.rows,self.base,[rec,rec])
        self.assertTrue(any('Scalar score mismatch' in e for e in result['errors']))
        self.assertTrue(any('Duplicate' in e for e in result['errors']))
    def test_no_metrics_import(self):
        import inspect
        import patchstudy.audit as module
        self.assertNotIn('from .metrics',inspect.getsource(module))

class SummaryAuditTests(unittest.TestCase):
    def test_independent_full_summary_and_tamper(self):
        from test_metrics import FullPanelTests
        from patchstudy.audit import audit_records,_audit_summary
        import json
        from pathlib import Path
        FullPanelTests.setUpClass()
        records=[]
        for rec in FullPanelTests.records:
            r=FullPanelTests.rows[rec['row_id']]; a=FullPanelTests.base[r['row_id']]; b=FullPanelTests.base[r['donor_row_id']]
            cond=rec['condition']; block=rec['block']; identity='none'
            if cond.startswith('identity') or cond=='donor_prefix_negative' or (block==29 and cond=='donor_word'): identity='recipient'
            if cond=='donor_all' or (block==29 and cond=='donor_final'): identity='donor'
            records.append(dict(rec,expected_identity=identity,identity_max_abs_error=0. if identity!='none' else None,**independent_score(rec['logits'],a,b,r['recipient_token_id'],r['donor_token_id'])))
        result=audit_records(FullPanelTests.rows,FullPanelTests.base,records)
        self.assertTrue(result['passed'],result['errors'][:3])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'summary.json').write_text(json.dumps(FullPanelTests.summary))
            _audit_summary(root,result)
            self.assertEqual(result['errors'],[])
            broken=json.loads(json.dumps(FullPanelTests.summary)); broken['cells'][0]['genuine_switch']['denominator']=32
            (root/'summary.json').write_text(json.dumps(broken)); _audit_summary(root,result)
            self.assertTrue(any('denominator' in e for e in result['errors']))
