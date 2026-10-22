import unittest
import numpy as np
from patchstudy.metrics import score_logits, summarize, normalized_eligibility, bootstrap_groups

def logits(a=0., b=0., other=-10.):
    z=np.full(49152, -10.,dtype=np.float32); z[1]=a; z[2]=b; z[3]=other; return z

class MetricsTests(unittest.TestCase):
    def test_orientation_switch_and_kl(self):
        b=logits(4,0); d=logits(0,4); p=logits(1,5)
        s=score_logits(p,b,d,1,2)
        self.assertEqual(s['raw_transfer_delta'],8)
        self.assertTrue(s['genuine_switch']); self.assertTrue(s['greedy_donor'])
        self.assertGreaterEqual(s['kl_baseline_to_patch'],0)
        self.assertEqual(score_logits(b,b,d,1,2)['kl_baseline_to_patch'],0)
    def test_replacement_not_switch_and_offtarget(self):
        s=score_logits(logits(0,4),logits(0,1,5),logits(0,4),1,2)
        self.assertTrue(s['greedy_donor']); self.assertFalse(s['genuine_switch'])
        s=score_logits(logits(0,1,5),logits(0,4),logits(0,4),1,2)
        self.assertTrue(s['off_target_move'])
    def test_tie_and_validation(self):
        self.assertTrue(score_logits(logits(),logits(),logits(),1,2)['exact_pairwise_tie'])
        for z in [np.zeros(4,dtype=np.float32),logits().astype(np.float64),np.full(49152,np.nan,dtype=np.float32)]:
            with self.assertRaises(ValueError): score_logits(z,logits(),logits(),1,2)
    def test_bootstrap_pair_groups_reproducible_and_zero_variance(self):
        x=np.column_stack([np.arange(32),np.ones(32)])
        a=bootstrap_groups(x); b=bootstrap_groups(x)
        self.assertEqual(a,b); self.assertEqual(a['group_count'],32)
        self.assertTrue(a['zero_variance'][1]); self.assertEqual(a['pointwise_95'][1],[1.,1.])
        with self.assertRaises(ValueError): bootstrap_groups(np.zeros((64,2)))
    def test_normalization_pairwise_eligibility_no_clipping(self):
        rows=[dict(row_id='a',pair_id='p',orientation=0,donor_row_id='b',recipient_token_id=1,donor_token_id=2),dict(row_id='b',pair_id='p',orientation=1,donor_row_id='a',recipient_token_id=2,donor_token_id=1)]
        e=normalized_eligibility(rows,{'a':logits(4,0),'b':logits(0,4)})
        self.assertEqual(e['eligible_pair_ids'],['p']); self.assertEqual(e['denominators']['a'],8)
        e=normalized_eligibility(rows,{'a':logits(.2,0),'b':logits(0,.2)})
        self.assertEqual(e['eligible_pair_ids'],[])
        self.assertEqual(score_logits(logits(0,20),logits(4,0),logits(0,4),1,2)['raw_transfer_delta']/8,3)

if __name__=='__main__': unittest.main()

class FullPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from patchstudy.metrics import CELLS
        cls.rows=[]; cls.base={}; cls.records=[]
        for i in range(64):
            cls.rows.append(dict(row_id=i,pair_id=i//2,orientation=i%2,recipient_word='word'+str(i),donor_row_id=i^1,recipient_token_id=i+1,donor_token_id=(i^1)+1))
            z=np.zeros(49152,dtype=np.float32); z[i+1]=5; cls.base[i]=z
        for row in cls.rows:
            for b,c in CELLS:
                keep=c.startswith('identity') or c=='donor_prefix_negative' or (b==29 and c=='donor_word')
                cls.records.append(dict(row_id=row['row_id'],block=b,condition=c,logits=cls.base[row['row_id'] if keep else row['donor_row_id']]))
        cls.summary=summarize(cls.rows,cls.base,cls.records)
    def test_all_cells_and_denominators(self):
        s=self.summary
        self.assertTrue(s['complete']); self.assertEqual(len(s['cells']),42)
        self.assertEqual(len(s['normalized_eligible_pair_ids']),32)
        self.assertEqual(len(s['bootstrap']['simultaneous_95']),12)
        c=s['cells'][0]; self.assertEqual(c['genuine_switch']['denominator'],64)
        self.assertEqual(c['mean_raw_transfer_delta'],10)
        self.assertEqual(c['descriptive_metrics']['normalized_transfer']['mean'],1)
        self.assertEqual(len(c['pair_level_outcomes']),32)
    def test_partial_missing_and_duplicate_fail(self):
        partial=summarize(self.rows,self.base,self.records[:1])
        self.assertFalse(partial['complete']); self.assertEqual(len(partial['missing_cells']),2687)
        self.assertIsNone(partial['bootstrap'])
        with self.assertRaises(ValueError): summarize(self.rows,self.base,[self.records[0],self.records[0]])
