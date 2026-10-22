"""No scientific tuning options. Unknown or changed values are rejected."""
BLOCKS=(0,5,10,14,19,24,29)
CONDITIONS=('donor_word','donor_final','donor_prefix_negative','identity_word','identity_final','donor_all')
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
SETTINGS=dict(schema_version=1,blocks=list(BLOCKS),conditions=list(CONDITIONS),n_words=64,n_pairs=32,vocab_size=49152,width=576,layers=30,identity_absolute_tolerance=1e-6,greedy_correct_min=58,positive_margin_min=58,jointly_competent_pairs_min=27,normalization_min_denominator=1.,bootstrap_seed=20261022,bootstrap_replicates=10000,model_revision=REVISION,torch='2.5.1+cpu',transformers='4.46.3',numpy='2.3.5')
BUDGET=dict(schema_version=1,max_attempted_forwards=2816,baseline_forward_cap=128,patch_forward_cap=2688,baseline_wall_seconds=300,patch_wall_seconds=2100,execution_wall_seconds=2400,termination_grace_seconds=15,max_rss_bytes=3221225472,max_output_bytes=1073741824,max_prefix_tokens=32,cpu_threads=2,paid_compute_limit=0)
def validate(value):
 if value!=SETTINGS:raise ValueError('Configuration changed or unknown key; separately versioned amendment required')
 return value
def plan(stage):
 for i in range(64):
  for b,c in ((None,'untouched'),(None,'capture_check')) if stage=='baseline' else ((b,c) for b in BLOCKS for c in CONDITIONS):
   yield dict(row_id=i,pair_id=i//2,orientation=i%2,block=b,condition=c)
