# Prospective validation-driver amendment

The [v1 source/hook checkpoint](README.md) and [training gate result](../copy-subspace-train-results/README.md) remain immutable. The new [v2 amendment](AMENDMENT_V2.md) fixes later-stage ordering and reporting before any validation/test model inference.

`driver_v2.py` is the prospective selection/test executable; `driver.py` remains the exact historical training executable. The new freeze is `SOURCE_FREEZE_V2.json`. All **22 milestone-local tests** pass, including six amendment regressions. Mock selection/test runs do not execute the pretrained model. No real validation/test or nonzero intervention has run at this source checkpoint.

V2 verifies and accepts only the exact reviewed v1 training archive for selection, copies fit bytes unchanged, checks the validation baseline before any intervention, separates margin-correct and greedy-correct groups, and records requested plus realized perturbation norms. It preserves all hypotheses, thresholds, data, layer/site, candidate k values and random seeds.

Review this source before selection; review the resulting selection digest before test. Expected maxima remain 1,300 validation forwards/900 cooperative seconds, then 460 test forwards/600 seconds for a nonzero selected k. No automatic stage transition exists.
