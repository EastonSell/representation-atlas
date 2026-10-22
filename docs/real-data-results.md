# Real-data learned representations

Three CPU-trained 32-unit ReLU MLPs used scikit-learn's locally bundled UCI Optical Recognition of Handwritten Digits subset. No data or weights were downloaded. The dataset contains 1,797 8×8 digit images. Input scaling is the fixed known pixel maximum, 16; every MLP, centroid, mean and PCA basis uses training examples only. A seeded stratified 70/30 split is recorded explicitly in `results/digits.json`.

Mean MLP test accuracy was **97.22%**. A separate nearest-centroid probe on its hidden activations achieved **95.49%** with centering only. Removing 1, 2, 4 or 8 leading PCs reduced this probe to **91.05%, 85.56%, 68.83%, 38.02%** respectively. Centering alone produced identical nearest-centroid predictions, as expected from translation invariance. These findings are direct measurements on a learned real-data representation, complementary to the synthetic failure control.

The 250-epoch training cap produced convergence warnings, saved per seed. We report the capped models rather than claim optimizer convergence. No hyperparameters or component count were selected on these test results. MLP accuracy and probe accuracy are distinct outcomes. Seeded split replicates overlap and are not independent datasets. Writer IDs are not available in this local subset, so this is not a writer-held-out benchmark or a reproduction of the original UCI train/test split. Digit classification does not establish language-model steering efficacy or brain correspondence.

## Reproduce
```
OPENBLAS_NUM_THREADS=1 python run_digits.py --out /tmp/representation-digits-new.json
python -m unittest discover -s tests -v
```
The NumPy/scikit-learn versions, dataset hash, model configuration, indices and per-seed warnings are recorded. Use a fresh output path for each run; existing paths are refused rather than overwritten. Timing fields can vary. No serialized third-party models are bundled.

## Attribution and license
Optical Recognition of Handwritten Digits, donated to UCI in 1998, attributed to E. Alpaydin and C. Kaynak. UCI currently licenses the dataset under **CC BY 4.0**: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits . License: https://creativecommons.org/licenses/by/4.0/ . This experiment divides pixels by 16 and makes new seeded splits; raw samples are not redistributed in this repository.
Scikit-learn documents that `load_digits` includes the 1,797-example original test subset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html . Scikit-learn is an external BSD-3-Clause dependency.
