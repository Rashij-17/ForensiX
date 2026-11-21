# Methodology

All parameters live in `config/params.yaml`.

## 1. Dataset construction (`forensix/data/dataset.py`)
Files are scanned, labelled (folder names or `metadata.csv`), inspected (dimensions, duration, fps, MD5) and written to one
metadata table. **Splits are created up front** (70/15/15, seeded) and stratified by modality and class. Exact duplicates
(MD5) and near-duplicate images (64-bit difference hash, Hamming distance <= 2) are grouped; only the first member of a
group is kept (`status = duplicate`), and a group with conflicting labels is excluded. Hence no sample or near-duplicate
can leak across splits.

## 2. Preprocessing (`forensix/preprocess/`)
* **Image:** corruption check, resize to 256x256 (area / cubic interpolation), RGB colour space, float [0, 1].
* **Audio:** decode check, resample to 16 kHz mono, remove DC offset and peak-normalise, reject silent (< -55 dBFS), too-short
  (< 0.5 s) or non-finite recordings, cut long recordings into 4 s segments (<= 3) whose features are averaged.
* **Video:** corruption check, every 2nd frame (<= 24), resized to 128x128, float [0, 1], consecutive pairs prepared for
  temporal comparison. Clips yielding < 6 sampled frames are rejected.

## 3. EDA (`forensix/eda.py`)
Per modality, on the **training split only**: class-conditional histograms, box plots of the most discriminative features,
Spearman correlation matrix, effect-size chart; image: mean FFT of real vs fake and example grids; audio: waveform,
spectrogram and MFCC heatmap; video: sampled frames and temporal plots. Output: `reports/eda/<modality>/`.

## 4. Features (`forensix/features/`)
See the table in the README. Every feature has a plain-language description in `features/catalog.py`, used by the SHAP
explanations and reports. Non-finite values are replaced by 0 before modelling.

## 5. Statistical validation (`forensix/stats/analysis.py`)
For each feature (train split): descriptive statistics, Welch t-test, Mann-Whitney U, two-sample KS, **Benjamini-Hochberg FDR**
across features, Cohen's d, 95% Welch CI of the mean difference, IQR outlier share and Spearman correlation / redundant pairs.
`significant` means Mann-Whitney q < 0.05. Output: `reports/stats_<modality>.{csv,json}`.

## 6. Modelling (`forensix/models/train.py`)
Logistic Regression and SVM (RBF) as baselines, Random Forest and XGBoost as tree models, each applied independently
per modality. Linear / kernel models are wrapped with a `StandardScaler`. All four are fitted on the train split; the winner
is the highest **validation** F1 (ties broken by ROC AUC); 5-fold CV F1 on the training split is reported for stability.
Every candidate is logged to MLflow and registered as `<Modality> <Model> v<N>`.

## 7. Explainability (`forensix/explain/`)
`TreeExplainer` for Random Forest / XGBoost; a probability-space permutation explainer for Logistic Regression / SVM. For a
single prediction: ranked contributions with direction, size, plain-language meaning and where the value sits relative to
the real / fake class means, plus totals per indicator group. Additivity (`base + sum(SHAP) = output`) is unit-tested.
Grad-CAM is intentionally **not** implemented; it is reserved for a future CNN-based image extension.

## 8. Evaluation
Accuracy, precision, recall, F1, ROC AUC, average precision, confusion matrix, ROC and precision-recall curves on the test
split (fake = positive class).

## 9. Robustness (`forensix/robustness/`)
The production model is re-scored on degraded copies of the **test split**:

| Image | Audio | Video |
|-------|-------|-------|
| JPEG q=70, 30 | Background noise 20, 10 dB SNR | H.264 CRF 28, 38 |
| Resize x0.75, x0.5 | MP3 64, 32 kbps | Downscale x0.75, x0.5 |
| Blur sigma 1, 2 | Resample 11 025, 8 000 Hz | Frame rate /2, /3 |
| Noise sigma 4, 10 | | |

Reported: clean F1, degraded F1, absolute and relative drop, fake / real recall, and `n_failed`. When some samples cannot be
analysed after a transformation, clean F1 is recomputed on the same surviving subset so the comparison is like for like.

## 10. Generator-wise analysis (`forensix/models/generators.py`)
Per source: detection rate (recall of fakes) with a Wilson 95% interval, F1 against the full real test set, gap to the best
source and a low-support flag. The false-alarm rate on real media is reported alongside. Generators are used **only** to
check consistency, never to attribute a file to a generator.

## 11-12. Deployment and MLOps
See [API.md](API.md) and [MLOPS.md](MLOPS.md).
