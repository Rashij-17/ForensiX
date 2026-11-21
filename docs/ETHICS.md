# Ethics, limitations and responsible use

| Commitment | How it is implemented |
|------------|----------------------|
| Purpose limitation | Detection only; no generation code; contributions that aid forgery are rejected (CONTRIBUTING.md) |
| Calibrated language | Verdict bands (`decision:` in `config/params.yaml`): *no strong indication of manipulation* / *inconclusive* / *potentially manipulated*; a test forbids un-negated claims of proof or certainty in reports |
| Human oversight | Every result, report and the About page say outputs need review by a qualified analyst before evidentiary, journalistic or legal use |
| Privacy | Uploads are read from memory / a temp dir and deleted; retained: SHA-256, numeric features, result, case notes. A test checks no media file is left behind |
| Fairness across sources | Per-generator detection rates with intervals and low-support flags; imbalances are reported |
| Transparency about limits | Every report contains the limitations list and the measured robustness / generator context |
| Dataset licence | Must be confirmed and recorded by the team (see DATASET.md) |

## Known limitations
* Feature-based classical models detect statistical traces of generation; they can miss new generators and can be fooled by
  deliberate post-processing.
* Reliability falls on compressed, resized, re-recorded or noisy media; see `reports/robustness_*.csv`.
* The video split of the benchmark is small (200 clips), so video estimates carry wide uncertainty.
* Real media can be flagged and manipulated media can pass; the probability is not calibrated to a legal standard.
