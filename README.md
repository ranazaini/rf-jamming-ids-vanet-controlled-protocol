# Multi-Seed Re-Evaluation of RF Jamming Intrusion Detection in Vehicular Networks

Official repository (code and per-seed outputs) for the paper **"Protocol-Controlled Multi-Seed Re-Evaluation of RF Jamming Intrusion Detection in Vehicular Networks."**

![License](https://img.shields.io/badge/License-MIT-yellow) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

The study proposes and evaluates a hybrid ensemble intrusion detection system for RF jamming attacks in VANETs, combining **multi-scale BiLSTM + XGBoost + k-NN meta-learner**, benchmarked against RF-Stat, LSTM-RELU-IO-L1, and BERT WordPiece baselines, within a protocol-controlled, multi-seed statistical evaluation framework across two mobility regimes (**15 m/s** and **25 m/s**).

## Repository Structure

```
.
├── README.md
├── notebooks/
│   ├── Detector_15ms_FINAL_20seed___Komputasi_Task_1.ipynb   # Main detector + baselines, 15 m/s regime
│   ├── Detector_25ms_FINAL_20seed___Komputasi_Task_1.ipynb   # Main detector + baselines, 25 m/s regime
│   ├── t1-t11_20_seed/                                       # Supplementary analysis notebooks (T1–T11)
│   └── 01_… – 07_…                                           # Revision notebooks (see "Revision" section)
├── data/jamshield/     # JamShield CSV files + source/license note (used by notebook 05)
├── src/                # Revision code (pcmef_fixed.py, pcmef_v2_infra.py, pcmef_corrected_stats.py, jamshield_e59_counter.py)
├── baselines_15ms/     # RF-Stat, LSTM-RELU-IO-L1, BERT-WP per-seed results — 15 m/s
├── baselines_25ms/     # Same baselines — 25 m/s
├── seed_runs_15ms/     # Detector per-seed results, ablation, convergence — 15 m/s
└── seed_runs_25ms/     # Detector per-seed results, ablation, convergence — 25 m/s
```

> Note on naming: the `ms` suffix in folder/file names refers to the **mobility regime in meters per second (m/s)**, not milliseconds.

## Dataset

This work uses the public **RF Jamming Dataset for Vehicular Wireless Networks** by Kosmanos et al.

| File | Speed Regime | Used Rows (filter) |
|---|---|---|
| `Dataset_1.csv` | 25 m/s | Speed ∈ [23, 27] m/s |
| `Dataset_2.csv` | 15 m/s | Speed ∈ [13, 17] m/s |

Columns used: `SNR`, `RSSI`, `PDR`, `Speed`, `Relative_Speed`, `Scenario`, `Time`

Source:[IEEE DataPort](https://ieee-dataport.org/documents/rf-jamming-dataset-vehicular-wireless-networks)

## Method Overview

- **Labels (3-class):** `Scenario 1 → Interference`, `Scenario 2 → Reactive`, `Scenario 3 → Constant`
- **Pseudo-episodes:** rows grouped into 60-second episodes (`floor(Time / 60)`)
- **Split:** stratified, episode-based (60% train / 20% val / 20% test), per-class shuffle with a fixed `RandomState(seed)`
- **Windowing:** multi-scale — 1s, 5s, and 15s non-overlapping windows within each episode
- **Normalisation:** `StandardScaler` fit on the training set only, applied to val/test
- **Detector model:** multi-scale BiLSTM (1s/5s/15s branches) + tabular XGBoost features, fused via a k-NN meta-learner
- **Baselines:** RF + statistical tabular features (Kosmanos et al.), LSTM-RELU-IO-L1 (Murshed et al.), BERT WordPiece / DistilBERT (Wickramasurendra et al.)
- **Leakage audit:** gap-split re-evaluation to check for adjacency leakage between train/test episodes

## Main Results (mean ± std, n = 20 seeds)

| Speed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 15 m/s | 87.76 ± 3.20 % | 83.24 ± 8.03 % | 82.54 ± 11.47 % | 81.84 ± 9.56 % |
| 25 m/s | 99.37 ± 0.85 % | 97.58 ± 7.44 % | 97.74 ± 7.81 % | 97.63 ± 7.61 % |

*Computed directly from `seed_runs_15ms/summary.json` and `seed_runs_25ms/summary.json` (20 seeds each). Note the wider precision/recall/F1 std at 25 m/s — this is likely driven by a small number of seeds where one class had few or no test episodes; cross-check against `leakage_audit_25ms.json` and per-class metrics before reporting.*

## Reproducibility Protocol

- **Seeds:** 20 fixed seeds — `42, 123, 456, 789, 2024, 7, 13, 21, 33, 55, 77, 99, 111, 222, 333, 444, 555, 777, 888, 999`. Seeds control NumPy, Python `random`, `PYTHONHASHSEED`, and TensorFlow (`tf.random.set_seed`) — see `set_all_seeds()` in the notebooks.
- **Convergence analysis:** results reported at checkpoints n = 5 → 10 → 15 → 20 seeds (always the same ordered subset) to demonstrate statistical stability as seed count grows.
- **Statistical testing:** paired t-test, Wilcoxon signed-rank, Cohen's d, bootstrap 95% CI, Shapiro-Wilk normality check, and one-sample t-test against literature baseline values.
- **Ablation study:** XGBoost-only, single-scale BiLSTM (1s/5s/15s), multi-scale BiLSTM without k-NN/XGB, multi-scale BiLSTM + k-NN, vs. the full Detector.

## How to Run

1. Open the relevant notebook in `notebooks/` (`Detector_15ms_...` or `Detector_25ms_...`) in Google Colab or Jupyter.
2. Set `DATASET_PATH` to the location of the corresponding CSV (`Dataset_1.csv` for 25 m/s, `Dataset_2.csv` for 15 m/s).
3. Run the notebook cells in order. The pipeline is **resumable** — per-seed results already saved to disk (in `seed_runs_*ms/` or `baselines_*ms/`) are automatically skipped, so re-running the notebook only computes missing seeds.
4. Supplementary analyses (leakage audit, feature ablation, LSTM sensitivity, etc.) are in `notebooks/t1-t11_20_seed/`.

## Outputs Reference

| File (in `seed_runs_*ms/` or `baselines_*ms/`) | Content |
|---|---|
| `summary.json` | Mean ± std accuracy/precision/recall/F1 across seeds |
| `ablation_summary.json` | Ablation study results per component |
| `ablation_paired_ttest.json` | Paired t-test + Cohen's d vs. full Detector |
| `knn_sensitivity_summary.json` | Accuracy/graph density at k = 3, 5, 7, 10 |
| `statistical_tests_vs_literature.json` | Significance test vs. literature baseline values |
| `meta_feature_importance_summary.json` | Feature-block contribution to the meta-learner |
| `convergence_accuracy_*.png` | Accuracy vs. seed-count convergence plot |
| `seed_<n>_result.json` | Raw per-seed metrics and predictions |
| `leakage_audit_*ms.json` | Gap-split leakage audit results |

## Known Limitations

- Wilcoxon signed-rank p-values have a resolution floor that depends on the number of paired seeds available for a given comparison.
- Relative-speed features may partially couple with the mobility regime itself; interpret feature-importance results with this in mind.
- The BiLSTM-15s branch tends to dominate the meta-learner's decision weight relative to other blocks — see `meta_feature_importance_summary.json`.
- **Class support gap at 25 m/s:** in 3 of the 20 seeds (`789`, `2024`, `444`), the stratified episode split resulted in **zero test-set samples for one class** (Interference or Constant), which drives precision/recall/F1 for that class to 0 by convention even though accuracy on the other classes remains high. This inflates the macro-metric std at 25 m/s relative to accuracy std — see individual `seed_<n>_result.json` confusion matrices for details. Recommend either excluding affected seeds from macro-average reporting, using `zero_division=nan` with NaN-aware averaging, or enforcing a minimum per-class episode count in the split.

## Baseline References

- **Kosmanos et al.** — Random Forest with statistical tabular features for VANET jamming detection (also the source of the RF Jamming dataset used in this work).
- **Murshed et al.** — LSTM-RELU-IO-L1 for jamming classification.
- **Wickramasurendra et al.** — BERT WordPiece for jamming signal classification.

## Revision: protocol-controlled re-evaluation (notebooks 01-07)

The notebooks and results above correspond to the original submission. The revised analysis (pseudo-episode-grouped partitioning, 2x2 protocol x model factorial, corrected statistics, JamShield check, CPU latency) is in `notebooks/01`-`07` and `src/`. **Numbers in the revised manuscript come from these notebooks; the "Main Results" table above is from the original pipeline and is kept for reference.**

| Path | Purpose |
|---|---|
| `src/pcmef_fixed.py` | Core library: windowing, grouped partitioning, RF-Stat / LSTM / BERT / k-NN runners, factorial decomposition, hybrid |
| `src/pcmef_v2_infra.py` | Sanity suite (no episode across partitions, unique alignment keys) |
| `src/pcmef_corrected_stats.py` | Inference corrected for non-independent seeds (SE rescaled by episode ratio) |
| `src/jamshield_e59_counter.py` | JamShield counter-column handling and RF runner |
| `notebooks/01_protocol_vs_model_15ms.ipynb` | Protocol vs model, 15 m/s (RF-Stat, LSTM, BERT) |
| `notebooks/02_protocol_vs_model_25ms.ipynb` | Protocol vs model, 25 m/s |
| `notebooks/03_knn_sensitivity.ipynb` | k-NN sensitivity |
| `notebooks/04_followup_hybrid_truncation.ipynb` | Overlap audit, hybrid vs baselines, TOST, BERT truncation, corrected statistics |
| `notebooks/05_jamshield_validation.ipynb` | JamShield independent-dataset check |
| `notebooks/06_latency_cpu.ipynb` | Single-thread CPU latency and memory footprint |
| `notebooks/07_fill_in_values.ipynb` | Corrected intervals, power, summary values from existing result JSON |

**Setup.** Notebooks target Google Colab with Drive mounted. Copy `src/pcmef_fixed.py`, `src/pcmef_v2_infra.py` and `src/pcmef_corrected_stats.py` to `MyDrive/Detector_Results/`, and `src/jamshield_e59_counter.py` to `MyDrive/JamShield_Dataset/`; the notebooks read them from there. The Kosmanos dataset is not included (download from IEEE DataPort, see above). The JamShield files are in `data/jamshield/` (source and license in `data/jamshield/README.md`); copy them to `MyDrive/JamShield_Dataset/` before running notebook 05.
**Run order.** 01 -> 02 -> 03 -> 04 -> 05 -> 07; run 06 in a fresh CPU runtime. Results are cached per cell/seed as JSON; a cache is reused only if the per-model code hash matches, so do not edit the model functions in `pcmef_fixed.py`.

**Inference note.** The 20 seeds re-draw the partition of the same pseudo-episodes, so they are not independent. Notebook 04 reports both plain paired tests and corrected ones (SE inflation about x2.5); the corrected ones are the conservative reference.

## License
Code: MIT. Datasets: subject to the original licenses (Kosmanos et al.; JamShield, see `data/jamshield/README.md`).
