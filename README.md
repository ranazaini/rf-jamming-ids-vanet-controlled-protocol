# Protocol-Controlled Multi-Seed Re-Evaluation of RF Jamming Intrusion Detection in Vehicular Networks

Official repository (code and per-seed outputs) for the paper **"Protocol-Controlled Multi-Seed Re-Evaluation of RF Jamming Intrusion Detection in Vehicular Networks."**

![License](https://img.shields.io/badge/License-MIT-yellow) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

The paper is a **re-evaluation study**, not a proposal of a new detector. It asks how much of the reported performance of RF-jamming detectors on the Kosmanos et al. dataset depends on the **evaluation protocol** (how windows are partitioned into train and test) rather than on the **model**. It separates the two with a 2x2 protocol × model factorial design over three detector families (RF-Stat, LSTM, BERT), 20 seeds, two mobility regimes (15 and 25 m/s), and statistics corrected for the non-independence of seeds. A multi-scale BiLSTM + XGBoost + k-NN hybrid is included only as a **reference model** for comparison; it is not presented as a contribution. An independent dataset (JamShield) and a CPU latency measurement complete the analysis.

> **Which part to use.** The revision (notebooks `01`–`08`, `src/`, `results_revision/`) is the code and data behind the revised manuscript. The pipeline of the original submission (folders `baselines_*`, `seed_runs_*`, the two `Detector_*` notebooks and `t1-t11_20_seed/`) is kept for traceability and is described at the end of this file; its numbers and its hybrid-centred framing are **superseded** by the revision.

## Repository structure

```
.
├── README.md
├── src/                          # Revision code
│   ├── pcmef_fixed.py            #   core library (windowing, grouped partitioning, runners, factorial, hybrid)
│   ├── pcmef_v2_infra.py         #   sanity suite
│   ├── pcmef_corrected_stats.py  #   inference corrected for non-independent seeds
│   ├── jamshield_e59_counter.py  #   JamShield counter-column handling, RF runner
│   └── jamshield_per_file.py     #   JamShield per-recording accuracy (leave-one-file-out)
├── notebooks/
│   ├── 01_… – 08_…               # Revision notebooks (table below)
│   ├── Detector_15ms_FINAL_20seed___Komputasi_Task_1.ipynb   # original submission, 15 m/s
│   ├── Detector_25ms_FINAL_20seed___Komputasi_Task_1.ipynb   # original submission, 25 m/s
│   └── t1-t11_20_seed/                                       # original supplementary notebooks
├── results_revision/             # Per-seed JSON and tables of the revision (see "Outputs of the revision")
├── baselines_15ms/  baselines_25ms/   # original submission: baseline per-seed results
└── seed_runs_15ms/  seed_runs_25ms/   # original submission: hybrid per-seed results
```

> The `ms` suffix in folder and file names means **metres per second (m/s)** of the mobility regime, not milliseconds.

## Datasets (not redistributed)

| Dataset | Use | Source |
|---|---|---|
| RF Jamming Dataset for Vehicular Wireless Networks (Kosmanos et al.) | main study | [IEEE DataPort](https://ieee-dataport.org/documents/rf-jamming-dataset-vehicular-wireless-networks) |
| JamShield (Panitsas, Yigit, Tassiulas, Maglaras, Canberk) | independent validation (Reviewer E) | [IEEE DataPort](https://ieee-dataport.org/documents/jamshield-dataset), DOI [10.21227/5hzf-w161](https://doi.org/10.21227/5hzf-w161); raw CSV in the [authors' repository](https://github.com/panitsasi/JamShield-Dataset) |

Neither dataset is included here (JamShield's authors state no license). Kosmanos files: `Dataset_1.csv` (25 m/s, Speed ∈ [23, 27]) and `Dataset_2.csv` (15 m/s, Speed ∈ [13, 17]); columns used: `SNR`, `RSSI`, `PDR`, `Speed`, `Relative_Speed`, `Scenario`, `Time`. For JamShield, download the CSV files from the `data/` folder of the authors' repository.

## Method of the revision

- **Protocol.** *Source-like* partitioning splits windows without keeping recordings together; *PCMEF* (pseudo-episode-grouped partitioning) keeps every 60-second pseudo-episode entirely in one partition (train, validation or test) and checks that no episode crosses partitions.
- **Factorial design.** A = source model + source-like split, B = source model + PCMEF, C = modified model + source-like split, D = modified model + PCMEF. Δprotocol = B − A, Δmodel = D − B, interaction = (D − B) − (C − A). Holm correction over {Δprotocol, Δmodel}. Main metrics: macro-F1 and balanced accuracy (accuracy as a complement).
- **Detector families.** RF-Stat, LSTM and BERT; BERT is also run with `MAX_LEN` = 288 for the modified model to remove token truncation.
- **Reference hybrid.** Multi-scale BiLSTM + XGBoost + k-NN meta-learner, compared with the baselines on a common subset of test windows (paired TOST, margin ±2 pp). It is a comparison model, not a proposed method.
- **Statistics.** 20 seeds re-draw the partition of the same pseudo-episodes, so they are not independent. Notebook 04 reports plain paired tests and tests with the standard error corrected by the episode ratio (inflation about ×2.5); the corrected ones are the conservative reference. Notebook 07 gives corrected intervals and the minimum detectable effect.
- **JamShield.** Counter-column correction (E5.9), file-level evaluation by leave-one-file-out and grouped 4-fold (E5.10, manual folds as the main result), per-recording accuracy, and the follow-up checks.
- **Latency.** Single-thread CPU, batch size 1, 100 warm-up and 1000 timed runs, 3 repeats, for all detector configurations.

## Code and notebooks

| File | Purpose |
|---|---|
| `src/pcmef_fixed.py` | Windowing, grouped partitioning, RF-Stat / LSTM / BERT / k-NN runners, factorial decomposition, hybrid. |
| `src/pcmef_v2_infra.py` | Sanity suite (no episode across partitions, unique alignment keys, train-only spatial k-NN). |
| `src/pcmef_corrected_stats.py` | Corrected TOST and factorial decomposition for non-independent seeds. |
| `src/jamshield_e59_counter.py` | JamShield counter-column handling and RF runner (E5.9). |
| `src/jamshield_per_file.py` | Per-recording accuracy under leave-one-file-out (same settings as E5.10). |
| `notebooks/01_protocol_vs_model_15ms.ipynb` | Protocol vs model at 15 m/s (RF-Stat, LSTM, BERT); split audits; rank flip. |
| `notebooks/02_protocol_vs_model_25ms.ipynb` | Protocol vs model at 25 m/s. |
| `notebooks/03_knn_sensitivity.ipynb` | k-NN sensitivity (15 and 25 m/s) and extended analysis. |
| `notebooks/04_followup_hybrid_truncation.ipynb` | Overlap audit, reference hybrid, TOST, BERT truncation and `MAX_LEN` = 288, corrected statistics. |
| `notebooks/05_jamshield_validation.ipynb` | JamShield E5.8–E5.11, including per-recording accuracy. |
| `notebooks/06_latency_cpu.ipynb` | Single-thread CPU latency and memory footprint. Needs a fresh CPU runtime. |
| `notebooks/07_fill_in_values.ipynb` | Corrected intervals, minimum detectable effect and summary values from existing JSON (no training). |
| `notebooks/08_collect_results.ipynb` | Collects the result files of the revision into `results_revision/` (no datasets). |

## How to run

Notebooks target Google Colab (GPU for LSTM/BERT) with Google Drive mounted.

1. Put the datasets on Drive: `MyDrive/Dataset RF Jamming/Dataset_1.csv`, `Dataset_2.csv`; JamShield CSV files in `MyDrive/JamShield_Dataset/`.
2. Copy `src/pcmef_fixed.py`, `src/pcmef_v2_infra.py`, `src/pcmef_corrected_stats.py` and `src/jamshield_per_file.py` to `MyDrive/Detector_Results/`, and `src/jamshield_e59_counter.py` to `MyDrive/JamShield_Dataset/` (these are the paths the notebooks read).
3. Run in order: `01` → `02` → `03` → `04` → `05` → `07`; run `06` in a fresh CPU runtime; run `08` last to collect results. Dependencies: `requirements.txt`.

Results are cached per cell and seed as JSON, so interrupted runs resume. A cache is reused only if the per-model code hash matches; changing a hashed function invalidates that model's cache, so do not edit the model functions in `pcmef_fixed.py` when reusing the provided results.

## Outputs of the revision (`results_revision/`)

| Folder | Content |
|---|---|
| `factorial/` | Per-seed JSON of cells A–D for RF-Stat, LSTM and BERT at 15 and 25 m/s. |
| `factorial_bert_maxlen288/` | BERT cells C and D with `MAX_LEN` = 288. |
| `e41_hybrid/` | Per-seed JSON of the reference hybrid, ablations and baselines on the common subset (15 m/s). |
| `knn_sensitivity/`, `knn_sensitivity_ext/` | k-NN sensitivity results. |
| `audit_e55/` | Split-independence audit outputs. |
| `tables_and_latency/` | CSV tables (TOST, corrected statistics, overlap, main table) and `latency_suite*.json`. |
| `jamshield/` | E5.9 per seed (`e59_counter_fix`), E5.10 (`e510_lofo_sgkf`, `e510_manual_folds`, `LOFO_per_file.csv/json`), E5.11, RF/LSTM runs, file inventory. |
| `MANIFEST.csv` | Path and size of every file. |

## Original submission (superseded)

The following describes the pipeline and results of the original submission. They are kept for traceability; the revised manuscript uses the revision above.

### Original method

- **Labels (3-class):** `Scenario 1 → Interference`, `Scenario 2 → Reactive`, `Scenario 3 → Constant`
- **Pseudo-episodes:** rows grouped into 60-second episodes (`floor(Time / 60)`)
- **Split:** stratified, episode-based (60% train / 20% val / 20% test), per-class shuffle with a fixed `RandomState(seed)`
- **Windowing:** multi-scale — 1s, 5s, and 15s non-overlapping windows within each episode
- **Normalisation:** `StandardScaler` fit on the training set only, applied to val/test
- **Detector model:** multi-scale BiLSTM (1s/5s/15s branches) + tabular XGBoost features, fused via a k-NN meta-learner
- **Baselines:** RF + statistical tabular features (Kosmanos et al.), LSTM-RELU-IO-L1 (Murshed et al.), BERT WordPiece / DistilBERT (Wickramasurendra et al.)
- **Leakage audit:** gap-split re-evaluation to check for adjacency leakage between train/test episodes

### Original results (mean ± std, n = 20 seeds)

| Speed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 15 m/s | 87.76 ± 3.20 % | 83.24 ± 8.03 % | 82.54 ± 11.47 % | 81.84 ± 9.56 % |
| 25 m/s | 99.37 ± 0.85 % | 97.58 ± 7.44 % | 97.74 ± 7.81 % | 97.63 ± 7.61 % |

*Computed directly from `seed_runs_15ms/summary.json` and `seed_runs_25ms/summary.json` (20 seeds each). Note the wider precision/recall/F1 std at 25 m/s — this is likely driven by a small number of seeds where one class had few or no test episodes; cross-check against `leakage_audit_25ms.json` and per-class metrics before reporting.*

### Original reproducibility protocol

- **Seeds:** 20 fixed seeds — `42, 123, 456, 789, 2024, 7, 13, 21, 33, 55, 77, 99, 111, 222, 333, 444, 555, 777, 888, 999`. Seeds control NumPy, Python `random`, `PYTHONHASHSEED`, and TensorFlow (`tf.random.set_seed`) — see `set_all_seeds()` in the notebooks.
- **Convergence analysis:** results reported at checkpoints n = 5 → 10 → 15 → 20 seeds (always the same ordered subset) to demonstrate statistical stability as seed count grows.
- **Statistical testing:** paired t-test, Wilcoxon signed-rank, Cohen's d, bootstrap 95% CI, Shapiro-Wilk normality check, and one-sample t-test against literature baseline values.
- **Ablation study:** XGBoost-only, single-scale BiLSTM (1s/5s/15s), multi-scale BiLSTM without k-NN/XGB, multi-scale BiLSTM + k-NN, vs. the full Detector.

### Original: how to run

1. Open the relevant notebook in `notebooks/` (`Detector_15ms_...` or `Detector_25ms_...`) in Google Colab or Jupyter.
2. Set `DATASET_PATH` to the location of the corresponding CSV (`Dataset_1.csv` for 25 m/s, `Dataset_2.csv` for 15 m/s).
3. Run the notebook cells in order. The pipeline is **resumable** — per-seed results already saved to disk (in `seed_runs_*ms/` or `baselines_*ms/`) are automatically skipped, so re-running the notebook only computes missing seeds.
4. Supplementary analyses (leakage audit, feature ablation, LSTM sensitivity, etc.) are in `notebooks/t1-t11_20_seed/`.

### Original outputs reference

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

### Original known limitations

- Wilcoxon signed-rank p-values have a resolution floor that depends on the number of paired seeds available for a given comparison.
- Relative-speed features may partially couple with the mobility regime itself; interpret feature-importance results with this in mind.
- The BiLSTM-15s branch tends to dominate the meta-learner's decision weight relative to other blocks — see `meta_feature_importance_summary.json`.
- **Class support gap at 25 m/s:** in 3 of the 20 seeds (`789`, `2024`, `444`), the stratified episode split resulted in **zero test-set samples for one class** (Interference or Constant), which drives precision/recall/F1 for that class to 0 by convention even though accuracy on the other classes remains high. This inflates the macro-metric std at 25 m/s relative to accuracy std — see individual `seed_<n>_result.json` confusion matrices for details. Recommend either excluding affected seeds from macro-average reporting, using `zero_division=nan` with NaN-aware averaging, or enforcing a minimum per-class episode count in the split.

## Baseline references

- **Kosmanos et al.** — Random Forest with statistical tabular features for VANET jamming detection (also the source of the RF Jamming dataset used in this work).
- **Murshed et al.** — LSTM-RELU-IO-L1 for jamming classification.
- **Wickramasurendra et al.** — BERT WordPiece for jamming signal classification.

## License

Code: MIT. Datasets are not redistributed and remain subject to their original terms (Kosmanos et al.; JamShield, Panitsas et al.).
