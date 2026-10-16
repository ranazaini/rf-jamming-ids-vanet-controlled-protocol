"""
jamshield_e59_counter.py: cumulative-counter fix for the JamShield features (E5.9).

Hypothesis: some raw JamShield features are cumulative per-station counters
(values only increase during a session), so window statistics (mean/max/...)
act as a proxy for position within the session rather than as a jamming signal.

Steps
  1. Identify columns that are monotonically increasing per (source_file,
     station): `identify_counter_columns`. On the real data 19 of 37 columns
     qualify (WiFi tx/rx packet, byte and retry counters, listed in
     COUNTER_COLS); the other 18 (rate, RSSI, noise floor, SINR per antenna)
     go up and down and are left unchanged.
  2. Replace those columns by per-interval differences within each
     (source_file, station) group: `apply_counter_diff`.
  3. Re-run the episode split and RF-Stat on the corrected features:
     `run_rf_jamshield_fixed`, `run_e59_experiment`.
  4. Compare with the original result (99.29% accuracy).
"""

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import pcmef_fixed as pf

# 19 kolom yang teridentifikasi sebagai counter kumulatif (Langkah 1)
COUNTER_COLS = [
    'tx_total_pkts', 'tx_total_bytes', 'tx_ucast_pkts', 'tx_ucast_bytes',
    'tx_mcast_bcast_pkts', 'tx_mcast_bcast_bytes', 'tx_failures',
    'rx_data_pkts', 'rx_data_bytes', 'rx_ucast_pkts', 'rx_ucast_bytes',
    'rx_mcast_bcast_pkts', 'rx_mcast_bcast_bytes', 'rx_decrypt_succeeds',
    'tx_data_pkts_retried', 'tx_total_pkts_sent', 'tx_pkts_retries',
    'tx_pkts_retry_exhausted', 'rx_total_pkts_retried',
]

# 18 kolom sinyal RF -- TIDAK diubah (bukan counter)
NON_COUNTER_COLS = [
    'rate_last_tx_pkt_min', 'rate_last_tx_pkt_max',
    'per_antenna_rssi_last_rx_data_frame_1', 'per_antenna_rssi_last_rx_data_frame_2',
    'per_antenna_rssi_last_rx_data_frame_3', 'per_antenna_rssi_last_rx_data_frame_4',
    'per_antenna_avg_rssi_rx_data_frames_1', 'per_antenna_avg_rssi_rx_data_frames_2',
    'per_antenna_avg_rssi_rx_data_frames_3', 'per_antenna_avg_rssi_rx_data_frames_4',
    'per_antenna_noise_floor_1', 'per_antenna_noise_floor_2',
    'per_antenna_noise_floor_3', 'per_antenna_noise_floor_4',
    'sinr_per_antenna_1', 'sinr_per_antenna_2', 'sinr_per_antenna_3', 'sinr_per_antenna_4',
]


def identify_counter_columns(df_loaded, raw_feature_cols, max_pct_decreasing=1.0):
    """
    Langkah 1. df_loaded = output load_jamshield() (Bagian 1, kolom asli
    'station_seq' untuk urutan waktu per station belum di-rename).
    Return list kolom yang <= max_pct_decreasing % langkahnya menurun.
    """
    rows = []
    for col in raw_feature_cols:
        n_dec, n_pairs = 0, 0
        for _, g in df_loaded.groupby(['source_file', 'station']):
            vals = g.sort_values('station_seq')[col].values.astype(float)
            if len(vals) < 2:
                continue
            d = np.diff(vals)
            n_dec += int((d < 0).sum())
            n_pairs += len(d)
        pct = 100.0 * n_dec / n_pairs if n_pairs else 0.0
        rows.append((col, pct))
    return [c for c, pct in rows if pct <= max_pct_decreasing], rows


def apply_counter_diff(df, counter_cols, group_cols=('source_file', 'station'),
                       order_col='station_seq'):
    """
    Langkah 2. Ubah counter_cols dari nilai kumulatif -> selisih per
    interval (diff), dihitung DI DALAM tiap grup (source_file, station)
    -- BUKAN di dalam tiap episode/bucket 60 detik -- karena counter
    hardware-nya memang kontinu sepanjang 1 sesi rekaman (file, station),
    bucket 60 detik cuma potongan buatan untuk pseudo-episode.

    Baris pertama tiap grup (tidak ada nilai sebelumnya) diisi 0.
    """
    df = df.sort_values(list(group_cols) + [order_col]).copy()
    for col in counter_cols:
        df[col] = df.groupby(list(group_cols))[col].diff().fillna(0.0)
    return df


def run_rf_jamshield_fixed(df_fixed, seed, n_features):
    """Sama persis dengan run_rf_jamshield() di jamshield_pipeline_part2_3.py,
    hanya beda input df (sudah di-diff-kan untuk COUNTER_COLS)."""
    tr_df, va_df, te_df = pf.split_by_episode(df_fixed, seed=seed)
    Xtr_l, ytr, _ = pf.create_windows_keyed(tr_df, 15, 1)
    Xte_l, yte, _ = pf.create_windows_keyed(te_df, 15, 1)
    if not Xtr_l or not Xte_l or len(np.unique(ytr)) < 2:
        return None

    Xtr = pf.extract_tabular_features(Xtr_l)
    Xte = pf.extract_tabular_features(Xte_l)
    assert Xtr.shape[1] == n_features * 7

    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                min_samples_split=5, min_samples_leaf=2,
                                n_jobs=-1, random_state=seed,
                                class_weight='balanced')
    rf.fit(sc.transform(Xtr), ytr)
    m = pf.compute_metrics(yte, rf.predict(sc.transform(Xte)), num_classes=2)
    m.update(model='RF_Stat_counterfix', dataset='JamShield', protocol='pcmef',
             arch='modified', seed=int(seed), n_train=int(len(Xtr)), n_test=int(len(Xte)))
    return m


def run_e59_experiment(df_loaded, raw_feature_cols, out_dir, seeds=None):
    """
    Langkah 2-4 lengkap: identifikasi -> diff -> rerun split+RF-Stat ->
    bandingkan ke hasil lama (99.29%, dari
    Reviewer_E_JamShield_Hasil_dan_Draft.md).
    """
    import pcmef_fixed as pf
    seeds = seeds or pf.SEEDS
    os.makedirs(out_dir, exist_ok=True)

    counter_cols, pct_table = identify_counter_columns(df_loaded, raw_feature_cols)
    print(f"Kolom teridentifikasi sebagai counter kumulatif: {len(counter_cols)}/{len(raw_feature_cols)}")

    df_fixed = apply_counter_diff(df_loaded, counter_cols)
    df_fixed = df_fixed.rename(columns={
        'pseudo_episode_id': 'pseudo_run_id', 'time_sec': 'Time', 'y': 'label',
    })
    pf.FEATURES = list(raw_feature_cols)

    results = []
    for seed in seeds:
        fp = os.path.join(out_dir, f'JamShield_RFStat_counterfix_seed{seed}.json')
        if os.path.exists(fp):
            results.append(json.load(open(fp)))
            continue
        r = run_rf_jamshield_fixed(df_fixed, seed, len(raw_feature_cols))
        if r is None:
            continue
        json.dump(r, open(fp, 'w'), indent=2)
        results.append(r)
        print(f"  [counterfix seed {seed:4d}] acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f}")

    accs = np.array([r['accuracy'] for r in results])
    OLD_MEAN_ACC = 0.9929  # dari Reviewer_E_JamShield_Hasil_dan_Draft.md, 20 seed
    print(f"\n=== HASIL SETELAH FIX COUNTER (n={len(accs)} seed) ===")
    print(f"  Accuracy mean = {accs.mean()*100:.2f}%  (SD={accs.std(ddof=1)*100:.2f}%)"
          if len(accs) > 1 else f"  Accuracy = {accs.mean()*100:.2f}%")
    print(f"  Accuracy LAMA (sebelum fix, 20 seed) = {OLD_MEAN_ACC*100:.2f}%")
    print(f"  Selisih = {(accs.mean()-OLD_MEAN_ACC)*100:+.2f} poin persentase")
    return results, counter_cols
