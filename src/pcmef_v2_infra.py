"""
pcmef_v2_infra.py: automatic sanity checks for the PCMEF pipeline.

Run once before the experiments, with `pcmef_fixed` imported as `pf`:

    import pcmef_v2_infra as v2
    v2.run_sanity_suite(pf, df, seeds=(42,))

Checks (per seed):
  a. no episode appears in more than one partition (train / val / test)
  b. (episode, t0) keys stay unique after align_multi
  c. SpatialKNN reference set contains train windows only and is not changed
     by transform() on validation data
"""

import numpy as np


# ==========================================================================
# E0.2 -- SANITY TEST OTOMATIS
# ==========================================================================

def check_no_episode_across_partitions(pf, df, seed=42):
    """
    E0.2a. split_by_episode() hanya assert train vs test disjoint secara
    internal (lihat pcmef_fixed.py baris ~260). Di sini dicek KETIGA
    pasangan (train-val, train-test, val-test) secara independen, tanpa
    percaya begitu saja pada assert internal kode.
    """
    tr_df, va_df, te_df = pf.split_by_episode(df, seed=seed)
    tr_eps = set(tr_df['pseudo_run_id'].unique())
    va_eps = set(va_df['pseudo_run_id'].unique())
    te_eps = set(te_df['pseudo_run_id'].unique())
    ok_tv = len(tr_eps & va_eps) == 0
    ok_tt = len(tr_eps & te_eps) == 0
    ok_vt = len(va_eps & te_eps) == 0
    assert ok_tv, f"SANITY GAGAL: {len(tr_eps & va_eps)} episode ada di train DAN val sekaligus"
    assert ok_tt, f"SANITY GAGAL: {len(tr_eps & te_eps)} episode ada di train DAN test sekaligus"
    assert ok_vt, f"SANITY GAGAL: {len(va_eps & te_eps)} episode ada di val DAN test sekaligus"
    return dict(ok=True, n_train_eps=len(tr_eps), n_val_eps=len(va_eps), n_test_eps=len(te_eps))


def check_align_multi_unique_keys(pf, df, seed=42, window_sizes=None):
    """
    E0.2b. Setelah align_multi() menyejajarkan window 1s/5s/15s, kunci
    (pseudo_run_id, t0) hasil kept_keys HARUS unik (tidak boleh ada
    duplikat) -- kalau ada duplikat berarti window dari episode/waktu
    yang sama ke-mapping dobel, dan baris X/y antar skala bisa salah
    pasang lagi (regresi ke bug A3).
    """
    window_sizes = window_sizes or pf.WINDOW_SIZES
    tr_df, va_df, te_df = pf.split_by_episode(df, seed=seed)
    per_scale = {}
    for w in window_sizes:
        X_l, y, keys = pf.create_windows_keyed(tr_df, w, step_sec=1)
        per_scale[w] = (keys, X_l)
    ref_w = window_sizes[0]
    keys_ref = per_scale[ref_w][0]
    arrays_with_keys = [(per_scale[w][0], np.zeros((len(per_scale[w][0]), 1)))
                        for w in window_sizes]
    _, keep, kept_keys = pf.align_multi(keys_ref, arrays_with_keys)
    n_dup = len(kept_keys) - len(set(kept_keys))
    assert n_dup == 0, f"SANITY GAGAL: {n_dup} kunci (episode,t0) duplikat setelah align_multi"
    return dict(ok=True, n_kept_keys=len(kept_keys), n_unique=len(set(kept_keys)))


def check_spatialknn_train_only(pf, df, seed=42, window_sec=15):
    """
    E0.2c. SpatialKNN.ref_ (referensi tetangga) HARUS di-fit dari window
    TRAIN saja -- jumlah baris ref_ setelah fit() harus SAMA PERSIS dengan
    jumlah window train yang dipakai untuk fit, dan TIDAK BOLEH berubah
    setelah transform() dipanggil ke val/test (fit-lah yang menentukan
    referensi, transform tidak boleh menambah/mengubah ref_). Regresi ke
    bug A4 kalau ref_ ikut memasukkan window val/test.
    """
    tr_df, va_df, te_df = pf.split_by_episode(df, seed=seed)
    Xtr_l, ytr, _ = pf.create_windows_keyed(tr_df, window_sec, step_sec=1)
    Xva_l, yva, _ = pf.create_windows_keyed(va_df, window_sec, step_sec=1)
    if not Xtr_l or not Xva_l:
        return dict(ok=None, note='data train/val kosong di seed ini, dilewati')

    max_len = max(max(len(x) for x in Xtr_l), max(len(x) for x in Xva_l))
    Xtr, _ = pf.pad_list(Xtr_l, max_len)
    Xva, _ = pf.pad_list(Xva_l, max_len)

    knn = pf.SpatialKNN(k=5)
    knn.fit(Xtr)
    n_ref_after_fit = len(knn.ref_)
    assert n_ref_after_fit == len(Xtr), (
        f"SANITY GAGAL: ref_ punya {n_ref_after_fit} baris, harusnya {len(Xtr)} "
        f"(=jumlah window TRAIN)")

    _ = knn.transform(Xva, is_train=False)  # transform ke val, TIDAK BOLEH ubah ref_
    assert len(knn.ref_) == n_ref_after_fit, (
        "SANITY GAGAL: ref_ berubah setelah transform() dipanggil ke val -- "
        "indikasi val ikut bocor ke referensi k-NN")
    return dict(ok=True, n_ref=n_ref_after_fit, n_train_windows=len(Xtr))


def run_sanity_suite(pf, df, seeds=(42,)):
    """
    E0.2. Jalankan ketiga sanity check di atas untuk tiap seed di `seeds`.
    Berhenti (AssertionError) di seed pertama yang gagal -- dipanggil SEKALI
    di awal sebelum eksperimen Tahap 1-5 dijalankan, bukan tiap seed produksi
    (supaya tidak menambah waktu training).
    """
    report = {}
    for seed in seeds:
        print(f"--- Sanity suite, seed {seed} ---")
        r1 = check_no_episode_across_partitions(pf, df, seed=seed)
        print(f"  [E0.2a] partisi episode tidak overlap : OK "
              f"(train={r1['n_train_eps']}, val={r1['n_val_eps']}, test={r1['n_test_eps']})")
        r2 = check_align_multi_unique_keys(pf, df, seed=seed)
        print(f"  [E0.2b] kunci (episode,t0) unik setelah align_multi : OK "
              f"({r2['n_kept_keys']} kunci)")
        r3 = check_spatialknn_train_only(pf, df, seed=seed)
        if r3['ok'] is None:
            print(f"  [E0.2c] SpatialKNN train-only : dilewati ({r3['note']})")
        else:
            print(f"  [E0.2c] SpatialKNN ref_ train-only, tidak berubah oleh transform(val) : OK "
                  f"({r3['n_ref']} referensi)")
        report[seed] = dict(partitions=r1, align_multi=r2, spatial_knn=r3)
    print("\n>> SEMUA SANITY CHECK LULUS. Aman lanjut ke eksperimen produksi.")
    return report
