"""
PCMEF -- protocol x model evaluation for RF-jamming intrusion detection
=======================================================================

2x2 design (Reviewer C, point 4):

    Cell  Model              Protocol
    A     source-faithful    source-like protocol
    B     source-faithful    PCMEF (pseudo-episode-grouped partitioning)
    C     modified           source-like protocol
    D     modified           PCMEF

    Delta_protocol = B - A   (model fixed to source-faithful)
    Delta_model    = D - B   (protocol fixed to PCMEF)
    interaction    = (D - B) - (C - A)

"source-like protocol" is our own definition (stratified random split at the
evaluation-unit level, no episode grouping); it is not a reconstruction of the
source paper's unknown protocol. See split_random_unit().

Sections
    1  Bug fixes: create_windows_keyed, align_by_key / align_multi,
       SpatialKNN (train-only reference set)
    2  Protocols: split_by_episode (PCMEF), split_random_unit (source-like)
    3  Protocol x model experiments (RF-Stat, LSTM, BERT; full 2x2 each)
    4  k-NN sensitivity with a train-only graph (Reviewer C, point 7)
    5  Aggregation: mean +- SD, deltas, bootstrap CI, Cohen's d, Holm
    6  E4.1: hybrid model, ablations, baselines on the same test-window keys

Usage (Colab): see the notebooks in this repository.
Note: comments inside the hashed model functions are intentionally left
untouched (see "RESULT VERSIONING").
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as sps

FEATURES = ['SNR', 'RSSI', 'PDR', 'Speed', 'Relative_Speed']
WINDOW_SIZES = [1, 5, 15]
EPISODE_LENGTH = 60
SEEDS = [42, 123, 456, 789, 2024, 7, 13, 21, 33, 55,
         77, 99, 111, 222, 333, 444, 555, 777, 888, 999]


# ==========================================================================
# RESULT VERSIONING (hash of the code that produced a cached result)
# ==========================================================================
# Runners reuse a cached JSON only if its stored "code_hash" matches the hash
# of the functions that model actually uses (model_code_hash). Editing the
# source of a hashed function (comments and docstrings included) therefore
# invalidates that model's cache. Hashing is per model: changing LSTM code
# does not invalidate RF-Stat, BERT or kNN results.

def _current_code_hash():
    """SHA-256 (12 karakter hex pertama) dari isi file pcmef_fixed.py yang
    SEDANG DIIMPOR saat ini. Kalau __file__ tidak tersedia (mis. dijalankan
    dari exec() interaktif tanpa file), return 'unknown' -- runner tetap
    jalan, cuma tanpa proteksi versioning."""
    import hashlib
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return 'unknown'


CODE_HASH = _current_code_hash()

# LEGACY_VALID_HASHES: whole-file hashes of earlier versions whose result files
# are known to be valid; they are accepted for every model.
LEGACY_VALID_HASHES = {'ec0582be6144'}

_COMMON_FUNCS = ['create_windows_keyed', 'split_by_episode',
                 'split_random_unit', 'filter_feasible_episodes',
                 'compute_metrics', 'pad_list', 'extract_tabular_features']
_MODEL_FUNCS = {
    'RF_Stat': ['run_rf_cell', '_rf_source_units', '_rf_adapted_units'],
    'LSTM': ['run_lstm_cell', 'build_lstm', '_make_gated_lstm_cell'],
    'BERT': ['run_bert_cell', '_tokens_source', '_tokens_modified',
             '_BertDataset', 'compute_feat_stats',
             'compute_feat_stats_from_windows'],
    'KNN': ['run_knn_full', 'run_knn_sensitivity_fixed',
            'build_embeddings_cache', '_build_temporal_bilstm',
            '_pad_normalize_fit_train', 'align_by_key', 'align_multi',
            'SpatialKNN'],
}


def model_code_hash(model_name):
    """Hash 12-hex dari source fungsi yang dipakai model_name (+ helper
    bersama + konstanta FEATURES/WINDOW_SIZES/EPISODE_LENGTH). Kalau source
    tidak bisa dibaca (exec interaktif), return 'unknown' -- runner tetap
    jalan tanpa proteksi versioning, sama seperti _current_code_hash()."""
    import hashlib, inspect
    g = globals()
    h = hashlib.sha256()
    h.update(repr((FEATURES, WINDOW_SIZES, EPISODE_LENGTH)).encode())
    try:
        for name in _COMMON_FUNCS + _MODEL_FUNCS[model_name]:
            h.update(inspect.getsource(g[name]).encode())
    except Exception:
        return 'unknown'
    return h.hexdigest()[:12]


def cache_is_valid(cached, model_name):
    """True kalau file JSON hasil lama boleh dipakai ulang untuk model_name:
    hash-nya = hash per-model sekarang, ATAU hash legacy yang sudah
    diverifikasi. 'unknown' tidak pernah dianggap valid."""
    ch = cached.get('code_hash')
    mh = model_code_hash(model_name)
    if ch is None or mh == 'unknown':
        return False
    return ch == mh or ch in LEGACY_VALID_HASHES


# ==========================================================================
# BAGIAN 1 -- PERBAIKAN BUG
# ==========================================================================

def create_windows_keyed(df, window_sec, step_sec=1):
    """
    Sama dengan create_windows() lama, tapi setiap window membawa kunci
    (pseudo_run_id, t0). Kunci inilah yang dipakai untuk menyejajarkan
    skala 1s / 5s / 15s.

    step_sec=1 -> overlapping (setting kamu sekarang)
    step_sec=window_sec -> non-overlapping (klaim di manuskrip)
    """
    X_list, y_list, key_list = [], [], []
    for run_id, group in df.groupby('pseudo_run_id'):
        group = group.sort_values('Time').reset_index(drop=True)
        times = group['Time'].values
        X_raw = group[FEATURES].values
        y_raw = group['label'].values
        t0, t_end = times[0], times[-1]
        while t0 + window_sec <= t_end:
            mask = (times >= t0) & (times < t0 + window_sec)
            if mask.sum() > 0:
                X_list.append(X_raw[mask])
                y_list.append(int(np.bincount(y_raw[mask]).argmax()))
                key_list.append((run_id, float(t0)))
            t0 += step_sec
    return X_list, np.array(y_list), key_list


def align_by_key(keys_ref, keys_other, arr_other):
    """
    FIX BUG A3.
    Mengembalikan arr_other yang disusun ulang mengikuti urutan keys_ref,
    plus mask baris keys_ref yang punya pasangan.

    Pipeline lama memakai align_min() yang hanya memotong ke jumlah baris
    terkecil. Itu memasangkan baris ke-i dari skala berbeda meskipun
    berasal dari episode berbeda.
    """
    idx = {k: i for i, k in enumerate(keys_other)}
    rows, mask = [], []
    for k in keys_ref:
        j = idx.get(k)
        mask.append(j is not None)
        if j is not None:
            rows.append(arr_other[j])
    mask = np.array(mask, dtype=bool)
    out = np.array(rows) if rows else np.zeros((0, arr_other.shape[1]))
    return out, mask


def align_multi(keys_ref, arrays_with_keys):
    """
    Sejajarkan beberapa array ke keys_ref sekaligus.
    arrays_with_keys: list of (keys, array)
    Mengembalikan (list array tersejajar, mask baris keys_ref yang lolos semua).
    """
    masks, mapped = [], []
    for keys, arr in arrays_with_keys:
        idx = {k: i for i, k in enumerate(keys)}
        m = np.array([k in idx for k in keys_ref], dtype=bool)
        masks.append(m)
        mapped.append((idx, arr))
    keep = np.logical_and.reduce(masks) if masks else np.ones(len(keys_ref), bool)
    kept_keys = [k for k, ok in zip(keys_ref, keep) if ok]
    out = []
    for idx, arr in mapped:
        out.append(np.array([arr[idx[k]] for k in kept_keys]))
    return out, keep, kept_keys


class SpatialKNN:
    """
    FIX BUG A4.
    Graf k-NN pseudo-spasial dengan tetangga HANYA dari training partition.

    fit(X_train_padded, train_episodes=None) -> simpan episode-mean vektor
                                    training, plus id episode tiap baris
                                    (untuk FIX E3.4 di bawah)
    transform(X_padded, is_train, query_episodes=None) -> descriptor 12-dim
                                    (mean tetangga 5-dim, std tetangga 5-dim,
                                     sim_mean, sim_std)

    is_train=True SEBELUMNYA cuma membuang self-match (diagonal, baris ke-i
    vs dirinya sendiri). FIX E3.4: sekarang membuang SEMUA tetangga yang
    berasal dari EPISODE YANG SAMA dengan query (bukan cuma baris index
    yang identik) -- supaya window lain dari episode yang sama (yang
    kondisinya nyaris identik) tidak jadi "tetangga termudah" secara
    artifisial. Butuh train_episodes (saat fit) dan query_episodes (saat
    transform, hanya untuk is_train=True) -- kalau salah satu tidak
    diberikan, perilaku mundur ke self-match-only (kompatibel mundur).
    """

    def __init__(self, k=5):
        self.k = k
        self.ref_ = None
        self.ref_episodes_ = None

    def fit(self, X_train_padded, train_episodes=None):
        self.ref_ = X_train_padded.mean(axis=1)
        self.ref_episodes_ = (np.asarray(train_episodes)
                              if train_episodes is not None else None)
        return self

    def transform(self, X_padded, is_train=False, query_episodes=None):
        if self.ref_ is None:
            raise RuntimeError('SpatialKNN belum di-fit')
        Xm = X_padded.mean(axis=1)
        n_feat = Xm.shape[1]
        if len(self.ref_) == 0:
            return np.zeros((len(Xm), n_feat * 2 + 2))
        sim = cosine_similarity(Xm, self.ref_)
        if is_train:
            np.fill_diagonal(sim, -np.inf)  # tetap buang self-match persis
            if self.ref_episodes_ is not None and query_episodes is not None:
                # FIX E3.4: buang SEMUA tetangga dari episode yang sama
                q_eps = np.asarray(query_episodes)
                same_episode = (self.ref_episodes_[None, :] == q_eps[:, None])
                sim = np.where(same_episode, -np.inf, sim)
        out = []
        for i in range(sim.shape[0]):
            s = sim[i]
            valid = np.isfinite(s)
            k_act = min(self.k, int(valid.sum()))
            if k_act <= 0:
                out.append(np.zeros(n_feat * 2 + 2))
                continue
            order = np.argsort(np.where(valid, s, -np.inf))[-k_act:]
            ns = s[order]
            nm = self.ref_[order].mean(axis=0)
            nstd = self.ref_[order].std(axis=0)
            out.append(np.concatenate([nm, nstd, [ns.mean()], [ns.std()]]))
        return np.array(out)

    def graph_density(self, n_query):
        n_ref = 0 if self.ref_ is None else len(self.ref_)
        max_edges = n_query * n_ref
        return float(n_query * min(self.k, n_ref) / max_edges) if max_edges else 0.0


def extract_tabular_features(X_list_or_padded):
    """35-dim statistik (7 statistik x 5 fitur). Terima list atau array padded."""
    out = []
    for x in X_list_or_padded:
        x = np.asarray(x)
        if x.ndim == 2 and (x == 0).all(axis=1).any():
            m = ~(x == 0).all(axis=1)
            x = x[m] if m.sum() > 0 else x
        f = []
        for c in range(x.shape[1]):
            v = x[:, c]
            f.extend([v.mean(), v.std(), v.min(), v.max(),
                      np.median(v), np.percentile(v, 25), np.percentile(v, 75)])
        out.append(f)
    return np.array(out)


def pad_list(X_list, max_len=None):
    if max_len is None:
        max_len = max(len(x) for x in X_list)
    n_feat = X_list[0].shape[1]
    out = np.zeros((len(X_list), max_len, n_feat), dtype=np.float32)
    for i, x in enumerate(X_list):
        sl = min(len(x), max_len)
        out[i, :sl, :] = x[:sl]
    return out, max_len


def compute_metrics(y_true, y_pred, num_classes=3, test_episodes=None):
    """
    FIX E0.3 (Tahap 0): selain accuracy/f1_macro/confusion_matrix yang sudah
    ada, sekarang OTOMATIS menghitung juga:
      - balanced_accuracy (lebih tahan kelas timpang dibanding accuracy)
      - majority_class + majority_class_baseline_accuracy (pembanding
        minimum yang wajib dikalahkan model)
      - test_window_count_per_class (jumlah window test per kelas)
      - test_episode_count_per_class (HANYA kalau test_episodes diberikan
        -- array id episode per baris y_true/y_pred, sepanjang y_true;
        dipakai pemanggil yang punya info episode per window)
    Field lama (accuracy, precision_macro, recall_macro, f1_macro,
    confusion_matrix) TIDAK berubah, jadi kode lama yang baca field-field
    itu tetap jalan tanpa perubahan.
    """
    from sklearn.metrics import balanced_accuracy_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    labels = list(range(num_classes))

    classes, counts = np.unique(y_true, return_counts=True)
    window_count_per_class = {int(c): int(n) for c, n in zip(classes, counts)}
    majority_class = int(classes[np.argmax(counts)]) if len(classes) else None
    majority_baseline = float(np.max(counts) / len(y_true)) if len(y_true) else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0

    out = dict(accuracy=acc, precision_macro=float(p), recall_macro=float(r),
              f1_macro=float(f1),
              confusion_matrix=confusion_matrix(y_true, y_pred,
                                                labels=labels).tolist(),
              balanced_accuracy=bal_acc,
              majority_class=majority_class,
              majority_class_baseline_accuracy=majority_baseline,
              test_window_count_per_class=window_count_per_class)

    if test_episodes is not None:
        test_episodes = np.asarray(test_episodes)
        ep_per_class = {}
        for c in classes:
            ep_per_class[int(c)] = int(len(set(test_episodes[y_true == c])))
        out['test_episode_count_per_class'] = ep_per_class

    return out


# ==========================================================================
# BAGIAN 2 -- PROTOKOL
# ==========================================================================

def split_by_episode(df, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Protokol PCMEF (kode 'pcmef'). Identik dengan notebook kamu."""
    rng = np.random.RandomState(seed)
    ep_to_label = df.groupby('pseudo_run_id')['label'].first().to_dict()
    eps_by_class = {}
    for ep, lbl in ep_to_label.items():
        eps_by_class.setdefault(lbl, []).append(ep)
    train_eps, val_eps, test_eps = [], [], []
    for lbl, eps in eps_by_class.items():
        eps = np.array(eps)
        rng.shuffle(eps)
        n = len(eps)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        train_eps += eps[:n_train].tolist()
        val_eps += eps[n_train:n_train + n_val].tolist()
        test_eps += eps[n_train + n_val:].tolist()
    assert not (set(train_eps) & set(test_eps))
    return (df[df['pseudo_run_id'].isin(train_eps)].copy(),
            df[df['pseudo_run_id'].isin(val_eps)].copy(),
            df[df['pseudo_run_id'].isin(test_eps)].copy())


def filter_feasible_episodes(df, min_duration_sec):
    """
    E3.2 (keputusan D3). Episode dianggap "layak" (feasible) untuk skala
    window tertentu kalau durasinya (rentang Time, max-min, di dalam
    episode itu) >= min_duration_sec. Episode yang lebih pendek dari skala
    terpanjang yang dipakai (15 detik untuk BiLSTM-15s / RF-Stat modified)
    TIDAK BISA menghasilkan satu window pun di skala itu -- memasukkannya
    ke populasi evaluasi membuat perbandingan antar-model jadi tidak adil
    (model yang window-nya pendek "menang" bukan karena lebih baik, tapi
    karena populasi test-nya beda).

    D3: SEMUA model (termasuk RF-Stat sel B, bukan cuma D) dievaluasi pada
    episode yang sama-sama layak untuk 15 detik -- panggil fungsi ini
    dengan min_duration_sec=15 SEBELUM split_by_episode()/run_*_cell(),
    lalu pakai df hasilnya untuk semua sel yang dibandingkan.

    Return (df_filtered, feasible_episode_ids, n_dropped_episodes).
    """
    dur = df.groupby('pseudo_run_id')['Time'].agg(lambda s: s.max() - s.min())
    feasible = set(dur[dur >= min_duration_sec].index)
    n_dropped = df['pseudo_run_id'].nunique() - len(feasible)
    return df[df['pseudo_run_id'].isin(feasible)].copy(), feasible, n_dropped


def split_random_unit(n_units, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Protokol "source-like" (kode 'source_like'). WAJIB disebut begini di
    response letter dan manuskrip -- JANGAN "original protocol" atau
    "source protocol", sesuai arahan dosen.

    PENTING: ini BUKAN klaim bahwa paper sumber memakai protokol ini.
    Manuskrip kamu sendiri (Sec 2.2) menyatakan protokol split paper sumber
    tidak dapat ditentukan dari teksnya. "source-like" didefinisikan sendiri
    sebagai: random stratified split di level unit evaluasi, tanpa grouping
    episode. Delta_protocol dibaca sebagai efek beralih dari split acak ke
    split ter-grup episode, bukan sebagai selisih terhadap paper sumber.

    Catatan dosen soal pairing: perbandingan A-vs-B (efek protokol) TIDAK
    strictly paired secara sampel test, karena ganti protokol = ganti isi
    split. bootstrap_ci di sini tetap dihitung per-seed (paired by seed,
    bukan paired by sample), yang valid, tapi laporkan A/B sebagai dua
    skema evaluasi berbeda, bukan estimasi paired yang bersih di level
    sampel. Lihat catatan yang sama di print_factorial().
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(n_units)
    tr, rest = train_test_split(idx, train_size=train_ratio,
                                stratify=y, random_state=seed)
    rel = val_ratio / (1 - train_ratio)
    va, te = train_test_split(rest, train_size=rel,
                              stratify=y[rest], random_state=seed)
    return tr, va, te


# ==========================================================================
# BAGIAN 3 -- EKSPERIMEN PROTOKOL x MODEL (Reviewer C poin 4)
# ==========================================================================
#
# arch:     'source'   = source-faithful model
#           'modified' = modified model (arsitektur/konfigurasi versi manuskrip)
# protocol: 'source_like' = protokol source-like (lihat split_random_unit)
#           'pcmef'       = protokol PCMEF (lihat split_by_episode)
#
# Sel   arch       protocol       kode
# A     source     source_like    A
# B     source     pcmef          B
# C     modified   source_like    C
# D     modified   pcmef          D

CELLS = [('source', 'source_like'), ('source', 'pcmef'),
         ('modified', 'source_like'), ('modified', 'pcmef')]
CELL_CODE = {('source', 'source_like'): 'A', ('source', 'pcmef'): 'B',
             ('modified', 'source_like'): 'C', ('modified', 'pcmef'): 'D'}

# RF-Stat uses the full 2x2 design, like LSTM and BERT.
RF_STAT_CELLS = CELLS

CELL_LABEL_MODEL = {'source': 'source-faithful', 'modified': 'modified'}
CELL_LABEL_PROTOCOL = {'source_like': 'source-like protocol', 'pcmef': 'PCMEF'}


# ---------- RF-Stat ----------

def _rf_source_units(df, vrs_window=5):
    """
    Arsitektur source-faithful Kosmanos: klasifikasi PER-SAMPEL (bukan window),
    fitur kanal + VRS (variasi relative speed) sebagai rolling std per episode.
    """
    parts = []
    for _, g in df.groupby('pseudo_run_id'):
        g = g.sort_values('Time').copy()
        g['VRS'] = (g['Relative_Speed']
                    .rolling(vrs_window, min_periods=1).std().fillna(0.0))
        parts.append(g)
    d = pd.concat(parts)
    X = d[['SNR', 'RSSI', 'PDR', 'Relative_Speed', 'VRS']].values
    return X, d['label'].values, d['pseudo_run_id'].values


def _rf_adapted_units(df, window_sec=15, step_sec=1):
    """Arsitektur adapted: 35-dim statistik atas window 15 detik."""
    Xl, y, keys = create_windows_keyed(df, window_sec, step_sec)
    if not Xl:
        return np.zeros((0, 35)), np.zeros(0), np.array([])
    return extract_tabular_features(Xl), y, np.array([k[0] for k in keys])


def run_rf_cell(df_speed, seed, arch, protocol, step_sec=1):
    """
    Satu sel untuk RF-Stat. Murah, hitungan detik.
    arch: 'source' atau 'modified'. protocol: 'source_like' atau 'pcmef'.

    D7 (direvisi): RF_STAT_CELLS sekarang = CELLS (ABCD penuh), bukan
    cuma B vs D lagi -- lihat run_factorial(..., cells=RF_STAT_CELLS).
    Fungsi ini sudah mendukung keempat sel sejak awal, jadi tidak ada
    perubahan logika di sini, hanya di cells yang dipanggil.
    """
    np.random.seed(seed)

    te_episodes = None
    if protocol == 'pcmef':
        tr_df, va_df, te_df = split_by_episode(df_speed, seed=seed)
        if arch == 'source':
            Xtr, ytr, _ = _rf_source_units(tr_df)
            Xte, yte, te_episodes = _rf_source_units(te_df)
        else:
            Xtr, ytr, _ = _rf_adapted_units(tr_df, step_sec=step_sec)
            Xte, yte, te_episodes = _rf_adapted_units(te_df, step_sec=step_sec)
    else:  # source_like: satu pool, split acak di level unit
        if arch == 'source':
            X, y, eps = _rf_source_units(df_speed)
        else:
            X, y, eps = _rf_adapted_units(df_speed, step_sec=step_sec)
        if len(X) == 0:
            return None
        tr, va, te = split_random_unit(len(X), y, seed=seed)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        te_episodes = eps[te] if len(eps) else None

    if len(Xtr) == 0 or len(Xte) == 0 or len(np.unique(ytr)) < 2:
        return None

    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                min_samples_split=5, min_samples_leaf=2,
                                n_jobs=-1, random_state=seed,
                                class_weight='balanced')
    rf.fit(sc.transform(Xtr), ytr)
    # E0.3: te_episodes dioper ke compute_metrics supaya
    # test_episode_count_per_class ikut terhitung otomatis.
    m = compute_metrics(yte, rf.predict(sc.transform(Xte)), test_episodes=te_episodes)
    m.update(model='RF_Stat', arch=arch, protocol=protocol,
             cell=CELL_CODE[(arch, protocol)], seed=int(seed),
             n_train=int(len(Xtr)), n_test=int(len(Xte)))
    return m


# ---------- LSTM ----------

# ==========================================================================
# FIX D2/E3.3 (keputusan dosen, dikonfirmasi lewat paper sumber [15]
# Table 2): replika arsitektur LSTM_RELU_IO-L1 -- varian LSTM yang
# ANGKANYA (95.67% @25m/s, 85.17% @15m/s) dikutip di Tabel 3 manuskrip,
# BUKAN varian "LSTM-L1" polos (83.33%/94.50%) yang sebelumnya terpasang
# secara keliru di sini. Bedanya ada di aktivasi PER-GERBANG:
#     input gate   : ReLU    (bukan sigmoid default Keras)
#     output gate  : ReLU    (bukan sigmoid default Keras)
#     forget gate  : sigmoid (sama seperti default)
#     cell/hidden  : tanh    (sama seperti default)
# Keras LSTM bawaan TIDAK bisa mengatur ini -- cuma ada satu
# `recurrent_activation` yang dipakai bersama untuk i/f/o, dan satu
# `activation` untuk cell/hidden. Paper sumber sendiri menyatakan mereka
# membuat "custom LSTM layer by extending the layer class from
# TensorFlow/Keras... facilitating control over activation functions for
# input gate, forget gate, output gate, and state updates" -- cell RNN
# kustom di bawah ini mereplikasi itu persis, memakai parameterisasi
# gerbang standar (persamaan (1)-(8) di paper: f_t, i_t, c~_t, c_t, o_t,
# h_t), dipasang lewat tf.keras.layers.RNN (cara yang disebutkan eksplisit
# di paper) supaya kompatibel penuh dengan training loop Keras biasa
# (compile/fit/predict, EarlyStopping, dst).
# ==========================================================================

def _make_gated_lstm_cell(units, input_activation='sigmoid',
                           forget_activation='sigmoid',
                           output_activation='sigmoid',
                           state_activation='tanh'):
    import tensorflow as tf

    class GatedLSTMCell(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = [units, units]   # [h, c]
            self.output_size = units
            self.act_i = tf.keras.activations.get(input_activation)
            self.act_f = tf.keras.activations.get(forget_activation)
            self.act_o = tf.keras.activations.get(output_activation)
            self.act_s = tf.keras.activations.get(state_activation)

        def build(self, input_shape):
            in_dim = int(input_shape[-1])
            # urutan blok: [i, f, c~, o] -- gaya Keras LSTMCell standar,
            # satu matmul besar lalu di-split 4, lebih efisien daripada
            # 4 matmul terpisah tapi secara matematis identik.
            self.W = self.add_weight(shape=(in_dim, self.units * 4),
                                      name='kernel',
                                      initializer='glorot_uniform')
            self.U = self.add_weight(shape=(self.units, self.units * 4),
                                      name='recurrent_kernel',
                                      initializer='orthogonal')
            self.b = self.add_weight(shape=(self.units * 4,), name='bias',
                                      initializer='zeros')
            self.built = True

        def call(self, inputs, states):
            h_prev, c_prev = states[0], states[1]
            z = (tf.matmul(inputs, self.W) + tf.matmul(h_prev, self.U)
                 + self.b)
            zi, zf, zc, zo = tf.split(z, num_or_size_splits=4, axis=-1)
            i_t = self.act_i(zi)
            f_t = self.act_f(zf)
            c_tilde = self.act_s(zc)
            o_t = self.act_o(zo)
            c_t = f_t * c_prev + i_t * c_tilde
            h_t = o_t * self.act_s(c_t)
            return h_t, [h_t, c_t]

        def get_config(self):
            cfg = super().get_config()
            cfg.update(units=self.units)
            return cfg

    return GatedLSTMCell()


def build_lstm(arch, input_shape, num_classes=3, seed=42):
    """
    arch='source'   : LSTM_RELU_IO-L1 (Murshed et al. [15], Table 2) --
                      1 layer, 50 unit, input/output gate ReLU, forget
                      gate sigmoid, cell/hidden tanh, dropout 0.2. Lihat
                      docstring _GatedLSTMCell di atas untuk alasan perlu
                      cell kustom (D2/E3.3).
    arch='modified' : 2-layer BiLSTM 128->64 + dense head (versi manuskrip)
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, RNN, Bidirectional,
                                         Dense, Dropout, BatchNormalization)
    tf.keras.utils.set_random_seed(seed)
    inp = Input(shape=input_shape)
    if arch == 'source':
        cell = _make_gated_lstm_cell(
            50, input_activation='relu', forget_activation='sigmoid',
            output_activation='relu', state_activation='tanh')
        x = RNN(cell, return_sequences=False)(inp)
        x = Dropout(0.2)(x)
    else:
        x = Bidirectional(LSTM(128, return_sequences=True))(inp)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)


def run_lstm_cell(df_speed, seed, arch, protocol, window_sec=15,
                  step_sec=1, epochs=100, batch_size=32):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if protocol == 'pcmef':
        tr_df, va_df, te_df = split_by_episode(df_speed, seed=seed)
        Xtr_l, ytr, _ = create_windows_keyed(tr_df, window_sec, step_sec)
        Xva_l, yva, _ = create_windows_keyed(va_df, window_sec, step_sec)
        Xte_l, yte, kte = create_windows_keyed(te_df, window_sec, step_sec)
        if min(len(Xtr_l), len(Xva_l), len(Xte_l)) == 0:
            return None
        max_len = max(max(len(x) for x in Xtr_l),
                      max(len(x) for x in Xva_l),
                      max(len(x) for x in Xte_l))
    else:
        X_l, y, k_l = create_windows_keyed(df_speed, window_sec, step_sec)
        if len(X_l) == 0:
            return None
        max_len = max(len(x) for x in X_l)
        tr, va, te = split_random_unit(len(X_l), y, seed=seed)
        Xtr_l = [X_l[i] for i in tr]; ytr = y[tr]
        Xva_l = [X_l[i] for i in va]; yva = y[va]
        Xte_l = [X_l[i] for i in te]; yte = y[te]
        kte = [k_l[i] for i in te]

    n_feat = len(FEATURES)
    # FIX E3.5: scaler fit dari baris ASLI (sebelum padding), bukan dari
    # array yang sudah di-nolkan -- lihat catatan lengkap di
    # _pad_normalize_fit_train(). Sebelumnya StandardScaler.fit() dipanggil
    # SETELAH pad_list(), jadi baris nol buatan ikut mempengaruhi mean/std.
    sc = StandardScaler().fit(np.vstack(Xtr_l))

    def pad_and_scale(X_list):
        Xp, _ = pad_list(X_list, max_len)
        scaled = sc.transform(Xp.reshape(-1, n_feat)).reshape(Xp.shape)
        for i, x in enumerate(X_list):
            sl = min(len(x), max_len)
            if sl < max_len:
                scaled[i, sl:, :] = 0.0
        return scaled

    Xtr = pad_and_scale(Xtr_l)
    Xva = pad_and_scale(Xva_l)
    Xte = pad_and_scale(Xte_l)

    model = build_lstm(arch, (max_len, n_feat), seed=seed)
    model.compile(optimizer=Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # FIX D2/E3.3: batch_size untuk arch='source' DIKUNCI ke 16, sesuai
    # Table 2 paper [15] untuk LSTM_RELU_IO-L1 (bukan nilai default 32
    # yang sebelumnya dipakai untuk kedua arch -- 32 itu memang cocok
    # untuk arch='modified' sesuai desain manuskrip, tapi bukan untuk
    # source-faithful). epochs=100 sudah cocok untuk kedua arch, tidak
    # perlu override (default fungsi ini juga 100).
    effective_batch_size = 16 if arch == 'source' else batch_size
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
              batch_size=effective_batch_size, verbose=0,
              callbacks=[EarlyStopping('val_loss', patience=15,
                                       restore_best_weights=True),
                         ReduceLROnPlateau('val_loss', factor=0.5,
                                           patience=5, min_lr=1e-6)])
    y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
    tf.keras.backend.clear_session()

    te_episodes = [k[0] for k in kte]  # E0.3: id episode per window test
    m = compute_metrics(yte, y_pred, test_episodes=te_episodes)
    m.update(model='LSTM', arch=arch, protocol=protocol,
             cell=CELL_CODE[(arch, protocol)], seed=int(seed),
             n_train=int(len(Xtr)), n_test=int(len(Xte)))
    return m


# ---------- BERT ----------
#
# Konfigurasi diambil PERSIS dari dua tempat, tidak ditebak:
#   modified (D)         : Cell 40 notebook kamu (BERT-WordPiece, DistilBERT)
#   source-faithful (A/B): manuskrip Sec 3.3 -- "The source paper [16] uses
#                          BERT-base without discretisation, sets MAX_LEN to
#                          the maximum training sequence length, and uses a
#                          learning rate of 3e-5 over 5 epochs."
#
# ASUMSI yang TIDAK dinyatakan di manuskrip maupun paper sumber [16], dan
# harus dicek/dijustifikasi di response letter kalau reviewer bertanya:
#   - batch size source-faithful disamakan 16 (sama dengan modified), karena
#     paper sumber tidak menyebutkan angkanya
#   - early-stopping patience=3 dipakai untuk KEDUA arsitektur, karena paper
#     sumber tidak menyebutkan mekanisme regularisasi -- ini demi keadilan
#     perbandingan (kalau tidak, source-faithful bisa overfit tanpa kontrol)
#   - source-faithful TANPA diskretisasi berarti nilai fitur mentah
#     di-stringify sebagai teks (mis. "12.340 5.670 0.890 ...") dan dibiarkan
#     WordPiece tokenizer BERT-base memecahnya sendiri -- ini interpretasi
#     paling harfiah dari "without discretisation", bukan fakta yang
#     dikonfirmasi dari kode paper sumber (kode [16] tidak dirilis publik)

#  D5 (dikonfirmasi dosen): sel C DITAMBAHKAN (sebelumnya skip demi hemat
#  compute), dan seed dinaikkan ke 20 kalau anggaran komputasi memungkinkan.
#  CATATAN BIAYA: 4 sel x 20 seed = 80 training BERT per kecepatan (naik
#  dari 3 sel x 10 seed = 30 sebelumnya) -- turunkan BERT_SEEDS_DEFAULT
#  manual (mis. SEEDS[:10]) kalau run di Colab jadi terlalu lama.
BERT_CELLS = CELLS  # A, B, C, D penuh
BERT_SEEDS_DEFAULT = SEEDS  # 20 seed; WAJIB dicatat di caption tabel kalau diturunkan


def compute_feat_stats(df):
    """Min/max per fitur dari df (HARUS dari train partition saja)."""
    return {f: {'min': float(df[f].min()), 'max': float(df[f].max())}
            for f in FEATURES}


def compute_feat_stats_from_windows(X_list):
    """Sama seperti compute_feat_stats tapi dari list window (source_like)."""
    allv = np.vstack(X_list)
    return {f: {'min': float(allv[:, i].min()), 'max': float(allv[:, i].max())}
            for i, f in enumerate(FEATURES)}


def _tokens_source(window_2d, precision=3):
    """Source-faithful: TANPA diskretisasi, nilai mentah di-stringify."""
    parts = []
    for t in range(window_2d.shape[0]):
        for c in range(window_2d.shape[1]):
            parts.append(f'{window_2d[t, c]:.{precision}f}')
    return ' '.join(parts)


def _tokens_modified(window_2d, feat_stats, n_bins=20):
    """Modified: identik dengan features_to_tokens_v2() di notebook kamu."""
    short = ['SNR', 'RSSI', 'PDR', 'SPD', 'RSP']
    tokens = []
    for t in range(window_2d.shape[0]):
        for c, feat in enumerate(FEATURES):
            v = window_2d[t, c]
            fmin, fmax = feat_stats[feat]['min'], feat_stats[feat]['max']
            v_bin = 0 if (fmax - fmin) < 1e-9 else \
                int(np.clip((v - fmin) / (fmax - fmin) * n_bins, 0, n_bins - 1))
            tokens.append(f'{short[c]}{v_bin}')
    return ' '.join(tokens)


class _BertDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        import torch
        self.torch = torch
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True,
                             padding='max_length', max_length=self.max_len,
                             return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels': self.torch.tensor(int(self.labels[idx]), dtype=self.torch.long)}


def run_bert_cell(df_speed, seed, arch, protocol, window_sec=15, step_sec=1,
                  patience=3, batch_size=16, verbose=True,
                  modified_max_len=128):
    """
    Satu sel BERT. MAHAL -- source-faithful (BERT-base, 5 epoch) dan
    modified (DistilBERT, 10 epoch) dilatih dari nol tiap panggilan.

    arch='source'   : BERT-base, tanpa diskretisasi, MAX_LEN dinamis
                      (maks panjang token training, dibatasi 512), LR 3e-5,
                      5 epoch  -- manuskrip Sec 3.3
    arch='modified' : DistilBERT, diskretisasi 20-bin, MAX_LEN 128, LR 2e-5,
                      10 epoch  -- notebook Cell 40
    """
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (BertTokenizerFast, BertForSequenceClassification,
                              DistilBertTokenizerFast,
                              DistilBertForSequenceClassification)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if protocol == 'pcmef':
        tr_df, va_df, te_df = split_by_episode(df_speed, seed=seed)
        Xtr_l, ytr, _ = create_windows_keyed(tr_df, window_sec, step_sec)
        Xva_l, yva, _ = create_windows_keyed(va_df, window_sec, step_sec)
        Xte_l, yte, kte = create_windows_keyed(te_df, window_sec, step_sec)
        if min(len(Xtr_l), len(Xva_l), len(Xte_l)) == 0:
            return None
        feat_stats = compute_feat_stats(tr_df)
    else:  # source_like
        X_l, y, k_l = create_windows_keyed(df_speed, window_sec, step_sec)
        if len(X_l) == 0:
            return None
        tr, va, te = split_random_unit(len(X_l), y, seed=seed)
        Xtr_l = [X_l[i] for i in tr]; ytr = y[tr]
        Xva_l = [X_l[i] for i in va]; yva = y[va]
        Xte_l = [X_l[i] for i in te]; yte = y[te]
        kte = [k_l[i] for i in te]
        if min(len(Xtr_l), len(Xva_l), len(Xte_l)) == 0:
            return None
        feat_stats = compute_feat_stats_from_windows(Xtr_l)

    if arch == 'source':
        tr_texts = [_tokens_source(x) for x in Xtr_l]
        va_texts = [_tokens_source(x) for x in Xva_l]
        te_texts = [_tokens_source(x) for x in Xte_l]
        tok_name, ModelCls, TokCls = 'bert-base-uncased', BertForSequenceClassification, BertTokenizerFast
        lr, epochs = 3e-5, 5
        tmp_tok = TokCls.from_pretrained(tok_name)
        sample_lens = [len(tmp_tok.encode(t)) for t in tr_texts[:200]]
        max_len = int(min(512, max(sample_lens) if sample_lens else 64))
    else:
        tr_texts = [_tokens_modified(x, feat_stats, n_bins=20) for x in Xtr_l]
        va_texts = [_tokens_modified(x, feat_stats, n_bins=20) for x in Xva_l]
        te_texts = [_tokens_modified(x, feat_stats, n_bins=20) for x in Xte_l]
        tok_name, ModelCls, TokCls = 'distilbert-base-uncased', DistilBertForSequenceClassification, DistilBertTokenizerFast
        # E3.6: modified_max_len default 128 = konfigurasi manuskrip Sec 3.3.
        # E1.4 menemukan 64% window terpotong di 128 (p99 ~268-296 token);
        # nilai lain HANYA untuk pilot/ablasi -- WAJIB pakai out_dir
        # terpisah (nama file cache tidak memuat max_len; max_len tersimpan
        # di field 'max_len' hasil).
        lr, epochs, max_len = 2e-5, 10, int(modified_max_len)

    tokenizer = TokCls.from_pretrained(tok_name)
    model = ModelCls.from_pretrained(tok_name, num_labels=3).to(device)

    ds_tr = _BertDataset(tr_texts, ytr.tolist(), tokenizer, max_len)
    ds_va = _BertDataset(va_texts, yva.tolist(), tokenizer, max_len)
    ds_te = _BertDataset(te_texts, yte.tolist(), tokenizer, max_len)
    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    best_val, best_state, bad = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for batch in ld_tr:
            optimizer.zero_grad()
            out = model(input_ids=batch['input_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device),
                       labels=batch['labels'].to(device))
            out.loss.backward(); optimizer.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in ld_va:
                out = model(input_ids=batch['input_ids'].to(device),
                           attention_mask=batch['attention_mask'].to(device),
                           labels=batch['labels'].to(device))
                vloss += out.loss.item()
        vloss /= max(1, len(ld_va))
        if verbose:
            print(f'    [{arch}/{protocol} seed {seed}] epoch {ep+1}/{epochs} val_loss={vloss:.4f}')
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in ld_te:
            out = model(input_ids=batch['input_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device))
            preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy().tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    te_episodes = [k[0] for k in kte]  # E0.3: id episode per window test
    m = compute_metrics(np.array(yte), np.array(preds), test_episodes=te_episodes)
    m.update(model='BERT', arch=arch, protocol=protocol,
             cell=CELL_CODE[(arch, protocol)], seed=int(seed),
             n_train=len(tr_texts), n_test=len(te_texts), max_len=max_len)
    return m


def run_factorial(df_speed, target_speed, model_name, out_dir,
                  seeds=None, cells=None, **kw):
    """
    Run (or resume) every cell x seed for one model and one speed regime.

    One JSON per (model, speed, cell, seed) is written to out_dir, so the run
    can be resumed after a Colab disconnect. A cached JSON is reused only if
    its stored "code_hash" matches model_code_hash(model_name); stale files
    are ignored and recomputed.

    cells defaults to CELLS (the four cells A/B/C/D). For BERT pass
    cells=BERT_CELLS and seeds=BERT_SEEDS_DEFAULT (all 20 seeds); report any
    reduction in the table caption.
    """
    seeds = seeds or SEEDS
    cells = cells or CELLS
    os.makedirs(out_dir, exist_ok=True)
    runner = {'RF_Stat': run_rf_cell, 'LSTM': run_lstm_cell,
             'BERT': run_bert_cell}[model_name]
    results = []
    n_stale = 0
    for arch, proto in cells:
        code = CELL_CODE[(arch, proto)]
        for seed in seeds:
            fp = os.path.join(
                out_dir, f'{model_name}_{target_speed}ms_{code}_seed{seed}.json')
            if os.path.exists(fp):
                cached = json.load(open(fp))
                if cache_is_valid(cached, model_name):
                    results.append(cached)
                    continue
                else:
                    n_stale += 1  # hash tidak cocok -> hitung ulang
            r = runner(df_speed, seed, arch, proto, **kw)
            if r is None:
                print(f'  [{code} seed {seed}] dilewati (data tidak cukup)')
                continue
            r['target_speed'] = target_speed
            r['code_hash'] = model_code_hash(model_name)
            json.dump(r, open(fp, 'w'), indent=2)
            results.append(r)
            print(f'  [{code} seed {seed:4d}] acc={r["accuracy"]:.4f} '
                  f'f1={r["f1_macro"]:.4f}')
    if n_stale:
        print(f'  >> {n_stale} cached file(s) ignored (code hash mismatch) and recomputed.')
    return results


# ==========================================================================
# BAGIAN 4 -- SENSITIVITAS k-NN DENGAN GRAF TRAIN-ONLY (Reviewer C poin 7)
# ==========================================================================
#
# build_embeddings_cache() meniru PERSIS arsitektur Detector asli kamu
# (Cell 10 & 12 notebook): BiLSTM(128)->BiLSTM(64)->Dense(64)->BatchNorm->
# Dense(32) [embedding] -> Dense(3) [softmax], satu model per skala (1s/5s/15s),
# dilatih terpisah, embedding-nya diambil dari sub-model sebelum layer softmax.
#
# BEDA dari kode asli (ini bagian yang diperbaiki):
#   - embedding tiap skala disimpan BERSAMA kuncinya (episode, t0) lewat
#     create_windows_keyed -- bukan array polos yang nanti dipotong paksa
#     ke jumlah baris terkecil (bug align_min/A3)
#   - graf k-NN pakai SpatialKNN (fit train-only) -- bukan
#     compute_spatial_features(X_test) yang bocor (bug A4)
# Penyatuan skala 1s/5s ke acuan 15s dilakukan NANTI oleh align_multi() di
# dalam run_knn_sensitivity_fixed(), bukan di sini -- fungsi ini hanya
# menyediakan bahan mentahnya (embedding + kunci + label + fitur 15s + xgb proba).

def _pad_normalize_fit_train(Xtr_l, Xva_l, Xte_l):
    """
    FIX E3.5: StandardScaler SEKARANG di-fit HANYA dari baris ASLI (sebelum
    padding) di TRAIN, bukan dari array yang SUDAH di-nolkan. Sebelumnya,
    scaler di-fit SETELAH padding -- baris nol buatan (dari window yang
    lebih pendek dari max_len) ikut masuk hitungan mean/std, membuat
    statistik condong ke nol (bias makin besar kalau banyak window pendek).

    Urutan baru: fit scaler dari concat seluruh baris ASLI train -> scale
    tiap window (termasuk baris padding, supaya shape tetap konsisten) ->
    baris padding DIPAKSA NOL LAGI setelah scaling (karena 0 yang di-scale
    TIDAK otomatis 0 lagi kalau mean != 0), supaya extract_tabular_features()
    tetap bisa mendeteksi & mengecualikan baris padding lewat mask
    "(x == 0).all(axis=1)" yang sudah ada.
    """
    max_len = max(len(x) for x in Xtr_l)
    n_feat = Xtr_l[0].shape[1]

    def pad(X_list):
        out = np.zeros((len(X_list), max_len, n_feat), dtype=np.float32)
        for i, x in enumerate(X_list):
            sl = min(len(x), max_len)
            out[i, :sl, :] = x[:sl]
        return out

    sc = StandardScaler().fit(np.vstack(Xtr_l))  # HANYA baris asli, tanpa padding

    def scale_padded(X_list):
        padded = pad(X_list)
        scaled = sc.transform(padded.reshape(-1, n_feat)).reshape(padded.shape)
        for i, x in enumerate(X_list):
            sl = min(len(x), max_len)
            if sl < max_len:
                scaled[i, sl:, :] = 0.0  # re-zero baris padding setelah scaling
        return scaled

    Xtr = scale_padded(Xtr_l)
    Xva = scale_padded(Xva_l)
    Xte = scale_padded(Xte_l)
    return Xtr, Xva, Xte, max_len


def _build_temporal_bilstm(input_shape, embedding_dim=32, name='lstm', seed=42):
    """Sama persis dengan build_temporal_lstm() di notebook Cell 10."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Bidirectional, Dense,
                                         Dropout, BatchNormalization)
    tf.keras.utils.set_random_seed(seed)
    inputs = Input(shape=input_shape, name=f'{name}_input')
    x = Bidirectional(LSTM(128, return_sequences=True), name=f'{name}_bilstm1')(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False), name=f'{name}_bilstm2')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name=f'{name}_dense')(x)
    x = BatchNormalization()(x)
    embedding = Dense(embedding_dim, activation='relu', name=f'{name}_emb')(x)
    outputs = Dense(3, activation='softmax', name=f'{name}_out')(embedding)
    full = Model(inputs=inputs, outputs=outputs, name=f'{name}_full')
    emb = Model(inputs=inputs, outputs=embedding, name=f'{name}_e')
    return full, emb


def build_embeddings_cache(df_speed, seed, window_sizes=(1, 5, 15), step_sec=1,
                           embedding_dim=32, epochs=100, batch_size=32, verbose=True):
    """
    MAHAL -- melatih 3 BiLSTM (satu per skala) + 1 XGBoost tabular per seed.
    Panggil SEKALI per seed, lalu pakai cache-nya untuk semua nilai k di
    run_knn_sensitivity_fixed() -- itulah kenapa BiLSTM tidak perlu dilatih
    ulang untuk tiap k (arahan rencana 48 jam sebelumnya).
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from xgboost import XGBClassifier

    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    train_df, val_df, test_df = split_by_episode(df_speed, seed=seed)
    if verbose:
        print(f'    [cache seed {seed}] split: train={len(train_df)}, '
              f'val={len(val_df)}, test={len(test_df)}')

    keys, emb = {}, {}
    X15_padded = {}
    y_by_split = {}

    for ws in window_sizes:
        Xtr_l, ytr, ktr = create_windows_keyed(train_df, ws, step_sec)
        Xva_l, yva, kva = create_windows_keyed(val_df, ws, step_sec)
        Xte_l, yte, kte = create_windows_keyed(test_df, ws, step_sec)
        if min(len(Xtr_l), len(Xva_l), len(Xte_l)) == 0:
            return None

        Xtr_p, Xva_p, Xte_p, max_len = _pad_normalize_fit_train(Xtr_l, Xva_l, Xte_l)

        full, emb_model = _build_temporal_bilstm(
            (max_len, len(FEATURES)), embedding_dim, name=f'lstm_{ws}s_{seed}', seed=seed)
        full.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        cb = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
              ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]
        full.fit(Xtr_p, ytr, validation_data=(Xva_p, yva), epochs=epochs,
                batch_size=batch_size, callbacks=cb, verbose=0)
        _, acc = full.evaluate(Xte_p, yte, verbose=0)
        if verbose:
            print(f'    [cache seed {seed}] BiLSTM {ws}s single-scale test acc: {acc:.4f}')

        keys[ws] = {'train': ktr, 'val': kva, 'test': kte}
        emb[ws] = {'train': emb_model.predict(Xtr_p, verbose=0),
                  'val': emb_model.predict(Xva_p, verbose=0),
                  'test': emb_model.predict(Xte_p, verbose=0)}
        if ws == 15:
            X15_padded = {'train': Xtr_p, 'val': Xva_p, 'test': Xte_p}
            y_by_split = {'train': ytr, 'val': yva, 'test': yte}

        del full, emb_model
        tf.keras.backend.clear_session()

    # XGBoost tabular (35-dim) di atas window 15s -- identik konfigurasi
    # dengan xgb tabular di notebook (Cell 12), BUKAN meta-XGBoost
    Xt_tr = extract_tabular_features(X15_padded['train'])
    Xt_va = extract_tabular_features(X15_padded['val'])
    Xt_te = extract_tabular_features(X15_padded['test'])
    xgb = XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=200,
                        max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed,
                        eval_metric='mlogloss', early_stopping_rounds=20)
    xgb.fit(Xt_tr, y_by_split['train'], eval_set=[(Xt_va, y_by_split['val'])], verbose=False)
    xgb_proba = {'train': xgb.predict_proba(Xt_tr), 'val': xgb.predict_proba(Xt_va),
                'test': xgb.predict_proba(Xt_te)}
    if verbose:
        xgb_acc = float((xgb.predict(Xt_te) == y_by_split['test']).mean())
        print(f'    [cache seed {seed}] XGBoost-only tabular test acc: {xgb_acc:.4f}')

    return {'keys': keys, 'emb': emb, 'X15': X15_padded, 'y': y_by_split,
           'xgb_proba': xgb_proba}


def run_knn_full(df_speed, target_speed, out_dir, seeds=None,
                 k_values=(1, 3, 5, 7, 10, 15), window_sizes=(1, 5, 15), step_sec=1):
    """
    Runner resumable Poin 7 -- satu file JSON per seed, aman kalau Colab
    putus. Untuk tiap seed: build_embeddings_cache() SEKALI (mahal), lalu
    run_knn_sensitivity_fixed() memakai cache itu untuk semua k (murah).

    FIX E0.1: sama seperti run_factorial() -- file JSON lama hanya dipakai
    ulang kalau 'code_hash' di dalamnya cocok dengan CODE_HASH kode yang
    sedang dipakai sekarang. Ini PENTING khusus untuk Poin 7, karena E3.1
    (meta-learner dilatih di val) dan E3.4 (k-NN exclude same-episode)
    PERSIS mengubah fungsi yang dipanggil di sini -- hasil lama dari
    sebelum fix itu HARUS dihitung ulang, bukan dipakai ulang.
    """
    seeds = seeds or SEEDS
    os.makedirs(out_dir, exist_ok=True)
    results = []
    n_stale = 0
    for seed in seeds:
        fp = os.path.join(out_dir, f'KNN_{target_speed}ms_seed{seed}.json')
        if os.path.exists(fp):
            cached = json.load(open(fp))
            if cache_is_valid(cached, 'KNN'):
                results.append(cached)
                print(f'  [seed {seed:4d}] sudah ada (hash cocok), dipakai ulang')
                continue
            else:
                n_stale += 1
        cache = build_embeddings_cache(df_speed, seed, window_sizes=window_sizes,
                                       step_sec=step_sec)
        if cache is None:
            print(f'  [seed {seed}] dilewati (data tidak cukup)')
            continue
        r = run_knn_sensitivity_fixed(df_speed, seed, embeddings_cache=cache,
                                      k_values=k_values, step_sec=step_sec)
        r['seed'] = int(seed)
        r['target_speed'] = target_speed
        r['code_hash'] = model_code_hash('KNN')
        json.dump(r, open(fp, 'w'), indent=2)
        results.append(r)
        print(f'  [seed {seed:4d}] selesai -> {fp}')
    if n_stale:
        print(f'  >> {n_stale} file hasil lama DIABAIKAN (hash kode beda) dan dihitung ulang.')
    return results


def run_knn_sensitivity_fixed(df_speed, seed, embeddings_cache=None,
                              k_values=(1, 3, 5, 7, 10, 15), step_sec=1):
    """
    Menjalankan konfigurasi: no_knn, knn_branch_only, dan setiap k.

    Perbedaan dari versi lama:
      - graf dibangun dengan SpatialKNN (tetangga dari TRAIN saja)
      - cabang 1s / 5s / 15s disejajarkan lewat align_multi, bukan align_min
      - ada konfigurasi knn_branch_only untuk menjawab
        "apakah cabang ini punya sinyal sama sekali"

    embeddings_cache: dict hasil temporal embedding yang sudah dihitung
    sekali per seed, supaya BiLSTM tidak dilatih ulang untuk setiap k.
    Format: {'keys': {ws: keys}, 'emb': {ws: {'train':..,'val':..,'test':..}},
             'X15': {'train':..,'val':..,'test':..},
             'y': {'train':..,'val':..,'test':..},
             'xgb_proba': {'train':..,'val':..,'test':..}}
    """
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_class_weight

    c = embeddings_cache
    if c is None:
        raise ValueError('run_knn_sensitivity_fixed butuh embeddings_cache. '
                         'Hitung sekali per seed lalu pakai ulang untuk semua k.')

    keys_ref = c['keys'][15]['test']  # acuan penyejajaran

    def assemble(split, sp_arr):
        arrs = [(c['keys'][ws][split], c['emb'][ws][split]) for ws in WINDOW_SIZES]
        arrs.append((c['keys'][15][split], c['xgb_proba'][split]))
        if sp_arr is not None:
            arrs.append((c['keys'][15][split], sp_arr))
        aligned, keep, _ = align_multi(c['keys'][15][split], arrs)
        return np.hstack(aligned), c['y'][split][keep]

    def fit_meta(Ztr, ytr, Zva, yva, Zte, yte, min_val_for_internal_split=60):
        """
        FIX E3.1 (keputusan D6): meta-learner DILATIH DI VALIDASI, bukan di
        TRAIN. Base model (BiLSTM per-skala, XGBoost tabular, SpatialKNN)
        semuanya sudah dilatih/di-fit di atas TRAIN -- jadi embedding/
        predict-proba MEREKA SENDIRI yang dihasilkan dari input TRAIN sudah
        "dilihat" waktu base model dilatih (terlalu optimis/leaky kalau
        dipakai lagi buat melatih meta-learner di atasnya, sama seperti
        model dites dengan data yang sudah dipakai latihan). Validasi
        belum pernah dilihat base model manapun, jadi itu yang representatif
        untuk melatih stacking meta-learner. Ztr/ytr TIDAK dipakai sama
        sekali di sini (sengaja) -- dibiarkan di signature untuk
        kompatibilitas pemanggil.

        Val dipecah lagi secara internal (75/25, stratified) kalau cukup
        besar, supaya tetap ada early-stopping set; kalau terlalu kecil
        (< min_val_for_internal_split baris atau <2 kelas di pecahan),
        meta-learner dilatih di SELURUH val tanpa early stopping -- ini
        kompromi praktis dari opsi "cross-fitting" di dokumen D6 untuk val
        yang kecil (cross-fitting OOF-per-fold penuh belum diimplementasi
        di sini; minta ditambahkan kalau ukuran val ternyata kerap terlalu
        kecil di semua seed/kecepatan).
        """
        from sklearn.model_selection import train_test_split

        base_params = dict(objective='multi:softprob', num_class=3,
                           n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=3, gamma=0.1, reg_alpha=0.1,
                           reg_lambda=1.0, random_state=seed,
                           eval_metric='mlogloss')

        classes_va = np.unique(yva)
        can_split_internal = (len(yva) >= min_val_for_internal_split
                              and len(classes_va) >= 2
                              and min(np.bincount(yva.astype(int))[c]
                                     for c in classes_va.astype(int)) >= 4)

        if can_split_internal:
            Zfit, Zes, yfit, yes = train_test_split(
                Zva, yva, test_size=0.25, stratify=yva, random_state=seed)
        else:
            Zfit, yfit, Zes, yes = Zva, yva, None, None

        sc = RobustScaler().fit(Zfit)
        cw = compute_class_weight('balanced', classes=np.unique(yfit), y=yfit)
        sw = np.array([cw[list(np.unique(yfit)).index(v)] for v in yfit])

        if Zes is not None:
            m = XGBClassifier(**base_params, early_stopping_rounds=20)
            m.fit(sc.transform(Zfit), yfit, sample_weight=sw,
                  eval_set=[(sc.transform(Zes), yes)], verbose=False)
        else:
            m = XGBClassifier(**base_params)
            m.fit(sc.transform(Zfit), yfit, sample_weight=sw, verbose=False)

        yp = m.predict(sc.transform(Zte))
        _, _, f1, _ = precision_recall_fscore_support(yte, yp, average='macro',
                                                      zero_division=0)
        return float(accuracy_score(yte, yp)), float(f1)

    out = {}

    # no_knn
    Ztr, ytr = assemble('train', None)
    Zva, yva = assemble('val', None)
    Zte, yte = assemble('test', None)
    acc, f1 = fit_meta(Ztr, ytr, Zva, yva, Zte, yte)
    out['no_knn'] = dict(accuracy=acc, macro_f1=f1, graph_density=0.0)

    for k in k_values:
        # FIX E3.4: fit dengan episode id train, transform(train,...) dengan
        # episode id query yang sama -- supaya tetangga dari episode yang
        # sama (bukan cuma baris identik) ikut dibuang untuk is_train=True.
        train_episodes = [key[0] for key in c['keys'][15]['train']]
        knn = SpatialKNN(k=k).fit(c['X15']['train'], train_episodes=train_episodes)
        sp = {}
        for s in ['train', 'val', 'test']:
            q_eps = train_episodes if s == 'train' else None
            sp[s] = knn.transform(c['X15'][s], is_train=(s == 'train'),
                                  query_episodes=q_eps)

        Ztr, ytr = assemble('train', sp['train'])
        Zva, yva = assemble('val', sp['val'])
        Zte, yte = assemble('test', sp['test'])
        acc, f1 = fit_meta(Ztr, ytr, Zva, yva, Zte, yte)
        out[f'k={k}'] = dict(accuracy=acc, macro_f1=f1,
                             graph_density=knn.graph_density(len(c['X15']['test'])))

        # branch-only: cabang k-NN sendirian, untuk menjawab apakah
        # cabang ini punya sinyal prediktif sama sekali
        if k == 5:
            sc = RobustScaler().fit(sp['train'])
            mb = XGBClassifier(objective='multi:softprob', num_class=3,
                               n_estimators=300, max_depth=4, learning_rate=0.05,
                               random_state=seed, eval_metric='mlogloss')
            mb.fit(sc.transform(sp['train']), c['y']['train'], verbose=False)
            yp = mb.predict(sc.transform(sp['test']))
            _, _, f1b, _ = precision_recall_fscore_support(
                c['y']['test'], yp, average='macro', zero_division=0)
            out['knn_branch_only_k5'] = dict(
                accuracy=float(accuracy_score(c['y']['test'], yp)),
                macro_f1=float(f1b), graph_density=out[f'k={k}']['graph_density'])

    return out


# ==========================================================================
# BAGIAN 5 -- AGREGASI & STATISTIK
# ==========================================================================

def bootstrap_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return (float('nan'), float('nan'))
    bm = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return float(lo), float(hi)


def paired_cohens_d(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


def unpaired_cohens_d(a, b):
    """E3.7. Cohen's d untuk dua sampel TIDAK berpasangan (pooled std)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1)
    pooled_std = np.sqrt(pooled_var)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def bootstrap_ci_unpaired(a, b, n_boot=10000, seed=0):
    """
    E3.7. Bootstrap CI untuk beda rata-rata dua sampel yang TIDAK
    berpasangan: a dan b di-resample INDEPENDEN (masing-masing dengan
    penggantian), bukan me-resample selisih per-seed seperti bootstrap_ci()
    biasa. Dipakai untuk Delta_protocol (B-A), karena A dan B adalah dua
    skema split/isi-test yang berbeda pada seed yang sama -- paired-by-seed
    tapi TIDAK paired-by-sample.
    """
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        am = a[rng.randint(0, len(a), len(a))].mean()
        bm = b[rng.randint(0, len(b), len(b))].mean()
        diffs[i] = am - bm
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_interaction_unpaired(vA, vB, vC, vD, n_boot=10000, seed=0):
    """
    E3.7. Bootstrap CI untuk interaksi (D-B)-(C-A), dengan KEEMPAT sel
    di-resample INDEPENDEN (bukan resampling selisih-dari-selisih yang
    sudah dipasangkan per seed seperti sebelumnya) -- konsisten dengan
    Delta_protocol yang sekarang unpaired.
    """
    rng = np.random.RandomState(seed)
    vA, vB, vC, vD = (np.asarray(x, float) for x in (vA, vB, vC, vD))
    out = np.empty(n_boot)
    for i in range(n_boot):
        a = vA[rng.randint(0, len(vA), len(vA))].mean()
        b = vB[rng.randint(0, len(vB), len(vB))].mean()
        c = vC[rng.randint(0, len(vC), len(vC))].mean()
        d = vD[rng.randint(0, len(vD), len(vD))].mean()
        out[i] = (d - b) - (c - a)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def holm_bonferroni(pvalues):
    """
    E3.7. Koreksi Holm-Bonferroni untuk satu KELUARGA pengujian (dipakai di
    decompose_factorial -- keluarga = {delta_protocol, delta_model,
    interaction} dalam satu pemanggilan -- dan di aggregate_knn --
    keluarga = semua nilai k dibandingkan ke no_knn). Mengembalikan array
    p-value TERKOREKSI dengan urutan SAMA seperti input (bukan urutan
    setelah di-sort).
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((n - rank) * p[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def decompose_factorial(results, metric='accuracy'):
    """
    Hitung Delta_protocol, Delta_model, dan interaction dari hasil 2x2
    penuh (A/B/C/D). Dipakai untuk LSTM dan BERT.

        A = source model  + source-like protocol
        B = source model  + PCMEF
        C = modified model + source-like protocol
        D = modified model + PCMEF

        Delta_protocol = B - A   (model dikunci ke source-faithful)
        Delta_model    = D - B   (protokol dikunci ke PCMEF)
        interaction    = (D - B) - (C - A)

    Semua dalam percentage point, paired per seed (nomor seed yang sama
    dipasangkan lintas sel).

    CATATAN PAIRING (dari dosen): pemasangan ini paired-by-seed, bukan
    paired-by-sample. Delta_protocol (B-A) membandingkan dua ISI SPLIT
    yang berbeda pada seed yang sama, bukan performa pada test set yang
    identik. Ini valid sebagai estimasi rata-rata efek protokol, tapi
    laporkan dengan hati-hati: A dan B adalah dua skema evaluasi berbeda,
    bukan before/after pada sampel test yang sama.
    """
    by_cell = {}
    for r in results:
        by_cell.setdefault(r['cell'], {})[r['seed']] = r[metric] * 100

    common = None
    for cell, d in by_cell.items():
        s = set(d)
        common = s if common is None else (common & s)
    common = sorted(common or [])
    if not common:
        return {'error': 'tidak ada seed yang lengkap di semua sel'}

    v = {cell: np.array([by_cell[cell][s] for s in common])
         for cell in by_cell}

    summary = {'n_seeds': len(common), 'seeds': common, 'cells': {}}
    for cell, arr in v.items():
        summary['cells'][cell] = dict(mean=float(arr.mean()),
                                      std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)

    def delta_paired(name, a, b):
        """Delta_model (D-B): TETAP paired-by-sample (B,D sama-sama di
        bawah protokol PCMEF, jadi isi split identik -> boleh dipasangkan
        per seed, sesuai E3.7 poin 1)."""
        if a not in v or b not in v:
            return
        d = v[a] - v[b]
        lo, hi = bootstrap_ci(d)
        t, p = sps.ttest_rel(v[a], v[b])
        try:
            _, pw = sps.wilcoxon(v[a], v[b])
        except ValueError:
            pw = float('nan')
        summary[name] = dict(mean_pp=float(d.mean()), ci95_pp=[round(lo, 3), round(hi, 3)],
                             p_ttest=float(p), p_wilcoxon=float(pw),
                             cohens_d=paired_cohens_d(v[a], v[b]), paired=True)

    def delta_unpaired(name, a, b):
        """FIX E3.7 poin 2: Delta_protocol (B-A) sekarang UNPAIRED -- Welch t
        (equal_var=False), Mann-Whitney U, bootstrap unpaired. A dan B isi
        split/test-nya berbeda meski seed-nya sama, jadi tidak boleh
        diperlakukan sebagai sampel berpasangan."""
        if a not in v or b not in v:
            return
        lo, hi = bootstrap_ci_unpaired(v[a], v[b])
        t, p = sps.ttest_ind(v[a], v[b], equal_var=False)  # Welch
        try:
            _, pw = sps.mannwhitneyu(v[a], v[b], alternative='two-sided')
        except ValueError:
            pw = float('nan')
        summary[name] = dict(mean_pp=float(v[a].mean() - v[b].mean()),
                             ci95_pp=[round(lo, 3), round(hi, 3)],
                             p_ttest=float(p), p_mannwhitney=float(pw),
                             cohens_d=unpaired_cohens_d(v[a], v[b]), paired=False)

    delta_unpaired('delta_protocol', 'B', 'A')
    delta_paired('delta_model', 'D', 'B')
    if all(c in v for c in ['A', 'B', 'C', 'D']):
        # FIX E3.7 poin 3: interaksi juga dihitung dengan bootstrap UNPAIRED
        # (resampling independen per sel), konsisten dengan Delta_protocol.
        inter_mean = (v['D'].mean() - v['B'].mean()) - (v['C'].mean() - v['A'].mean())
        lo, hi = bootstrap_ci_interaction_unpaired(v['A'], v['B'], v['C'], v['D'])
        summary['interaction'] = dict(mean_pp=float(inter_mean),
                                      ci95_pp=[round(lo, 3), round(hi, 3)])

    # FIX E3.7 poin 4: koreksi Holm per KELUARGA -- keluarga di sini adalah
    # {delta_protocol, delta_model} dalam satu pemanggilan decompose_factorial
    # ini (interaction tidak punya p-value uji, cuma CI, jadi tidak diikutkan).
    fam_names = [k for k in ('delta_protocol', 'delta_model') if k in summary]
    if fam_names:
        pvals = [summary[k]['p_ttest'] for k in fam_names]
        adj = holm_bonferroni(pvals)
        for k, pa in zip(fam_names, adj):
            summary[k]['p_ttest_holm'] = float(pa)
    return summary


def print_factorial(summary, title='', metric_label='Accuracy'):
    print('\n' + '=' * 78)
    print(f'  PROTOCOL x MODEL DECOMPOSITION (full 2x2, cells A/B/C/D)  {title}')
    print('=' * 78)
    if 'error' in summary:
        print('  ' + summary['error'])
        return
    print(f'  n seeds = {summary["n_seeds"]}\n')
    print(f'  {"Cell":6s} {"Model":18s} {"Protocol":22s} {metric_label:>16s}')
    desc = {'A': ('source-faithful', 'source-like protocol'),
            'B': ('source-faithful', 'PCMEF'),
            'C': ('modified', 'source-like protocol'),
            'D': ('modified', 'PCMEF')}
    for cell in ['A', 'B', 'C', 'D']:
        if cell in summary['cells']:
            c = summary['cells'][cell]
            m, p = desc[cell]
            print(f'  {cell:6s} {m:18s} {p:22s} {c["mean"]:8.2f} ± {c["std"]:5.2f}')
    print()
    for key, lab in [('delta_protocol', 'Δ protocol  (B−A)'),
                     ('delta_model', 'Δ model     (D−B)'),
                     ('interaction', 'Interaction       ')]:
        if key in summary:
            d = summary[key]
            line = (f'  {lab:22s} {d["mean_pp"]:+7.2f} pp   '
                    f'CI95 [{d["ci95_pp"][0]:+.2f}, {d["ci95_pp"][1]:+.2f}]')
            if 'p_ttest' in d:
                line += f'   p={d["p_ttest"]:.4f}'
                if 'p_ttest_holm' in d:
                    line += f' (Holm={d["p_ttest_holm"]:.4f})'
                line += f'  d={d["cohens_d"]:+.2f}'
            print(line)
    print('\n  Reading: a CI that excludes zero = effect detected at this number of seeds.')
    print('  Δ protocol is measured with the model fixed, Δ model with the protocol fixed.')
    print('  Δ protocol (B−A) is UNPAIRED (Welch t, Mann-Whitney U, unpaired bootstrap):')
    print('  A and B have different split/test contents even for the same seed.')
    print('  Δ model (D−B) is paired, since B and D share the same PCMEF test sets.')
    print('  Holm = Holm-Bonferroni correction over {Δ protocol, Δ model} in this table.')


def aggregate_knn(results_per_seed, n_boot=10000):
    """results_per_seed: list dict hasil run_knn_sensitivity_fixed() (boleh
    ada metadata tambahan seperti 'seed'/'target_speed' dari run_knn_full,
    itu otomatis dilewati)."""
    cfgs = {}
    for r in results_per_seed:
        for name, v in r.items():
            if not isinstance(v, dict) or 'accuracy' not in v:
                continue  # lewati metadata non-konfigurasi (seed, target_speed)
            cfgs.setdefault(name, {'acc': [], 'f1': [], 'dens': []})
            cfgs[name]['acc'].append(v['accuracy'])
            cfgs[name]['f1'].append(v['macro_f1'])
            cfgs[name]['dens'].append(v.get('graph_density', 0.0))
    base = np.array(cfgs['no_knn']['acc'])
    order = ['no_knn', 'knn_branch_only_k5'] + [f'k={k}' for k in (1, 3, 5, 7, 10, 15)]
    k_family = [f'k={k}' for k in (1, 3, 5, 7, 10, 15)]  # E5.6/E3.7: satu keluarga Holm
    summary = {}
    for name in order:
        if name not in cfgs:
            continue
        a = np.array(cfgs[name]['acc']) * 100
        f = np.array(cfgs[name]['f1']) * 100
        n = min(len(a), len(base))
        d = a[:n] - base[:n] * 100
        lo, hi = bootstrap_ci(d, n_boot=n_boot)
        entry = dict(n_seeds=len(a),
                    acc_mean=float(a.mean()),
                    acc_std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    f1_mean=float(f.mean()),
                    f1_std=float(f.std(ddof=1)) if len(f) > 1 else 0.0,
                    delta_pp=float(d.mean()), ci95_pp=[round(lo, 3), round(hi, 3)],
                    graph_density=float(np.mean(cfgs[name]['dens'])))
        # FIX E3.7/E5.6 poin 5: uji PAIRED terhadap no_knn (seed yang sama
        # dipasangkan -- no_knn dan k=... dilatih di atas split IDENTIK
        # per seed, jadi sah dipasangkan), lalu Holm per keluarga nilai k.
        if name in k_family and n > 1:
            t, p = sps.ttest_rel(a[:n], base[:n] * 100)
            try:
                _, pw = sps.wilcoxon(a[:n], base[:n] * 100)
            except ValueError:
                pw = float('nan')
            entry['p_ttest_vs_no_knn'] = float(p)
            entry['p_wilcoxon_vs_no_knn'] = float(pw)
        summary[name] = entry

    fam_pvals = [summary[k]['p_ttest_vs_no_knn'] for k in k_family
                if k in summary and 'p_ttest_vs_no_knn' in summary[k]]
    fam_keys = [k for k in k_family if k in summary and 'p_ttest_vs_no_knn' in summary[k]]
    if fam_pvals:
        adj = holm_bonferroni(fam_pvals)
        for k, pa in zip(fam_keys, adj):
            summary[k]['p_ttest_holm_vs_no_knn'] = float(pa)

    print('\n' + '=' * 92)
    print('  SENSITIVITAS k-NN (graf train-only)')
    print('=' * 92)
    print(f'  {"Config":20s} {"Accuracy":>16s} {"Macro-F1":>16s} '
          f'{"Δ vs No-kNN":>13s} {"95% CI":>20s}  {"p (Holm)":>14s}')
    for name in order:
        if name not in summary:
            continue
        s = summary[name]
        p_str = ''
        if 'p_ttest_holm_vs_no_knn' in s:
            p_str = f'  p={s["p_ttest_vs_no_knn"]:.4f} (Holm={s["p_ttest_holm_vs_no_knn"]:.4f})'
        print(f'  {name:20s} {s["acc_mean"]:7.2f} ± {s["acc_std"]:4.2f}  '
              f'{s["f1_mean"]:7.2f} ± {s["f1_std"]:4.2f}  '
              f'{s["delta_pp"]:+9.2f} pp  '
              f'[{s["ci95_pp"][0]:+.2f}, {s["ci95_pp"][1]:+.2f}]{p_str}')
    print('\n  Kalau semua CI memuat nol / p Holm > 0.05, kalimat yang boleh ditulis')
    print('  adalah: "tidak ditemukan kontribusi prediktif tambahan pada konstruksi')
    print('  graf dan rentang k yang diuji" -- BUKAN "cabang ini tidak punya sinyal".')
    print('  Baris knn_branch_only_k5 adalah bukti langsung untuk klaim itu.')
    print('  p Holm dihitung atas keluarga {k=1,3,5,7,10,15} vs no_knn (E3.7/E5.6).')
    return summary


# ==========================================================================
# EXAMPLE
# ==========================================================================
if __name__ == '__main__':
    print(__doc__)



# ==========================================================================
# E4.1 -- hybrid model, ablations and baselines on the SAME test-window keys,
# with per-window predictions saved.
# ==========================================================================
# Nothing in this section is part of _COMMON_FUNCS / _MODEL_FUNCS, so it does not
# change model_code_hash(): existing RF-Stat / LSTM / BERT / kNN result files
# stay valid.
#
# Implementation notes:
#  - The meta-learner is trained on VALIDATION windows (_e41_fit_meta); base
#    models never see validation data.
#  - SpatialKNN excludes same-episode neighbours for training queries; the
#    reference set is always TRAIN only.
#  - Scaling is fit on real rows before padding (_pad_normalize_fit_train);
#    padded rows are re-zeroed. The BiLSTMs have no Masking layer.
#  - df_speed must already be filtered with filter_feasible_episodes(
#    min_duration_sec=15) by the caller.


E41_META_CONFIGS = {
    'hybrid_full':   ('emb_1s', 'emb_5s', 'emb_15s', 'xgb_proba', 'knn_descriptor'),
    'multi_no_knn':  ('emb_1s', 'emb_5s', 'emb_15s', 'xgb_proba'),
    'multi_no_xgb':  ('emb_1s', 'emb_5s', 'emb_15s', 'knn_descriptor'),
    'multi_emb_only': ('emb_1s', 'emb_5s', 'emb_15s'),
}
E41_DIRECT_CONFIGS = ('bilstm_1s', 'bilstm_5s', 'bilstm_15s', 'xgb_only')
E41_BASELINES = ('RF_Stat', 'LSTM', 'BERT')   # semuanya sel D (modified + PCMEF)
E41_VERSION = 'v1'


def _e41_capture(fn, *args, **kw):
    """Jalankan fn (run_rf_cell / run_lstm_cell / run_bert_cell) sambil
    MENANGKAP (y_true, y_pred) yang dioper ke compute_metrics, TANPA mengubah
    source fungsi-fungsi itu (jadi hash cache-nya tidak berubah)."""
    orig = globals()['compute_metrics']
    store = {}

    def _cap(y_true, y_pred, *a, **k):
        store['y_true'] = np.asarray(y_true)
        store['y_pred'] = np.asarray(y_pred)
        return orig(y_true, y_pred, *a, **k)

    globals()['compute_metrics'] = _cap
    try:
        m = fn(*args, **kw)
    finally:
        globals()['compute_metrics'] = orig
    return m, store.get('y_true'), store.get('y_pred')


def e41_build_cache(df_speed, seed, window_sizes=(1, 5, 15), step_sec=1,
                    embedding_dim=32, epochs=100, batch_size=32, verbose=True):
    """Sama dengan build_embeddings_cache(), tetapi juga menyimpan softmax
    tiap BiLSTM (utk baris ablation 'BiLSTM single-scale') dan id episode."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from xgboost import XGBClassifier

    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    train_df, val_df, test_df = split_by_episode(df_speed, seed=seed)

    keys, emb, probs = {}, {}, {}
    X15_padded, y_by_split = {}, {}
    for ws in window_sizes:
        Xtr_l, ytr, ktr = create_windows_keyed(train_df, ws, step_sec)
        Xva_l, yva, kva = create_windows_keyed(val_df, ws, step_sec)
        Xte_l, yte, kte = create_windows_keyed(test_df, ws, step_sec)
        if min(len(Xtr_l), len(Xva_l), len(Xte_l)) == 0:
            return None
        Xtr_p, Xva_p, Xte_p, max_len = _pad_normalize_fit_train(Xtr_l, Xva_l, Xte_l)
        full, emb_model = _build_temporal_bilstm(
            (max_len, len(FEATURES)), embedding_dim, name=f'e41_{ws}s_{seed}', seed=seed)
        full.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        cb = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
              ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]
        full.fit(Xtr_p, ytr, validation_data=(Xva_p, yva), epochs=epochs,
                 batch_size=batch_size, callbacks=cb, verbose=0)
        keys[ws] = {'train': ktr, 'val': kva, 'test': kte}
        emb[ws] = {'train': emb_model.predict(Xtr_p, verbose=0),
                   'val': emb_model.predict(Xva_p, verbose=0),
                   'test': emb_model.predict(Xte_p, verbose=0)}
        probs[ws] = {'train': full.predict(Xtr_p, verbose=0),
                     'val': full.predict(Xva_p, verbose=0),
                     'test': full.predict(Xte_p, verbose=0)}
        if ws == 15:
            X15_padded = {'train': Xtr_p, 'val': Xva_p, 'test': Xte_p}
            y_by_split = {'train': ytr, 'val': yva, 'test': yte}
        del full, emb_model
        tf.keras.backend.clear_session()

    Xt = {s: extract_tabular_features(X15_padded[s]) for s in ('train', 'val', 'test')}
    xgb = XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=200,
                        max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed,
                        eval_metric='mlogloss', early_stopping_rounds=20)
    xgb.fit(Xt['train'], y_by_split['train'], eval_set=[(Xt['val'], y_by_split['val'])],
            verbose=False)
    xgb_proba = {s: xgb.predict_proba(Xt[s]) for s in ('train', 'val', 'test')}
    return {'keys': keys, 'emb': emb, 'probs': probs, 'X15': X15_padded,
            'y': y_by_split, 'xgb_proba': xgb_proba}


def _e41_fit_meta(Zva, yva, Zte, seed, min_val_for_internal_split=60):
    """Meta-learner XGBoost dilatih di VALIDATION (E3.1). Identik dengan
    fit_meta() di run_knn_sensitivity_fixed(), tetapi mengembalikan prediksi
    per window + feature importance + info ukuran validasi (Langkah 2.5)."""
    from xgboost import XGBClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split

    if len(np.unique(yva)) < 2:
        raise ValueError('validasi (setelah align) hanya punya <2 kelas')
    params = dict(objective='multi:softprob', num_class=3, n_estimators=300,
                  max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
                  eval_metric='mlogloss')
    classes_va = np.unique(yva)
    counts = np.bincount(yva.astype(int))
    internal = (len(yva) >= min_val_for_internal_split and len(classes_va) >= 2
                and min(counts[c] for c in classes_va.astype(int)) >= 4)
    if internal:
        Zfit, Zes, yfit, yes = train_test_split(Zva, yva, test_size=0.25,
                                                stratify=yva, random_state=seed)
    else:
        Zfit, yfit, Zes, yes = Zva, yva, None, None
    sc = RobustScaler().fit(Zfit)
    cw = compute_class_weight('balanced', classes=np.unique(yfit), y=yfit)
    sw = np.array([cw[list(np.unique(yfit)).index(v)] for v in yfit])
    if Zes is not None:
        m = XGBClassifier(**params, early_stopping_rounds=20)
        m.fit(sc.transform(Zfit), yfit, sample_weight=sw,
              eval_set=[(sc.transform(Zes), yes)], verbose=False)
    else:
        m = XGBClassifier(**params)
        m.fit(sc.transform(Zfit), yfit, sample_weight=sw, verbose=False)
    pred = m.predict(sc.transform(Zte))
    return pred, m.feature_importances_, dict(n_val=int(len(yva)), internal_split=bool(internal),
                                             val_class_counts=[int(x) for x in counts])


def e41_run_seed(df_speed, seed, hybrid_k=5, step_sec=1, epochs=100, batch_size=32,
                 baselines=('RF_Stat', 'LSTM'), verbose=True):
    """
    Satu seed E4.1. Mengembalikan (hasil_json, arrays) atau None.
    Semua konfigurasi dinilai pada KUNCI test yang sama K = window 15 s test
    yang punya pasangan di skala 1 s dan 5 s (kunci yang sama dipakai hibrida).
    Baseline (RF-Stat D, LSTM D, BERT D) dilatih ulang pada split yang sama,
    lalu prediksinya diambil pada K saja. n_test diassert identik.
    """
    c = e41_build_cache(df_speed, seed, step_sec=step_sec, epochs=epochs,
                        batch_size=batch_size, verbose=verbose)
    if c is None:
        return None
    WS = (1, 5, 15)
    k15 = {s: c['keys'][15][s] for s in ('val', 'test')}
    train_eps = [k[0] for k in c['keys'][15]['train']]
    knn = SpatialKNN(k=hybrid_k).fit(c['X15']['train'], train_episodes=train_eps)
    sp = {s: knn.transform(c['X15'][s], is_train=False) for s in ('val', 'test')}

    def aligned(split):
        arrs = ([(c['keys'][w][split], c['emb'][w][split]) for w in WS]
                + [(k15[split], c['xgb_proba'][split]), (k15[split], sp[split])]
                + [(c['keys'][w][split], c['probs'][w][split]) for w in WS])
        out, keep, kept = align_multi(k15[split], arrs)
        blocks = dict(emb_1s=out[0], emb_5s=out[1], emb_15s=out[2], xgb_proba=out[3],
                      knn_descriptor=out[4], p_1s=out[5], p_5s=out[6], p_15s=out[7])
        y = c['y'][split][keep]
        return blocks, y, keep, kept

    bva, yva, keep_va, _ = aligned('val')
    bte, yte, keep_te, kept_te = aligned('test')
    if len(yte) == 0:
        return None

    configs, importance, info = {}, {}, {}
    for name, blocks in E41_META_CONFIGS.items():
        Zva = np.hstack([bva[b] for b in blocks]); Zte = np.hstack([bte[b] for b in blocks])
        pred, fi, inf = _e41_fit_meta(Zva, yva, Zte, seed)
        widths = [bva[b].shape[1] for b in blocks]; bnd = np.cumsum([0] + widths)
        tot = float(fi.sum()) or 1.0
        importance[name] = {b: float(fi[bnd[i]:bnd[i + 1]].sum() / tot) for i, b in enumerate(blocks)}
        info[name] = inf
        configs[name] = pred
    configs['bilstm_1s'] = bte['p_1s'].argmax(1)
    configs['bilstm_5s'] = bte['p_5s'].argmax(1)
    configs['bilstm_15s'] = bte['p_15s'].argmax(1)
    configs['xgb_only'] = bte['xgb_proba'].argmax(1)

    # ---- baseline pada split yang sama, dievaluasi pada kunci K saja ----
    tr_df, va_df, te_df = split_by_episode(df_speed, seed=seed)
    _, y_ref, k_ref = create_windows_keyed(te_df, 15, step_sec)
    assert list(k_ref) == list(k15['test']), 'urutan kunci test beda dari cache'
    fns = {'RF_Stat': run_rf_cell, 'LSTM': run_lstm_cell, 'BERT': run_bert_cell}
    for bname in baselines:
        _, yt, yp = _e41_capture(fns[bname], df_speed, seed, 'modified', 'pcmef', step_sec=step_sec)
        if yp is None:
            continue
        assert len(yp) == len(y_ref) and np.array_equal(yt, y_ref), \
            f'{bname}: y_true tidak sejajar dengan kunci test'
        configs[f'{bname}_D'] = yp[keep_te]

    n_test = int(len(yte))
    for name, pred in configs.items():
        assert len(pred) == n_test, f'n_test tidak identik utk {name}: {len(pred)} vs {n_test}'

    res = dict(seed=int(seed), n_test_common=n_test, n_test_native=int(len(keep_te)),
               n_val_aligned=int(len(yva)), meta_info=info, block_importance=importance,
               hybrid_k=int(hybrid_k), e41_version=E41_VERSION,
               code_hash=model_code_hash('KNN'), metrics={})
    for name, pred in configs.items():
        m = compute_metrics(yte, pred, test_episodes=[k[0] for k in kept_te])
        res['metrics'][name] = {k: m[k] for k in ('accuracy', 'f1_macro', 'balanced_accuracy',
                                                  'confusion_matrix') if k in m}
    arrays = dict(ep=np.array([k[0] for k in kept_te]), t0=np.array([k[1] for k in kept_te]),
                  y_true=np.asarray(yte), **{f'pred_{n}': np.asarray(p) for n, p in configs.items()})
    return res, arrays


def e41_run_all(df_speed, target_speed, out_dir, seeds=None, **kw):
    """Runner resumable E4.1: satu JSON + satu NPZ (prediksi per window) per seed."""
    seeds = seeds or SEEDS
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for seed in seeds:
        fp = os.path.join(out_dir, f'E41_{target_speed}ms_seed{seed}.json')
        fz = os.path.join(out_dir, f'E41_{target_speed}ms_seed{seed}.npz')
        if os.path.exists(fp) and os.path.exists(fz):
            cached = json.load(open(fp))
            if cached.get('e41_version') == E41_VERSION and cache_is_valid(cached, 'KNN'):
                results.append(cached)
                print(f'  [seed {seed:4d}] sudah ada, dipakai ulang')
                continue
        try:
            out = e41_run_seed(df_speed, seed, **kw)
        except ValueError as e:
            print(f'  [seed {seed}] DILEWATI -- {e}')
            continue
        if out is None:
            print(f'  [seed {seed}] dilewati (data tidak cukup)')
            continue
        r, arr = out
        r['target_speed'] = target_speed
        np.savez_compressed(fz, **arr)
        json.dump(r, open(fp, 'w'), indent=2)
        results.append(r)
        print(f'  [seed {seed:4d}] n_test={r["n_test_common"]} (native {r["n_test_native"]}) '
              f'hybrid acc={r["metrics"]["hybrid_full"]["accuracy"]*100:.2f}%')
    return results


def e41_paired_table(vals, ref, others, metric='accuracy', n_boot=10000):
    """vals: {config: {seed: dict metrik}}; membandingkan tiap `others` terhadap
    `ref`, dipasangkan per seed (n_test identik). Mengembalikan DataFrame
    dengan selisih (pp), CI bootstrap, p paired-t, p Wilcoxon, d, p Holm."""
    rows = []
    for o in others:
        seeds = sorted(set(vals[ref]) & set(vals.get(o, {})))
        if len(seeds) < 3:
            continue
        a = np.array([vals[ref][s][metric] for s in seeds]) * 100
        b = np.array([vals[o][s][metric] for s in seeds]) * 100
        d = a - b
        lo, hi = bootstrap_ci(d, n_boot=n_boot)
        t, p_t = sps.ttest_rel(a, b)
        try:
            p_w = sps.wilcoxon(a, b)[1]
        except ValueError:
            p_w = float('nan')
        rows.append(dict(vs=o, n_seed=len(seeds), ref_mean=a.mean(), other_mean=b.mean(),
                         diff_pp=d.mean(), ci_lo=lo, ci_hi=hi, p_t=p_t, p_wilcoxon=p_w,
                         cohens_d=paired_cohens_d(a, b)))
    df = pd.DataFrame(rows)
    if len(df):
        pp = df['p_wilcoxon'].fillna(df['p_t']).values
        df['p_holm'] = holm_bonferroni(pp)
    return df


# ==========================================================================
# BiLSTM-15s on the NATIVE test population (all 15 s test windows)
# ==========================================================================
# E4.1 stores predictions only on the common-key subset. This trains the 15 s
# BiLSTM alone (same recipe as e41_build_cache) and returns predictions for
# every native test window, so native vs. subset can be compared for the SAME
# model. It is a new model (different RNG stream): do not mix its predictions
# with those saved by E4.1.

def e41_bilstm15_native(df_speed, seed, step_sec=1, embedding_dim=32, epochs=100, batch_size=32):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    train_df, val_df, test_df = split_by_episode(df_speed, seed=seed)
    Xtr_l, ytr, _ = create_windows_keyed(train_df, 15, step_sec)
    Xva_l, yva, _ = create_windows_keyed(val_df, 15, step_sec)
    Xte_l, yte, kte = create_windows_keyed(test_df, 15, step_sec)
    if min(len(Xtr_l), len(Xva_l), len(Xte_l)) == 0:
        return None
    Xtr_p, Xva_p, Xte_p, max_len = _pad_normalize_fit_train(Xtr_l, Xva_l, Xte_l)
    full, _emb = _build_temporal_bilstm((max_len, len(FEATURES)), embedding_dim,
                                        name=f'e41n_15s_{seed}', seed=seed)
    full.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]
    full.fit(Xtr_p, ytr, validation_data=(Xva_p, yva), epochs=epochs,
             batch_size=batch_size, callbacks=cb, verbose=0)
    pred = full.predict(Xte_p, verbose=0).argmax(1)
    tf.keras.backend.clear_session()
    return dict(y_true=np.asarray(yte), y_pred=np.asarray(pred), keys=kte)
