"""
jamshield_per_file.py -- kinerja per rekaman (per file) JamShield di bawah
leave-one-file-out (LOFO), untuk Section 4.7 dan jawaban Reviewer E.

Mengulang LOFO E5.10 dengan pengaturan yang SAMA (fit_eval, RF_PARAMS, seed=42),
tetapi menyimpan prediksi per window sehingga akurasi bisa dihitung per file.
Karena label setiap file konstan, akurasi per file = proporsi window file itu
yang diklasifikasikan benar (untuk file benign = 1 - false-positive rate,
untuk file attack = recall/detection rate pada file tersebut).

PEMAKAIAN (notebook 05_jamshield_validation):
  1. Jalankan sel-sel awal sampai df_fixed tersedia, lalu sel E5.10
     (sel yang mendefinisikan Xall, yall, file_of_window, fit_eval).
     LOFO_pooled.json yang sudah ada membuat sel itu tidak melatih ulang;
     yang dibutuhkan hanya variabelnya.
  2. Sel baru:
        import shutil, sys
        shutil.copy('/content/drive/MyDrive/Detector_Results/jamshield_per_file.py',
                    '/content/jamshield_per_file.py')
        sys.path.insert(0, '/content')
        import jamshield_per_file as JPF
        per_file = JPF.run(Xall, yall, file_of_window, fit_eval, OUT_E510)
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


def run(Xall, yall, file_of_window, fit_eval, out_dir, seed=42, benign_label=None):
    yall = np.asarray(yall)
    file_of_window = np.asarray(file_of_window)
    files = sorted(set(file_of_window))

    # label per file (harus konstan); label benign ditebak dari nama file bila tidak diberikan
    file_label = {f: int(pd.Series(yall[file_of_window == f]).mode()[0]) for f in files}
    mixed = [f for f in files if len(np.unique(yall[file_of_window == f])) > 1]
    if mixed:
        print('PERINGATAN: file dengan label campuran:', mixed)
    if benign_label is None:
        benign_files = [f for f in files if 'benign' in str(f).lower()]
        benign_label = file_label[benign_files[0]] if benign_files else 0
    print(f'Label benign = {benign_label} | jumlah file = {len(files)} | jumlah window = {len(yall)}')

    rows, yt_pool, yp_pool = [], [], []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        te = file_of_window == f
        tr = ~te
        if len(np.unique(yall[tr])) < 2 or te.sum() == 0:
            print(f'  lewati {f}'); continue
        pred = np.asarray(fit_eval(Xall[tr], yall[tr], Xall[te], seed=seed))
        yte = yall[te]
        yt_pool.append(yte); yp_pool.append(pred)
        rows.append(dict(
            file=str(f),
            label='benign' if file_label[f] == benign_label else 'attack',
            n_windows=int(te.sum()),
            n_correct=int((pred == yte).sum()),
            accuracy_pct=round(100 * float((pred == yte).mean()), 2),
        ))
        print(f'  [{i:2d}/{len(files)}] {f}: akurasi {rows[-1]["accuracy_pct"]:.2f}% '
              f'(n={rows[-1]["n_windows"]})  {time.time() - t0:.0f} s')

    yt = np.concatenate(yt_pool); yp = np.concatenate(yp_pool)
    pooled = dict(balanced_accuracy=round(100 * balanced_accuracy_score(yt, yp), 2),
                  f1_macro=round(100 * f1_score(yt, yp, average='macro'), 2),
                  n_pooled=int(len(yt)))
    print(f'\nPooled (harus = Tabel 8, LOFO: 67.39 / 68.87): '
          f'balanced acc {pooled["balanced_accuracy"]}  macro-F1 {pooled["f1_macro"]}')

    df = pd.DataFrame(rows).sort_values(['label', 'accuracy_pct'])
    print('\n=== Akurasi per file (LOFO) ===')
    print(df.to_string(index=False))

    summ = df.groupby('label').agg(n_files=('file', 'count'),
                                   mean_acc=('accuracy_pct', 'mean'),
                                   min_acc=('accuracy_pct', 'min'),
                                   max_acc=('accuracy_pct', 'max'),
                                   windows=('n_windows', 'sum')).round(2)
    # akurasi ter-bobot window per kelas = recall kelas pada prediksi pooled
    summ['pooled_recall'] = [round(100 * float((yp[yt == (benign_label if lab == 'benign' else 1 - benign_label)]
                                                == (benign_label if lab == 'benign' else 1 - benign_label)).mean()), 2)
                             for lab in summ.index]
    print('\n=== Ringkasan per kelas ===')
    print(summ.to_string())

    df.to_csv(os.path.join(out_dir, 'LOFO_per_file.csv'), index=False)
    json.dump(dict(pooled=pooled, per_file=rows, summary=summ.reset_index().to_dict('records')),
              open(os.path.join(out_dir, 'LOFO_per_file.json'), 'w'), indent=2)
    print(f'\nDisimpan: {out_dir}/LOFO_per_file.csv dan LOFO_per_file.json')
    return df
