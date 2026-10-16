"""
pcmef_corrected_stats.py
======================================================================
Uji terkoreksi untuk evaluasi multi-seed yang test set-nya SALING
TUMPANG TINDIH antar-seed (split_by_episode membagi ulang 39 episode
yang sama di setiap seed, sehingga 20 seed BUKAN 20 replikasi
independen).

Dasar: corrected resampled t-test, Nadeau & Bengio (2003),
"Inference for the Generalization Error", Machine Learning 52(3).
Varians naif s^2/J diganti dengan (1/J + n_test/n_train) * s^2.
Faktor n_test/n_train adalah HEURISTIK untuk korelasi antar-split,
bukan nilai eksak; laporkan sebagai uji konservatif-pendekatan.

Catatan penerapan:
- Paired (Delta model D-B, TOST hybrid vs pembanding): penerapan
  standar Nadeau-Bengio.
- Unpaired (Delta protokol B-A, interaksi): APROKSIMASI. Varians tiap
  sel dikoreksi dengan rasionya sendiri lalu dijumlahkan, konsisten
  dengan bootstrap unpaired di pcmef_fixed.decompose_factorial.
  Sebut ini eksplisit di Methods.

Rasio n_test/n_train:
- PCMEF (split_by_episode, 60/20/20 per kelas): pakai jumlah EPISODE,
  bukan window. Kosmanos 39 episode -> 6/24 = 0.25. Hitung dengan
  episode_ratio(df, P) supaya tidak salah tulis.
- source-like (split_random_unit, 60/20/20 di level unit): 0.2/0.6.

Pemakaian (di notebook, setelah import pcmef_fixed as P):
    import pcmef_corrected_stats as CS
    r_pc = CS.episode_ratio(dfs15, P)                 # ~0.25
    sm = CS.decompose_factorial_corrected(res_bert15_288, 'f1_macro',
                                          ratio_pcmef=r_pc)
    CS.print_corrected(sm, 'BERT 15 m/s macro-F1 (C,D @288)')
"""

import numpy as np
from scipy import stats as sps

RATIO_SOURCE_LIKE = 0.2 / 0.6


def episode_ratio(df, P, seed=42):
    """n_test/n_train dalam jumlah episode untuk split_by_episode."""
    tr, _, te = P.split_by_episode(df, seed=seed)
    return te['pseudo_run_id'].nunique() / tr['pseudo_run_id'].nunique()


def _holm(p):
    p = np.asarray(p, float); n = len(p); out = np.empty(n); run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min((n - rank) * p[i], 1.0)); out[i] = run
    return out


def nb_paired(a, b, ratio, alpha=0.05):
    """Selisih a-b berpasangan per seed, Nadeau-Bengio terkoreksi."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d); m = d.mean(); s2 = d.var(ddof=1)
    se_naive = np.sqrt(s2 / J)
    se = np.sqrt((1.0 / J + ratio) * s2)
    t = m / se if se > 0 else np.inf
    p = float(2 * sps.t.sf(abs(t), J - 1))
    tc = sps.t.ppf(1 - alpha / 2, J - 1)
    return dict(mean_pp=float(m), se_naive=float(se_naive), se_corr=float(se),
                ci_corr=[float(m - tc * se), float(m + tc * se)], t=float(t),
                p_corr=p, df=J - 1, inflation=float(se / se_naive) if se_naive else np.nan)


def nb_unpaired(a, b, ratio_a, ratio_b, alpha=0.05):
    """Aproksimasi unpaired: mean(a)-mean(b), varians per sel dikoreksi
    dengan rasionya sendiri; derajat bebas Welch-Satterthwaite."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    Ja, Jb = len(a), len(b)
    va = (1.0 / Ja + ratio_a) * a.var(ddof=1)
    vb = (1.0 / Jb + ratio_b) * b.var(ddof=1)
    m = a.mean() - b.mean(); se = np.sqrt(va + vb)
    dfree = (va + vb) ** 2 / (va ** 2 / (Ja - 1) + vb ** 2 / (Jb - 1))
    t = m / se if se > 0 else np.inf
    p = float(2 * sps.t.sf(abs(t), dfree))
    tc = sps.t.ppf(1 - alpha / 2, dfree)
    se_naive = np.sqrt(a.var(ddof=1) / Ja + b.var(ddof=1) / Jb)
    return dict(mean_pp=float(m), se_naive=float(se_naive), se_corr=float(se),
                ci_corr=[float(m - tc * se), float(m + tc * se)], t=float(t),
                p_corr=p, df=float(dfree), inflation=float(se / se_naive) if se_naive else np.nan)


def nb_interaction(vA, vB, vC, vD, rA, rB, rC, rD, alpha=0.05):
    """(D-B)-(C-A), keempat sel diperlakukan independen (aproksimasi)."""
    cells = [(np.asarray(x, float), r) for x, r in ((vA, rA), (vB, rB), (vC, rC), (vD, rD))]
    var = [(1.0 / len(x) + r) * x.var(ddof=1) for x, r in cells]
    m = (cells[3][0].mean() - cells[1][0].mean()) - (cells[2][0].mean() - cells[0][0].mean())
    se = np.sqrt(sum(var))
    dfree = sum(var) ** 2 / sum(v ** 2 / (len(x) - 1) for v, (x, _) in zip(var, cells))
    t = m / se if se > 0 else np.inf
    tc = sps.t.ppf(1 - alpha / 2, dfree)
    return dict(mean_pp=float(m), se_corr=float(se), ci_corr=[float(m - tc * se), float(m + tc * se)],
                t=float(t), p_corr=float(2 * sps.t.sf(abs(t), dfree)), df=float(dfree))


def decompose_factorial_corrected(results, metric='accuracy', ratio_pcmef=0.25,
                                  ratio_source_like=RATIO_SOURCE_LIKE):
    """Pendamping pcmef_fixed.decompose_factorial: struktur input sama
    (list dict JSON hasil run_factorial), menambahkan uji terkoreksi.
    Holm atas {delta_protocol, delta_model, interaction}."""
    by = {}
    for r in results:
        by.setdefault(r['cell'], {})[r['seed']] = r[metric] * 100
    common = sorted(set.intersection(*[set(d) for d in by.values()]))
    v = {c: np.array([by[c][s] for s in common]) for c in by}
    rr = {'A': ratio_source_like, 'C': ratio_source_like, 'B': ratio_pcmef, 'D': ratio_pcmef}
    out = dict(metric=metric, n_seeds=len(common),
               cells={c: dict(mean=float(x.mean()), sd=float(x.std(ddof=1))) for c, x in v.items()})
    if 'A' in v and 'B' in v:
        out['delta_protocol'] = nb_unpaired(v['B'], v['A'], rr['B'], rr['A'])
    if 'B' in v and 'D' in v:
        out['delta_model'] = nb_paired(v['D'], v['B'], ratio_pcmef)
    if all(c in v for c in 'ABCD'):
        out['interaction'] = nb_interaction(v['A'], v['B'], v['C'], v['D'],
                                            rr['A'], rr['B'], rr['C'], rr['D'])
    fam = [k for k in ('delta_protocol', 'delta_model', 'interaction') if k in out]
    for k, ph in zip(fam, _holm([out[k]['p_corr'] for k in fam])):
        out[k]['p_corr_holm'] = float(ph)
    return out


def tost_paired_corrected(a, b, margin=2.0, ratio=0.25):
    """TOST berpasangan dengan SE Nadeau-Bengio. Ekuivalen bila CI 90%
    terkoreksi berada di dalam (-margin, +margin)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d); m = d.mean(); se = np.sqrt((1.0 / J + ratio) * d.var(ddof=1))
    p_lo = sps.t.sf((m + margin) / se, J - 1)
    p_hi = sps.t.sf((margin - m) / se, J - 1)
    tc = sps.t.ppf(0.95, J - 1)
    return dict(n=J, diff_pp=float(m), ci90_corr=[float(m - tc * se), float(m + tc * se)],
                p_tost_corr=float(max(p_lo, p_hi)))


def print_corrected(sm, title=''):
    print(f"\n=== {title} | metrik={sm['metric']} | n seed={sm['n_seeds']} ===")
    for c in sorted(sm['cells']):
        print(f"  {c}: {sm['cells'][c]['mean']:6.2f} ± {sm['cells'][c]['sd']:5.2f}")
    for k in ('delta_protocol', 'delta_model', 'interaction'):
        if k in sm:
            r = sm[k]
            infl = f"  SE x{r['inflation']:.2f}" if 'inflation' in r else ''
            print(f"  {k:15s} {r['mean_pp']:+7.2f} pp  CI95 terkoreksi "
                  f"[{r['ci_corr'][0]:+.2f}, {r['ci_corr'][1]:+.2f}]  "
                  f"p={r['p_corr']:.4f}  Holm={r['p_corr_holm']:.4f}{infl}")
