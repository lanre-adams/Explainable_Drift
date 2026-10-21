"""Explanation-drift pilot engine.

Question: can label-free monitoring signals tell when an intrusion detector is
failing (harmful drift) without raising alarms when traffic merely changes
(harmless drift)?

Detector: XGBoost trained on part of NSL-KDD KDDTrain+. Reference set: held-out
validation records. Each monitoring window holds `window_size` records; the
monitors never see labels. Window error (from labels) is used only to score them.

Monitors
  INPUT   mean per-feature two-sample KS statistic, raw features vs reference
  EXPL    naive explanation drift: mean per-feature Wasserstein distance between
          window and reference TreeSHAP values, scaled by reference SHAP spread
  EXPLc   class-conditional explanation drift: EXPL computed within each
          predicted class, weighted by the window's predicted-class shares
  CONF    confidence-based estimate (ATC; Garg et al., 2022): share of window
          predictions below the confidence threshold matching validation error
  CONFORM inductive conformal evaluation: share of window predictions whose
          conformal credibility (p-value of the predicted label, nonconformity
          1 - p(predicted class), class-conditional calibration) is below alpha

Scenarios
  A  harmful drift: windows mixing validation records with an increasing share
     of shifted test records (KDDTest+ / KDDTest-21)
  B  harmless drift: validation records only, with the attack share changed;
     error stays near validation level, so any alarm is a false alarm

Each monitor is mapped to an error estimate with an isotonic fit on that seed's
scenario-A windows. A window "fails" when its true error exceeds `fail_threshold`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MONITORS = ["INPUT", "EXPL", "EXPLc", "CONF", "CONFORM"]
MONITOR_LABELS = {
    "INPUT": "Input drift (KS)",
    "EXPL": "Explanation drift, naive",
    "EXPLc": "Explanation drift, class-conditional",
    "CONF": "Confidence (ATC)",
    "CONFORM": "Conformal credibility",
}
TEST_SETS = {"KDDTest+": "KDDTest+.txt", "KDDTest-21": "KDDTest-21.txt"}


@dataclass
class Config:
    seeds: int = 3
    window_size: int = 1000
    reps: int = 4
    mix_steps: int = 10            # harmful-drift mixing shares 0, 1/steps, ..., 1
    test_sets: list = field(default_factory=lambda: ["KDDTest+", "KDDTest-21"])
    benign_shares: list = field(default_factory=lambda: [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    n_estimators: int = 200
    max_depth: int = 6
    reference_size: int = 5000
    fail_threshold: float = 0.05   # window error above this = failing
    conformal_alpha: float = 0.05

    def validate(self) -> "Config":
        self.seeds = int(min(max(self.seeds, 1), 10))
        self.window_size = int(min(max(self.window_size, 200), 3000))
        self.reps = int(min(max(self.reps, 1), 10))
        self.mix_steps = int(min(max(self.mix_steps, 2), 20))
        self.test_sets = [t for t in self.test_sets if t in TEST_SETS] or ["KDDTest+"]
        self.benign_shares = sorted({float(min(max(s, 0.01), 0.99)) for s in self.benign_shares}) or [0.2, 0.5, 0.8]
        self.n_estimators = int(min(max(self.n_estimators, 20), 1000))
        self.max_depth = int(min(max(self.max_depth, 2), 12))
        self.reference_size = int(min(max(self.reference_size, 1000), 10000))
        self.fail_threshold = float(min(max(self.fail_threshold, 0.005), 0.5))
        self.conformal_alpha = float(min(max(self.conformal_alpha, 0.001), 0.5))
        return self

    def total_windows(self) -> int:
        a = len(self.test_sets) * (self.mix_steps + 1) * self.reps
        b = len(self.benign_shares) * self.reps
        return self.seeds * (a + b)


_COLS = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
_CACHE: dict = {}


def load_data():
    if "data" in _CACHE:
        return _CACHE["data"]
    frames = {"train": pd.read_csv(DATA_DIR / "KDDTrain+.txt", header=None, names=_COLS)}
    for name, f in TEST_SETS.items():
        frames[name] = pd.read_csv(DATA_DIR / f, header=None, names=_COLS)
    allX = pd.concat([d.drop(columns=["label", "difficulty"]) for d in frames.values()])
    for c in ["f1", "f2", "f3"]:  # protocol, service, flag
        allX[c] = allX[c].astype("category").cat.codes
    out, start = {}, 0
    for name, d in frames.items():
        n = len(d)
        out[name] = (allX.iloc[start:start + n].values.astype(float), (d.label != "normal").astype(int).values)
        start += n
    _CACHE["data"] = out
    return out


def _wd(A, B, sd):
    return float(np.mean([wasserstein_distance(A[:, j], B[:, j]) / sd[j] for j in range(A.shape[1])]))


def run_pilot(cfg: Config, progress: Callable[[int, int, str], None] | None = None) -> dict:
    cfg.validate()
    data = load_data()
    Xtr, ytr = data["train"]
    total, done = cfg.total_windows(), 0
    rows, timings, train_info = [], {m: [] for m in MONITORS + ["PREDICT", "SHAP"]}, []
    t0 = time.time()

    def tick(msg):
        nonlocal done
        done += 1
        if progress and (done % 5 == 0 or done == total):
            progress(done, total, msg)

    for seed in range(cfg.seeds):
        rng = np.random.default_rng(seed)
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
        if progress:
            progress(done, total, f"Seed {seed + 1}/{cfg.seeds}: training detector")
        model = xgb.XGBClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=0.1,
                                  subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=4,
                                  eval_metric="logloss").fit(Xa, ya)
        explainer = shap.TreeExplainer(model)
        idx = rng.permutation(len(Xv))
        ref_i, pool_i = idx[:cfg.reference_size], idx[cfg.reference_size:]
        Xref, yref = Xv[ref_i], yv[ref_i]
        Sref = explainer.shap_values(Xref)
        sd = Sref.std(0) + 1e-9
        pref = model.predict_proba(Xref)[:, 1]
        yref_hat = (pref > 0.5).astype(int)
        err_ref = float(np.mean(yref_hat != yref))
        conf_ref = np.maximum(pref, 1 - pref)
        tau = float(np.quantile(conf_ref, err_ref))
        sdc = {c: Sref[yref_hat == c].std(0) + 1e-9 for c in (0, 1)}
        # conformal calibration: nonconformity of the TRUE label on reference, per class
        nc_ref = 1 - np.where(yref == 1, pref, 1 - pref)
        cal = {c: np.sort(nc_ref[yref == c]) for c in (0, 1)}
        train_info.append(dict(seed=seed, val_error=err_ref, val_fpr=float(yref_hat[yref == 0].mean()),
                               val_dr=float(yref_hat[yref == 1].mean())))

        def signals(Xw):
            t = time.perf_counter(); pw = model.predict_proba(Xw)[:, 1]; timings["PREDICT"].append(time.perf_counter() - t)
            yh = (pw > 0.5).astype(int)
            t = time.perf_counter(); Sw = explainer.shap_values(Xw); timings["SHAP"].append(time.perf_counter() - t)
            s = {}
            t = time.perf_counter()
            s["INPUT"] = float(np.mean([ks_2samp(Xref[:, j], Xw[:, j]).statistic for j in range(Xw.shape[1])]))
            timings["INPUT"].append(time.perf_counter() - t)
            t = time.perf_counter(); s["EXPL"] = _wd(Sref, Sw, sd); timings["EXPL"].append(time.perf_counter() - t)
            t = time.perf_counter()
            s["EXPLc"] = float(sum((yh == c).mean() * _wd(Sref[yref_hat == c], Sw[yh == c], sdc[c])
                                   for c in (0, 1) if (yh == c).sum() > 5))
            timings["EXPLc"].append(time.perf_counter() - t)
            t = time.perf_counter(); s["CONF"] = float(np.mean(np.maximum(pw, 1 - pw) < tau)); timings["CONF"].append(time.perf_counter() - t)
            t = time.perf_counter()
            nc_w = 1 - np.maximum(pw, 1 - pw)
            cred = np.empty(len(Xw))
            for c in (0, 1):
                m = yh == c
                if m.any():
                    cal_c = cal[c]
                    cred[m] = (len(cal_c) - np.searchsorted(cal_c, nc_w[m], side="left") + 1) / (len(cal_c) + 1)
            s["CONFORM"] = float(np.mean(cred < cfg.conformal_alpha))
            timings["CONFORM"].append(time.perf_counter() - t)
            return yh, s

        for tname in cfg.test_sets:
            XT, yT = data[tname]
            for p in np.round(np.linspace(0, 1, cfg.mix_steps + 1), 3):
                for r in range(cfg.reps):
                    k = int(round(p * cfg.window_size))
                    a = rng.choice(pool_i, cfg.window_size - k, replace=False)
                    b = rng.choice(len(XT), k, replace=False)
                    Xw = np.vstack([Xv[a], XT[b]]); yw = np.concatenate([yv[a], yT[b]])
                    yh, s = signals(Xw)
                    neg, pos = yw == 0, yw == 1
                    rows.append(dict(seed=seed, scenario="harmful", test_set=tname, share=float(p), rep=r,
                                     error=float(np.mean(yh != yw)),
                                     fpr=float(yh[neg].mean()) if neg.any() else np.nan,
                                     detection_rate=float(yh[pos].mean()) if pos.any() else np.nan,
                                     attack_share=float(pos.mean()), val_error=err_ref, **s))
                    tick(f"Seed {seed + 1}/{cfg.seeds}: harmful drift, {tname}, share {p:.0%}")
        pos_pool, neg_pool = pool_i[yv[pool_i] == 1], pool_i[yv[pool_i] == 0]
        for sh in cfg.benign_shares:
            for r in range(cfg.reps):
                k = int(round(sh * cfg.window_size))
                ii = np.concatenate([rng.choice(pos_pool, k, replace=False),
                                     rng.choice(neg_pool, cfg.window_size - k, replace=False)])
                yw = yv[ii]
                yh, s = signals(Xv[ii])
                rows.append(dict(seed=seed, scenario="harmless", test_set="validation", share=float(sh), rep=r,
                                 error=float(np.mean(yh != yw)), fpr=float(yh[yw == 0].mean()),
                                 detection_rate=float(yh[yw == 1].mean()), attack_share=float(yw.mean()),
                                 val_error=err_ref, **s))
                tick(f"Seed {seed + 1}/{cfg.seeds}: harmless drift, attack share {sh:.0%}")

    df = pd.DataFrame(rows)
    for seed in range(cfg.seeds):
        A = df[(df.seed == seed) & (df.scenario == "harmful")]
        sel = df.seed == seed
        for m in MONITORS:
            iso = IsotonicRegression(out_of_bounds="clip").fit(A[m], A.error)
            df.loc[sel, "est_" + m] = iso.predict(df.loc[sel, m])
    df["failing"] = (df.error > cfg.fail_threshold).astype(int)
    return summarise(df, cfg, timings, train_info, time.time() - t0)


def _ci(vals):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)])
    if len(v) == 0:
        return None, None, None
    m = float(v.mean())
    h = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, m - h, m + h


def summarise(df, cfg, timings, train_info, elapsed):
    A, B = df[df.scenario == "harmful"], df[df.scenario == "harmless"]
    monitors = []
    for m in MONITORS:
        rho = [spearmanr(g[m], g.error).correlation for _, g in A.groupby("seed")]
        auc, fa, est_b = [], [], []
        for seed, g in df.groupby("seed"):
            if 0 < g.failing.sum() < len(g):
                auc.append(roc_auc_score(g.failing, g[m]))
            b = g[g.scenario == "harmless"]
            fa.append(float((b["est_" + m] > cfg.fail_threshold).mean()))
            est_b.append(float(b["est_" + m].mean()))
        mae = [float((g["est_" + m] - g.error).abs().mean()) for _, g in A.groupby("seed")]
        t_ms = 1000 * float(np.mean(timings[m]))
        if m in ("EXPL", "EXPLc"):
            t_ms += 1000 * float(np.mean(timings["SHAP"]))
        r, a, f, e, mm = _ci(rho), _ci(auc), _ci(fa), _ci(est_b), _ci(mae)
        monitors.append(dict(key=m, label=MONITOR_LABELS[m],
                             rho=r[0], rho_lo=r[1], rho_hi=r[2],
                             auroc=a[0], auroc_lo=a[1], auroc_hi=a[2],
                             false_alarm=f[0], false_alarm_lo=f[1], false_alarm_hi=f[2],
                             benign_est_error=e[0], mae_harmful=mm[0],
                             ms_per_window=t_ms, needs_shap=m in ("EXPL", "EXPLc")))
    harm_curve = (A.groupby(["test_set", "share"])[["error", "fpr", "detection_rate"] + ["est_" + m for m in MONITORS]]
                  .mean().reset_index())
    benign_curve = (B.groupby("share")[["error"] + ["est_" + m for m in MONITORS]].mean().reset_index())
    return dict(
        config=asdict(cfg),
        elapsed_s=elapsed,
        n_windows=dict(harmful=int(len(A)), harmless=int(len(B))),
        detector=dict(val_error=float(np.mean([t["val_error"] for t in train_info])),
                      val_fpr=float(np.mean([t["val_fpr"] for t in train_info])),
                      val_dr=float(np.mean([t["val_dr"] for t in train_info])),
                      worst_harmful=A[A.share == A.share.max()].groupby("test_set")[["error", "fpr", "detection_rate"]]
                      .mean().reset_index().to_dict("records"),
                      benign_true_error_max=float(B.error.max())),
        monitors=monitors,
        harm_curve=harm_curve.round(5).to_dict("records"),
        benign_curve=benign_curve.round(5).to_dict("records"),
        timing_ms=dict(predict=1000 * float(np.mean(timings["PREDICT"])), shap=1000 * float(np.mean(timings["SHAP"]))),
        _windows=df,
    )
