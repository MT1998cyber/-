import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------- Manually edit before running ----------
REGIONAL_FEATURE_CSVS = [
    # one entry per step-6 run, in any order; which rows are "train" comes from SPLIT_COL
    r"REGIONAL_FEATURE_CSV_DIR\regional_features_all.csv",
]
CLINICAL_CSV = r"CLINICAL_CSV_DIR"
ID_COL = "ID"                 # <Center>__<PatientID>, same as step-6 CaseID
SPLIT_COL = "group"           # any column holding your split
TRAIN_VALUES = ["train"]      # rows used to FIT everything below; all other rows are only transformed
LABEL_COL = "label"
OUT_DIR = r"OUTPUT_DIR"

MAX_MISSING_FRAC = 0.10       # e.g. a habitat absent in many small hematomas
HARMONIZE = True
# Centers that never occur in train (true external centers) cannot use train-fitted
# center parameters. "self": estimate that center's location/scale from its own
# feature values (label-free, transductive). "none": leave it unharmonized.
UNSEEN_CENTER_MODE = "self"
P_THRESHOLD = 0.05
CORR_THRESHOLD = 0.9
N_FOLDS = 10
N_C = 60                      # grid size for the inverse regularisation strength C
LASSO_RULE = "1se"            # "min" (best CV AUC) or "1se" (sparsest C within 1 SE – more stable)
MAX_FEATURES = None           # e.g. 16 to force a fixed branch width (keeps largest |coef|)
SEED = 0
ID_COLUMNS = ["CaseID", "Center", "PatientID"]
from config_loader import apply_config
apply_config("step8", globals())   # values from pipeline_config.py (if present)


# ----------------------------------------------------------------------------- harmonization
class LocationScaleHarmonizer:
    """ComBat-style per-center location/scale adjustment WITHOUT empirical-Bayes shrinkage
    and without covariates (the HE label must not be used, it is unknown at test time).
    Each center is mapped to the pooled train mean/SD:  x' = (x - mu_c) / sd_c * sd_pool + mu_pool.
    With dozens of patients per center the EB shrinkage of full ComBat changes little;
    swap in neuroCombat if you need the exact original algorithm."""

    def fit(self, X: pd.DataFrame, centers: pd.Series):
        self.mu_pool_ = X.mean()
        self.sd_pool_ = X.std(ddof=1).replace(0, 1.0)
        self.params_ = {}
        for c, idx in X.groupby(centers).groups.items():
            self.params_[c] = self._center_stats(X.loc[idx])
        return self

    def _center_stats(self, Xc):
        mu = Xc.mean()
        sd = Xc.std(ddof=1)
        if len(Xc) < 3:
            sd = self.sd_pool_.copy()
        sd = sd.where(sd > 1e-12, self.sd_pool_)
        return mu, sd

    def transform(self, X: pd.DataFrame, centers: pd.Series, unseen="self"):
        out = X.copy()
        log = {}
        for c, idx in X.groupby(centers).groups.items():
            if c in self.params_:
                mu, sd = self.params_[c]
                log[c] = "train-fitted"
            elif unseen == "self":
                mu, sd = self._center_stats(X.loc[idx])
                log[c] = "self-estimated (unseen center)"
            else:
                log[c] = "not harmonized (unseen center)"
                continue
            out.loc[idx] = (X.loc[idx] - mu) / sd * self.sd_pool_ + self.mu_pool_
        return out, log


# ----------------------------------------------------------------------------- helpers
def load_features():
    frames = []
    for p in REGIONAL_FEATURE_CSVS:
        f = pd.read_csv(p, dtype={"CaseID": str, "Center": str, "PatientID": str}, encoding="utf-8-sig")
        frames.append(f)
    feats = pd.concat(frames, ignore_index=True)
    dup = feats["CaseID"][feats["CaseID"].duplicated()]
    if len(dup):
        raise ValueError(f"Same CaseID in several regional tables: {dup.tolist()[:10]}")
    clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    for c in (ID_COL, SPLIT_COL, LABEL_COL):
        if c not in clin.columns:
            raise ValueError(f"Column {c!r} not found in {CLINICAL_CSV}")
    splits = pd.DataFrame({"CaseID": clin[ID_COL].astype(str).str.strip(),
                           "split": clin[SPLIT_COL].astype(str).str.strip(),
                           "label": clin[LABEL_COL]})
    if splits["CaseID"].duplicated().any():
        raise ValueError(f"Duplicated IDs in clinical CSV: {splits.CaseID[splits.CaseID.duplicated()].tolist()[:10]}")
    if not splits["split"].isin(TRAIN_VALUES).any():
        raise ValueError(f"No rows with {SPLIT_COL} in {TRAIN_VALUES}; values present: {sorted(splits.split.unique())}")
    tr = splits["split"].isin(TRAIN_VALUES)
    if splits.loc[tr, "label"].isna().any() or not splits.loc[tr, "label"].isin([0, 1]).all():
        raise ValueError("Training rows need a 0/1 label without missing values")
    merged = splits.merge(feats, on="CaseID", how="inner")
    missing = sorted(set(splits.CaseID) - set(feats.CaseID))
    extra = sorted(set(feats.CaseID) - set(splits.CaseID))
    if missing:
        print(f"WARNING: {len(missing)} CSV cases have no regional features (e.g. {missing[:5]}) – excluded")
    if extra:
        print(f"WARNING: {len(extra)} feature rows not in the clinical CSV (e.g. {extra[:5]}) – excluded")
    merged["Center"] = merged["CaseID"].str.split("__").str[0]
    return merged


def greedy_decorrelate(X, order, thr):
    """Walk features from most to least significant; keep one if |rho| <= thr with all kept."""
    rho = np.abs(spearmanr(X[order].to_numpy()).statistic) if len(order) > 2 else \
        np.abs(np.corrcoef(X[order].rank().to_numpy().T))
    rho = np.atleast_2d(rho)
    kept = []
    for i, f in enumerate(order):
        if all(rho[i, order.index(k)] <= thr for k in kept):
            kept.append(f)
    return kept


def lasso_select(X, y):
    Cs = np.logspace(-3, 1, N_C)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros((N_FOLDS, len(Cs)))
    nnz = np.zeros(len(Cs))
    for f, (tr, va) in enumerate(skf.split(X, y)):
        for j, C in enumerate(Cs):
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                   class_weight="balanced", max_iter=5000)
            m.fit(X[tr], y[tr])
            p = m.decision_function(X[va])
            scores[f, j] = roc_auc_score(y[va], p) if len(np.unique(y[va])) == 2 else np.nan
    mean, se = np.nanmean(scores, 0), np.nanstd(scores, 0, ddof=1) / np.sqrt(N_FOLDS)
    for j, C in enumerate(Cs):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=5000)
        nnz[j] = np.count_nonzero(m.fit(X, y).coef_)
    j_min = int(np.nanargmax(mean))
    if LASSO_RULE == "1se":
        ok = np.where((mean >= mean[j_min] - se[j_min]) & (nnz > 0))[0]
        j = int(ok.min()) if len(ok) else j_min     # smallest C = strongest penalty = sparsest
    else:
        j = j_min
    final = LogisticRegression(penalty="l1", solver="liblinear", C=Cs[j],
                               class_weight="balanced", max_iter=5000).fit(X, y)
    path = pd.DataFrame({"C": Cs, "cv_auc_mean": mean, "cv_auc_se": se, "n_nonzero": nnz})
    return final, Cs[j], path


# ----------------------------------------------------------------------------- main
def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    data = load_features()
    meta_cols = ["CaseID", "Center", "PatientID", "split", "label"]
    feat_cols = [c for c in data.columns if c not in meta_cols]
    X_all = data[feat_cols].apply(pd.to_numeric, errors="coerce")
    is_tr = data["split"].isin(TRAIN_VALUES).to_numpy()
    y_tr = data.loc[is_tr, "label"].astype(int).to_numpy()
    log = {"split_col": SPLIT_COL, "train_values": TRAIN_VALUES,
           "n_cases": data["split"].value_counts().to_dict(), "n_features_raw": len(feat_cols)}

    # 2. missingness / variance (train only)
    miss = X_all[is_tr].isna().mean()
    keep = miss[miss <= MAX_MISSING_FRAC].index
    std = X_all.loc[is_tr, keep].std()
    keep = std[std > 1e-8].index.tolist()
    log["n_after_missing_variance"] = len(keep)
    medians = X_all.loc[is_tr, keep].median()
    X = X_all[keep].fillna(medians)

    # 3. harmonization
    harm, harm_log = None, {}
    if HARMONIZE:
        harm = LocationScaleHarmonizer().fit(X[is_tr], data.loc[is_tr, "Center"])
        X, harm_log = harm.transform(X, data["Center"], unseen=UNSEEN_CENTER_MODE)
    log["harmonization"] = harm_log

    # 4. z-score
    mu, sd = X[is_tr].mean(), X[is_tr].std(ddof=0).replace(0, 1.0)
    Z = (X - mu) / sd

    # 5. Mann-Whitney U
    Ztr = Z[is_tr]
    pvals = {}
    for f in keep:
        a, b = Ztr.loc[y_tr == 1, f], Ztr.loc[y_tr == 0, f]
        pvals[f] = mannwhitneyu(a, b, alternative="two-sided").pvalue
    uni = pd.Series(pvals, name="p_mannwhitney").sort_values()
    uni.to_csv(out / "univariate_tests.csv", encoding="utf-8-sig")
    sig = uni[uni < P_THRESHOLD].index.tolist()
    log["n_after_utest"] = len(sig)
    if len(sig) < 2:
        raise RuntimeError(f"Only {len(sig)} features pass p<{P_THRESHOLD}; check labels/features before relaxing")

    # 6. Spearman redundancy
    dec = greedy_decorrelate(Ztr, sig, CORR_THRESHOLD)
    log["n_after_spearman"] = len(dec)

    # 7. LASSO
    model, C, path = lasso_select(Ztr[dec].to_numpy(), y_tr)
    path.to_csv(out / "lasso_path.csv", index=False)
    coef = pd.Series(model.coef_.ravel(), index=dec)
    selected = coef[coef != 0].abs().sort_values(ascending=False)
    if MAX_FEATURES:
        selected = selected.iloc[:MAX_FEATURES]
    selected = selected.index.tolist()
    if not selected:
        raise RuntimeError("LASSO kept no feature; inspect lasso_path.csv")
    log.update({"lasso_rule": LASSO_RULE, "lasso_C": float(C), "n_selected": len(selected),
                "selected": {f: {"coef": float(coef[f]), "p_utest": float(uni[f])} for f in selected}})

    # outputs
    table = pd.concat([data[["CaseID", "Center", "split", "label"]], Z[selected]], axis=1)
    table = table.rename(columns={"split": SPLIT_COL})
    table.to_csv(out / "habitat_features_selected.csv", index=False, encoding="utf-8-sig")
    # plain pandas objects only, so the file loads without importing this script
    harm_state = None if harm is None else {
        "mu_pool": harm.mu_pool_, "sd_pool": harm.sd_pool_,
        "center_mu": {c: v[0] for c, v in harm.params_.items()},
        "center_sd": {c: v[1] for c, v in harm.params_.items()}}
    joblib.dump({"kept_after_missing": keep, "medians": medians, "harmonizer": harm_state,
                 "unseen_center_mode": UNSEEN_CENTER_MODE, "z_mean": mu, "z_sd": sd,
                 "selected": selected, "lasso": model, "decorrelated": dec},
                out / "habitat_preprocessor.joblib")
    (out / "feature_selection_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "selected"}, indent=2, ensure_ascii=False))
    print("Selected features:", *selected, sep="\n  ")
    for s in data["split"].unique():
        m = data["split"].eq(s) & data["label"].isin([0, 1])
        if m.any() and data.loc[m, "label"].nunique() == 2:
            auc = roc_auc_score(data.loc[m, "label"].astype(int), model.decision_function(Z.loc[m, dec].to_numpy()))
            print(f"LASSO habitat-only AUC [{s}]: {auc:.3f}  (sanity check, not the final model)")


if __name__ == "__main__":
    main()
