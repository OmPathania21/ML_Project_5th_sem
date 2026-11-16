# --- Setup: works in this repo (run from code/notebooks) and in Google Colab ---
import subprocess, sys, importlib
for pkg in ["xgboost", "lightgbm", "catboost", "shap"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import json, time, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

BASE = Path("..") if (Path("..") / "data" / "raw").exists() else Path(".")
RAW, PROCESSED, FIGURES, MODELS = BASE / "data/raw", BASE / "data/processed", BASE / "outputs/figures", BASE / "outputs/models"
for d in (RAW / "diabetes", PROCESSED, FIGURES, MODELS):
    d.mkdir(parents=True, exist_ok=True)

CSV = RAW / "diabetes" / "diabetes_prediction_dataset.csv"
if not CSV.exists():   # Colab: download with kagglehub (public dataset)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=False)
    import kagglehub
    src = Path(kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")) / "diabetes_prediction_dataset.csv"
    shutil.copy(src, CSV)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, GREEN = SERIES[0], SERIES[1], SERIES[2]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9e2323"])
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "text.color": INK, "legend.frameon": False, "lines.linewidth": 2,
    "axes.prop_cycle": mpl.cycler(color=SERIES)})

def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight"); plt.show()

def save_json(obj, name):
    (MODELS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))

SEED = 42
df_raw = pd.read_csv(CSV)
print(df_raw.shape)
df_raw.head()

audit = pd.DataFrame({"dtype": df_raw.dtypes.astype(str), "missing": df_raw.isna().sum(), "unique": df_raw.nunique(),
                      "min": df_raw.min(numeric_only=True), "max": df_raw.max(numeric_only=True)})
print("Exact duplicate rows:", df_raw.duplicated().sum())
print("Gender:", df_raw.gender.value_counts().to_dict())
print("Smoking history:", df_raw.smoking_history.value_counts().to_dict())
print("Diabetes prevalence:", round(df_raw.diabetes.mean(), 4))
audit

df = df_raw.drop_duplicates()
df = df[df.gender != "Other"].reset_index(drop=True)
print(f"Rows: {len(df_raw):,} → {len(df):,} | diabetes prevalence {df.diabetes.mean():.3%}")
df.to_csv(PROCESSED / "diabetes_clean.csv", index=False)
NUM = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BIN = ["hypertension", "heart_disease"]
CAT = ["gender", "smoking_history"]
df[NUM].describe().T.round(2)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
vc = df.diabetes.value_counts().sort_index()
ax.bar(["No diabetes", "Diabetes"], vc.values, color=[MUTED, ORANGE], width=0.55)
for i, v in enumerate(vc.values):
    ax.text(i, v + 900, f"{v:,}\n({v / vc.sum():.1%})", ha="center", color=INK_2, fontsize=9.5)
ax.set(title="Class balance: 1 in 11 patients is diabetic", ylabel="Patients", ylim=(0, vc.max() * 1.18))
ax.grid(axis="x", visible=False)
save(fig, "db_class_balance")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col, unit in zip(axes.ravel(), NUM, ["years", "kg/m²", "%", "mg/dL"]):
    for cls, c, lab in [(0, BLUE, "No diabetes"), (1, ORANGE, "Diabetes")]:
        ax.hist(df.loc[df.diabetes == cls, col], bins=40, density=True, alpha=0.55, color=c, label=lab)
    ax.set(title=col.replace("_", " "), xlabel=unit, ylabel="Density")
axes[0, 0].legend()
fig.suptitle("Distribution of clinical features by diabetes status", x=0.01, ha="left", fontweight="semibold")
save(fig, "db_feature_distributions")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
age_bins = pd.cut(df.age, [0, 20, 30, 40, 50, 60, 70, 81], right=False,
                  labels=["<20", "20–29", "30–39", "40–49", "50–59", "60–69", "70–80"])
r = df.groupby(age_bins, observed=True).diabetes.mean() * 100
axes[0].bar(r.index.astype(str), r.values, color=BLUE, width=0.65); axes[0].set(title="Diabetes rate by age", ylabel="%")
bmi_bins = pd.cut(df.bmi, [0, 18.5, 25, 30, 35, 100], labels=["Under", "Normal", "Over", "Obese I", "Obese II+"])
r = df.groupby(bmi_bins, observed=True).diabetes.mean() * 100
axes[1].bar(r.index.astype(str), r.values, color=BLUE, width=0.65); axes[1].set(title="Diabetes rate by BMI category")
comb = pd.Series({"None": df[(df.hypertension == 0) & (df.heart_disease == 0)].diabetes.mean(),
                  "Hypertension": df[df.hypertension == 1].diabetes.mean(),
                  "Heart disease": df[df.heart_disease == 1].diabetes.mean()}) * 100
axes[2].bar(comb.index, comb.values, color=[MUTED, ORANGE, ORANGE], width=0.6); axes[2].set(title="Diabetes rate by comorbidity")
for ax in axes:
    ax.axhline(df.diabetes.mean() * 100, color=INK_2, ls="--", lw=1); ax.grid(axis="x", visible=False)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.4, f"{p.get_height():.1f}", ha="center", fontsize=8.5, color=INK_2)
fig.suptitle("Risk factors (dashed line = overall prevalence 8.8 %)", x=0.01, ha="left", fontweight="semibold")
save(fig, "db_risk_factors")

fig, ax = plt.subplots(figsize=(6.6, 4.6))
smp = df.sample(8000, random_state=SEED)
jit = np.random.default_rng(0).normal(0, 0.04, len(smp))
for cls, c, lab in [(0, BLUE, "No diabetes"), (1, ORANGE, "Diabetes")]:
    m = smp.diabetes == cls
    ax.scatter(smp.HbA1c_level[m] + jit[m], smp.blood_glucose_level[m], s=7, alpha=0.35 if cls == 0 else 0.6, color=c,
               linewidths=0, label=lab)
ax.axvline(6.5, color=INK_2, ls="--", lw=1); ax.axhline(200, color=INK_2, ls="--", lw=1)
ax.text(6.55, 285, "HbA1c ≥ 6.5 %", fontsize=9, color=INK_2); ax.text(3.6, 205, "glucose ≥ 200", fontsize=9, color=INK_2)
ax.set(title="HbA1c × blood glucose (8,000-patient sample)", xlabel="HbA1c (%)", ylabel="Blood glucose (mg/dL)")
ax.legend(markerscale=3, loc="upper left")
save(fig, "db_hba1c_glucose")

high = (df.HbA1c_level > 6.6) | (df.blood_glucose_level > 200)
print(f"Patients with HbA1c > 6.6 or glucose > 200: {high.mean():.1%} of all; diabetes rate among them {df.diabetes[high].mean():.1%}")
print(f"Diabetics captured by that rule: {high[df.diabetes == 1].mean():.1%}  (the rest are the hard cases)")

enc = pd.get_dummies(df, columns=CAT, drop_first=False, dtype=float)
corr = df[NUM + BIN + ["diabetes"]].corr()
fig, ax = plt.subplots(figsize=(6.6, 5.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, square=True, linewidths=1.5, linecolor=SURFACE,
            cbar_kws={"shrink": 0.75}, ax=ax, annot_kws={"fontsize": 9})
ax.set(title="Correlation of numeric features with diabetes"); ax.grid(False)
save(fig, "db_correlation")
corr["diabetes"].drop("diabetes").sort_values(ascending=False).round(3)

sm = df.groupby("smoking_history").diabetes.agg(["mean", "size"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.barh(sm.index, sm["mean"] * 100, color=BLUE, height=0.6)
for y, (m, n) in enumerate(zip(sm["mean"], sm["size"])):
    ax.text(m * 100 + 0.2, y, f"{m:.1%}  (n={n:,})", va="center", fontsize=9, color=INK_2)
ax.axvline(df.diabetes.mean() * 100, color=INK_2, ls="--", lw=1)
ax.set(title="Diabetes rate by smoking history", xlabel="%", xlim=(0, sm["mean"].max() * 140))
ax.grid(axis="y", visible=False)
save(fig, "db_smoking")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier, StackingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

X = df.drop(columns="diabetes"); y = df.diabetes.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train {X_train.shape}, positives {y_train.mean():.3%} | Test {X_test.shape}, positives {y_test.mean():.3%}")
pre = ColumnTransformer([("num", StandardScaler(), NUM + BIN),
                         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pos_w = (y_train == 0).sum() / (y_train == 1).sum()

def P(model):
    return Pipeline([("pre", pre), ("model", model)])

def base_learners():
    return [("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
            ("xgb", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", n_jobs=-1, random_state=SEED)),
            ("lgbm", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, subsample_freq=1,
                                    colsample_bytree=0.9, verbose=-1, n_jobs=-1, random_state=SEED)),
            ("cat", CatBoostClassifier(iterations=500, learning_rate=0.08, depth=6, verbose=0, random_seed=SEED,
                                       thread_count=-1))]

models = {
    "Logistic Regression": P(LogisticRegression(max_iter=2000)),
    "Decision Tree": P(DecisionTreeClassifier(random_state=SEED)),
    "Bagging": P(BaggingClassifier(DecisionTreeClassifier(random_state=SEED), n_estimators=200, n_jobs=-1, random_state=SEED)),
    "Random Forest": P(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    "Extra Trees": P(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    "AdaBoost": P(AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED)),
    "Gradient Boosting": P(GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=SEED)),
    "XGBoost": P(dict(base_learners())["xgb"]),
    "LightGBM": P(dict(base_learners())["lgbm"]),
    "CatBoost": P(dict(base_learners())["cat"]),
    "Soft Voting": P(VotingClassifier(base_learners(), voting="soft", n_jobs=1)),
    "Stacking": P(StackingClassifier(base_learners(), final_estimator=LogisticRegression(max_iter=2000), cv=5,
                                     stack_method="predict_proba", n_jobs=1)),
}
FAMILY = {"Logistic Regression": "Baseline", "Decision Tree": "Baseline", "Bagging": "Bagging", "Random Forest": "Bagging",
          "Extra Trees": "Bagging", "AdaBoost": "Boosting", "Gradient Boosting": "Boosting", "XGBoost": "Boosting",
          "LightGBM": "Boosting", "CatBoost": "Boosting", "Soft Voting": "Meta-ensemble", "Stacking": "Meta-ensemble"}

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1",
           "roc_auc": "roc_auc", "pr_auc": "average_precision"}
cv_rows, cv_f1_folds = [], {}
for name, model in models.items():
    t0 = time.time()
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
    cv_f1_folds[name] = res["test_f1"]
    cv_rows.append({"Model": name, "Family": FAMILY[name],
                    **{f"CV {k}": res[f"test_{k}"].mean() for k in SCORING},
                    "CV F1 std": res["test_f1"].std(), "CV ROC-AUC std": res["test_roc_auc"].std(),
                    "CV time (s)": time.time() - t0})
    print(f"{name:20s} F1 {res['test_f1'].mean():.4f} ± {res['test_f1'].std():.4f} | ROC-AUC {res['test_roc_auc'].mean():.4f}")
cvres = pd.DataFrame(cv_rows).sort_values("CV f1", ascending=False).reset_index(drop=True)
cvres.to_csv(MODELS / "diabetes_cv_results.csv", index=False)
cvres.round(4)

test_rows, fitted, proba = [], {}, {}
for name, model in models.items():
    t0 = time.time(); model.fit(X_train, y_train); fit_s = time.time() - t0
    p = model.predict_proba(X_test)[:, 1]; pred = (p >= 0.5).astype(int)
    fitted[name], proba[name] = model, p
    test_rows.append({"Model": name, "Family": FAMILY[name], "Accuracy": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                      "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, p),
                      "PR-AUC": average_precision_score(y_test, p), "Brier": brier_score_loss(y_test, p), "Fit time (s)": fit_s})
test = pd.DataFrame(test_rows).sort_values("F1", ascending=False).reset_index(drop=True)
test.to_csv(MODELS / "diabetes_test_results.csv", index=False)
test.round(4)

param_dist = {
    "model__n_estimators": [200, 400, 600, 800, 1000], "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "model__num_leaves": [15, 31, 63, 127], "model__max_depth": [-1, 4, 6, 8, 10],
    "model__min_child_samples": [10, 20, 40, 80], "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0], "model__reg_alpha": [0, 0.01, 0.1, 1.0],
    "model__reg_lambda": [0, 0.1, 1.0, 5.0],
}
search = RandomizedSearchCV(P(LGBMClassifier(subsample_freq=1, verbose=-1, n_jobs=-1, random_state=SEED)), param_dist,
                            n_iter=40, scoring="average_precision", cv=cv, random_state=SEED, n_jobs=1, refit=True)
t0 = time.time(); search.fit(X_train, y_train); tune_s = time.time() - t0
best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
print(f"Best CV PR-AUC {search.best_score_:.4f} in {tune_s:.0f}s with", best_params)
tuned = search.best_estimator_
fitted["LightGBM (tuned)"] = tuned
proba["LightGBM (tuned)"] = tuned.predict_proba(X_test)[:, 1]

def best_threshold(model):
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    pr, rc, th = precision_recall_curve(y_train, oof)
    f1s = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[np.argmax(f1s)]), oof

final_name = cvres.iloc[0]["Model"]
print("Final model (highest CV F1):", final_name)
thr_rows, thresholds = [], {}
candidates = ["Logistic Regression", final_name, "XGBoost", "Random Forest", "CatBoost", "Stacking", "LightGBM (tuned)"]
for name in dict.fromkeys(candidates):
    mdl = tuned if name == "LightGBM (tuned)" else models[name]
    t, oof = best_threshold(mdl)
    thresholds[name] = t
    p = proba[name]; pred = (p >= t).astype(int)
    thr_rows.append({"Model": name, "Threshold": t, "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, p), "PR-AUC": average_precision_score(y_test, p)})
thr = pd.DataFrame(thr_rows).sort_values("F1", ascending=False).reset_index(drop=True)
thr.to_csv(MODELS / "diabetes_threshold_results.csv", index=False)
thr.round(4)

fam_color = {"Baseline": MUTED, "Bagging": BLUE, "Boosting": ORANGE, "Meta-ensemble": GREEN}
plot = cvres.sort_values("CV f1")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, col, err, title, lo in [(axes[0], "CV f1", "CV F1 std", "5-fold CV F1-score", 0.6),
                                (axes[1], "CV roc_auc", "CV ROC-AUC std", "5-fold CV ROC-AUC", 0.80)]:
    ax.barh(plot.Model, plot[col], xerr=plot[err], color=[fam_color[f] for f in plot.Family], height=0.65,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    for yv, v in enumerate(plot[col]):
        ax.text(v + 0.004, yv, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.set(title=title, xlim=(lo, 1.0)); ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
axes[1].legend(handles, fam_color.keys(), loc="lower right", fontsize=9)
save(fig, "db_cv_comparison")

show = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost", "LightGBM (tuned)", "Stacking"]
cols = [MUTED, "#c3c2b7", BLUE, SERIES[3], ORANGE, GREEN]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for n, c in zip(show, cols):
    fpr, tpr, _ = roc_curve(y_test, proba[n])
    axes[0].plot(fpr, tpr, color=c, lw=1.8, label=f"{n} ({roc_auc_score(y_test, proba[n]):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, proba[n])
    axes[1].plot(rc, pr, color=c, lw=1.8, label=f"{n} ({average_precision_score(y_test, proba[n]):.3f})")
axes[0].plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
axes[0].set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate")
axes[1].axhline(y_test.mean(), color=AXIS, ls="--", lw=1)
axes[1].set(title="Precision–recall curves (test set)", xlabel="Recall", ylabel="Precision")
axes[0].legend(fontsize=8.5, loc="lower right"); axes[1].legend(fontsize=8.5, loc="lower left")
save(fig, "db_roc_pr")

best_name = final_name
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (title, p, t) in zip(axes, [("Logistic Regression (baseline), 0.50", proba["Logistic Regression"], 0.5),
                                    (f"{best_name}, 0.50", proba[best_name], 0.5),
                                    (f"{best_name}, tuned threshold {thresholds[best_name]:.2f}", proba[best_name], thresholds[best_name])]):
    cm = confusion_matrix(y_test, (p >= t).astype(int))
    sns.heatmap(cm, annot=False, cmap=SEQ_BLUE, cbar=False, linewidths=2, linecolor=SURFACE, ax=ax)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.5, f"{v:,}", ha="center", va="center", fontsize=12, color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0.5, 1.5], ["Pred: no", "Pred: yes"]); ax.set_yticks([0.5, 1.5], ["True: no", "True: yes"], rotation=0)
    ax.set(title=title)
fig.suptitle(f"Confusion matrices on the held-out test set (n = {len(y_test):,})", x=0.01, ha="left", fontweight="semibold")
save(fig, "db_confusion")

p_best, oof_best = proba[best_name], None
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ts = np.linspace(0.05, 0.95, 91)
for n, c in [("Logistic Regression", MUTED), (best_name, ORANGE)]:
    axes[0].plot(ts, [f1_score(y_test, (proba[n] >= t).astype(int)) for t in ts], color=c, label=n)
axes[0].axvline(0.5, color=AXIS, ls=":", lw=1)
axes[0].axvline(thresholds[best_name], color=ORANGE, ls="--", lw=1)
axes[0].set(title="Test F1 vs decision threshold", xlabel="Threshold", ylabel="F1"); axes[0].legend(fontsize=9)
for n, c in [("Logistic Regression", MUTED), ("Random Forest", BLUE), (best_name, ORANGE)]:
    fy, mx = calibration_curve(y_test, proba[n], n_bins=12, strategy="quantile")
    axes[1].plot(mx, fy, marker="o", ms=4, color=c, label=f"{n} (Brier {brier_score_loss(y_test, proba[n]):.4f})")
axes[1].plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
axes[1].set(title="Calibration (reliability) curve", xlabel="Mean predicted probability", ylabel="Observed frequency")
axes[1].legend(fontsize=8.5)
save(fig, "db_threshold_calibration")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for _, r in test.iterrows():
    ax.scatter(r["Fit time (s)"], r["F1"], s=70, color=fam_color[r.Family], zorder=3)
    ax.annotate(r.Model, (r["Fit time (s)"], r["F1"]), xytext=(5, 3), textcoords="offset points", fontsize=8.5, color=INK_2)
ax.set(xscale="log", title="Accuracy–cost trade-off (test F1 vs training time)", xlabel="Training time (s, log scale)", ylabel="Test F1 @ 0.5")
save(fig, "db_time_vs_f1")

from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
sig_rows = []
for n in [m for m in models if m != "Logistic Regression"]:
    t, pv = ttest_rel(cv_f1_folds[n], cv_f1_folds["Logistic Regression"])
    sig_rows.append({"Model": n, "Mean ΔF1 vs LR": (cv_f1_folds[n] - cv_f1_folds["Logistic Regression"]).mean(), "p-value": pv})
sig = pd.DataFrame(sig_rows).sort_values("Mean ΔF1 vs LR", ascending=False)
a = (proba[best_name] >= thresholds[best_name]).astype(int) == y_test
b = (proba["Logistic Regression"] >= 0.5).astype(int) == y_test
tbl = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
mc = mcnemar(tbl, exact=False, correction=True)
print(f"McNemar ({best_name} vs Logistic Regression): only best correct {tbl[0][1]}, only LR correct {tbl[1][0]}, "
      f"chi2 = {mc.statistic:.1f}, p = {mc.pvalue:.2e}")
sig.round(5)

import shap
from sklearn.inspection import permutation_importance
shap_pipe = fitted[best_name] if FAMILY[best_name] in ("Boosting", "Bagging") else tuned   # tree model for TreeExplainer
shap_name = best_name if shap_pipe is fitted.get(best_name) else "LightGBM (tuned)"
lgb_model = shap_pipe.named_steps["model"]
Xt_test = pd.DataFrame(shap_pipe.named_steps["pre"].transform(X_test), columns=shap_pipe.named_steps["pre"].get_feature_names_out())
Xt_test.columns = [c.split("__")[1] for c in Xt_test.columns]
shap_sample = Xt_test.sample(3000, random_state=SEED)
explainer = shap.TreeExplainer(lgb_model)
sv = explainer.shap_values(shap_sample)
sv = sv[1] if isinstance(sv, list) else sv
sv = sv[:, :, 1] if sv.ndim == 3 else sv
plt.figure(figsize=(7.5, 5))
shap.summary_plot(sv, shap_sample, show=False, max_display=12, plot_size=None)
plt.title(f"SHAP summary ({shap_name}): impact on diabetes risk", loc="left", fontsize=12, fontweight="semibold")
save(plt.gcf(), "db_shap_summary")
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=shap_sample.columns).sort_values(ascending=False)
mean_abs.head(10).round(4)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, f in zip(axes, ["HbA1c_level", "blood_glucose_level"]):
    raw = X_test.iloc[shap_sample.index][f].values           # shap_sample keeps positional row numbers
    ax.scatter(raw, sv[:, list(shap_sample.columns).index(f)], s=6, alpha=0.4, color=BLUE, linewidths=0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set(title=f"SHAP dependence: {f.replace('_', ' ')}", xlabel=f.replace("_", " "), ylabel="SHAP value (log-odds)")
save(fig, "db_shap_dependence")

perm = permutation_importance(shap_pipe, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=1)
pi = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.barh(pi.index, pi.values, color=BLUE, height=0.6)
ax.set(title=f"Permutation importance ({shap_name})", xlabel="Drop in test PR-AUC"); ax.grid(axis="y", visible=False)
save(fig, "db_permutation_importance")
pi.sort_values(ascending=False).round(4)

gb = fitted["Gradient Boosting"]; Xt_tr = gb.named_steps["pre"].transform(X_train); Xt_te = gb.named_steps["pre"].transform(X_test)
gbm = gb.named_steps["model"]
stage_tr = [roc_auc_score(y_train, p[:, 1]) for p in gbm.staged_predict_proba(Xt_tr)]
stage_te = [roc_auc_score(y_test, p[:, 1]) for p in gbm.staged_predict_proba(Xt_te)]
rf = fitted["Random Forest"].named_steps["model"]
Xr_te = fitted["Random Forest"].named_steps["pre"].transform(X_test)
tree_probs = np.array([t.predict_proba(Xr_te)[:, 1] for t in rf.estimators_])
ns = [1, 2, 5, 10, 20, 50, 100, 200, 300]
rf_auc = [roc_auc_score(y_test, tree_probs[:k].mean(axis=0)) for k in ns]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, len(stage_tr) + 1), stage_tr, color=MUTED, label="train")
axes[0].plot(range(1, len(stage_te) + 1), stage_te, color=ORANGE, label="test")
axes[0].set(title="Boosting: ROC-AUC as trees are added", xlabel="Number of boosting stages", ylabel="ROC-AUC"); axes[0].legend()
axes[1].plot(ns, rf_auc, color=BLUE, marker="o", ms=4)
axes[1].set(xscale="log", title="Bagging: averaging more trees (Random Forest)", xlabel="Number of trees (log)", ylabel="Test ROC-AUC")
save(fig, "db_ensemble_size")
print(f"Single tree AUC {rf_auc[0]:.3f} → 300 trees {rf_auc[-1]:.3f}; GB stage 1 {stage_te[0]:.3f} → {stage_te[-1]:.3f}")

lr_t = test.set_index("Model").loc["Logistic Regression"]; dt_t = test.set_index("Model").loc["Decision Tree"]
best_row = thr.set_index("Model").loc[best_name]
summary = {
    "rows_raw": len(df_raw), "rows_clean": len(df), "duplicates_removed": int(df_raw.duplicated().sum()),
    "prevalence": float(df.diabetes.mean()), "train": len(X_train), "test": len(X_test),
    "cv": cvres.to_dict("records"), "test_default": test.to_dict("records"), "threshold": thr.to_dict("records"),
    "best_model": best_name, "best_threshold": thresholds[best_name], "best": best_row.to_dict(),
    "best_default": test.set_index("Model").loc[best_name].to_dict(), "shap_model": shap_name,
    "tuned_params": best_params, "tuning_cv_pr_auc": float(search.best_score_), "tuning_seconds": tune_s,
    "lr_test": lr_t.to_dict(), "dt_test": dt_t.to_dict(), "significance": sig.to_dict("records"),
    "mcnemar": {"only_best_correct": int(tbl[0][1]), "only_lr_correct": int(tbl[1][0]), "chi2": float(mc.statistic), "p": float(mc.pvalue)},
    "shap_mean_abs": mean_abs.head(10).to_dict(), "perm_importance": pi.sort_values(ascending=False).to_dict(),
    "rf_auc_by_trees": dict(zip(map(str, ns), rf_auc)), "gb_stage_auc_first_last": [stage_te[0], stage_te[-1]],
    "rule_share": float(high.mean()), "rule_diabetes_rate": float(df.diabetes[high].mean()),
    "rule_capture": float(high[df.diabetes == 1].mean()),
}
save_json(summary, "diabetes_summary")
pd.Series({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
