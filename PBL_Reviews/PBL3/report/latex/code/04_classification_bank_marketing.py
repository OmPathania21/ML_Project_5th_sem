import sys; sys.path.insert(0, "..")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.viz import *

set_style()
SEED = 42
df_raw = pd.read_csv(RAW / "bank_marketing" / "bank-additional" / "bank-additional-full.csv", sep=";")
print(df_raw.shape)
df_raw.head()

print("NaN values:", df_raw.isna().sum().sum(), "| Duplicate rows:", df_raw.duplicated().sum())
cat_cols_all = df_raw.select_dtypes("object").columns.drop("y")
unknown = (df_raw[cat_cols_all] == "unknown").sum()
print("'unknown' placeholders:\n", unknown[unknown > 0])
print("\nTarget balance:\n", df_raw["y"].value_counts(normalize=True).round(4))

df = df_raw.drop_duplicates().reset_index(drop=True)
df["target"] = (df["y"] == "yes").astype(int)
base_rate = df["target"].mean()

fig, ax = plt.subplots(figsize=(5, 3.4))
vc = df["y"].value_counts()
ax.bar(["No", "Yes"], vc.values, color=[MUTED, BLUE], width=0.55)
for i, v in enumerate(vc.values):
    ax.text(i, v + 500, f"{v:,}\n({v / vc.sum():.1%})", ha="center", color=INK_2, fontsize=9)
ax.set(title="Only 11% of clients subscribe", ylabel="Clients", ylim=(0, vc.max() * 1.2))
ax.grid(axis="x", visible=False)
save(fig, "bk_target_balance")

def rate_plot(ax, col, order=None, title=None):
    r = df.groupby(col)["target"].agg(["mean", "size"])
    r = r.loc[order] if order else r.sort_values("mean")
    ax.barh(r.index.astype(str), r["mean"] * 100, color=BLUE, height=0.7)
    ax.axvline(base_rate * 100, color=INK_2, lw=1, ls="--")
    for yv, (m, n) in enumerate(zip(r["mean"], r["size"])):
        ax.text(m * 100 + 0.8, yv, f"{m:.0%}", va="center", fontsize=8.5, color=INK_2)
    ax.set(title=title or col, xlabel="Subscription rate (%)")
    ax.grid(axis="y", visible=False)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
rate_plot(axes[0], "poutcome", title="Previous campaign outcome")
rate_plot(axes[1], "month", order=["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
          title="Month of last contact")
rate_plot(axes[2], "job", title="Job")
fig.suptitle("Subscription rate by segment (dashed line = overall rate, 11%)", x=0.01, ha="left",
             fontweight="semibold")
save(fig, "bk_rate_by_category")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
bins = [16, 25, 30, 40, 50, 60, 100]
a = df.groupby(pd.cut(df["age"], bins), observed=True)["target"].mean() * 100
axes[0].bar([f"{int(i.left) + 1}–{int(i.right)}" for i in a.index], a.values, color=BLUE, width=0.65)
axes[0].set(title="Age group", ylabel="Subscription rate (%)")
c = df.assign(calls=df["campaign"].clip(upper=8)).groupby("calls")["target"].mean() * 100
axes[1].bar([str(i) if i < 8 else "8+" for i in c.index], c.values, color=BLUE, width=0.65)
axes[1].set(title="Calls in this campaign", xlabel="Number of contacts")
e = df.groupby("nr.employed")["target"].mean() * 100
axes[2].bar([f"{v:.0f}" for v in e.index], e.values, color=BLUE, width=0.65)
axes[2].set(title="Economy: no. of employees (thousands)", xlabel="nr.employed (quarterly indicator)")
axes[2].tick_params(axis="x", rotation=60, labelsize=8)
for ax in axes:
    ax.axhline(base_rate * 100, color=INK_2, lw=1, ls="--")
    ax.grid(axis="x", visible=False)
save(fig, "bk_rate_by_numeric")

num_all = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed", "target"]
corr = df[num_all].corr()
fig, ax = plt.subplots(figsize=(7.4, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, square=True, linewidths=1.5,
            linecolor=SURFACE, cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"fontsize": 8})
ax.set(title="Correlation of numeric features")
ax.grid(False)
save(fig, "bk_correlation")
corr["target"].drop("target").sort_values()

df["prev_contacted"] = (df["pdays"] != 999).astype(int)
X = df.drop(columns=["y", "target", "duration", "pdays"])
y = df["target"]
cat_cols = X.select_dtypes("object").columns.tolist()
num_cols = X.select_dtypes("number").columns.tolist()
print("Categorical:", cat_cols, "\nNumeric:", num_cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(X_train.shape, X_test.shape, f"positive rate train {y_train.mean():.3f} / test {y_test.mean():.3f}")
pd.concat([X, y], axis=1).to_csv(PROCESSED / "bank_marketing_model_ready.csv", index=False)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
spw = (y_train == 0).sum() / (y_train == 1).sum()

def pipe(model, smote=False):
    steps = [("pre", pre)] + ([("smote", SMOTE(random_state=SEED))] if smote else []) + [("model", model)]
    return ImbPipeline(steps)

models = {
    "Baseline (always 'no')": pipe(DummyClassifier(strategy="most_frequent")),
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000)),
    "Logistic Reg. (balanced)": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Logistic Reg. + SMOTE": pipe(LogisticRegression(max_iter=2000), smote=True),
    "Naive Bayes": pipe(GaussianNB()),
    "KNN (k=15)": pipe(KNeighborsClassifier(n_neighbors=15)),
    "Decision Tree (balanced)": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                            class_weight="balanced", random_state=SEED)),
    "Random Forest (balanced)": pipe(RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                            class_weight="balanced_subsample",
                                                            n_jobs=-1, random_state=SEED)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                         random_state=SEED)),
    "XGBoost (weighted)": pipe(XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
                                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                             random_state=SEED, n_jobs=-1)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, fitted, probas = [], {}, {}
for name, model in models.items():
    cvres = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1, scoring=["roc_auc", "f1"])
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_s = time.time() - t0
    p = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    fitted[name], probas[name] = model, p
    rows.append({
        "Model": name,
        "CV ROC-AUC": cvres["test_roc_auc"].mean(), "CV ROC-AUC std": cvres["test_roc_auc"].std(),
        "CV F1": cvres["test_f1"].mean(),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, p),
        "PR-AUC": average_precision_score(y_test, p),
        "Fit time (s)": fit_s,
    })
results = pd.DataFrame(rows)
results.round(4)

ranked = results[results["Model"] != "Baseline (always 'no')"].sort_values("CV ROC-AUC", ascending=False)
best_name = ranked.iloc[0]["Model"]
print("Best by CV ROC-AUC:", best_name)
oof = cross_val_predict(models[best_name], X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_thr = float(thr[np.argmax(f1s)])
print(f"F1-optimal threshold (OOF): {best_thr:.3f}")

p = probas[best_name]
pred_t = (p >= best_thr).astype(int)
tuned_row = {"Model": f"{best_name} @ thr {best_thr:.2f}",
             "CV ROC-AUC": ranked.iloc[0]["CV ROC-AUC"], "CV ROC-AUC std": ranked.iloc[0]["CV ROC-AUC std"],
             "CV F1": f1s.max(),
             "Accuracy": accuracy_score(y_test, pred_t), "Precision": precision_score(y_test, pred_t),
             "Recall": recall_score(y_test, pred_t), "F1": f1_score(y_test, pred_t),
             "ROC-AUC": roc_auc_score(y_test, p), "PR-AUC": average_precision_score(y_test, p),
             "Fit time (s)": np.nan}
results = pd.concat([results, pd.DataFrame([tuned_row])]).sort_values("F1", ascending=False).reset_index(drop=True)
results.to_csv(MODELS / "classification_results.csv", index=False)
results.round(4)

plot = results[~results["Model"].str.startswith("Baseline")].sort_values("F1")
tuned_name = tuned_row["Model"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
for ax, m, lim in [(axes[0], "F1", (0, 0.6)), (axes[1], "Recall", (0, 1.0)), (axes[2], "ROC-AUC", (0.6, 0.83))]:
    colors = [SERIES[1] if n == tuned_name else BLUE for n in plot["Model"]]
    ax.barh(plot["Model"], plot[m], color=colors, height=0.7)
    for yv, v in enumerate(plot[m]):
        ax.text(v + (lim[1] - lim[0]) * 0.01, yv, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.set(title=m, xlim=(lim[0], lim[1] * 1.04))
    ax.grid(axis="y", visible=False)
fig.suptitle("Test-set comparison (higher is better)", x=0.01, ha="left", fontweight="semibold")
save(fig, "bk_model_comparison")

show = [best_name, "Logistic Reg. (balanced)", "Decision Tree (balanced)"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for i, n in enumerate(show):
    fpr, tpr, _ = roc_curve(y_test, probas[n])
    axes[0].plot(fpr, tpr, color=SERIES[i], label=f"{n} (AUC {roc_auc_score(y_test, probas[n]):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, probas[n])
    axes[1].plot(rc, pr, color=SERIES[i], label=f"{n} (AP {average_precision_score(y_test, probas[n]):.3f})")
axes[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random (AUC 0.5)")
axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
axes[1].axhline(y_test.mean(), color=MUTED, lw=1, ls="--", label=f"Random (AP {y_test.mean():.3f})")
axes[1].set(title="Precision–Recall curve", xlabel="Recall", ylabel="Precision")
for ax in axes:
    ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=8.5)
save(fig, "bk_roc_pr")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (title, pr) in zip(axes, [("Threshold 0.50", fitted[best_name].predict(X_test)),
                                   (f"Tuned threshold {best_thr:.2f}", pred_t)]):
    cm = confusion_matrix(y_test, pr)
    sns.heatmap(cm, annot=False, cmap=SEQ_BLUE, cbar=False, linewidths=2, linecolor=SURFACE, ax=ax)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.5, f"{v:,}\n{v / cm[i].sum():.0%} of row", ha="center", va="center",
                color="white" if v > cm.max() * 0.5 else INK, fontsize=10)
    ax.set_xticks([0.5, 1.5], ["Pred: no", "Pred: yes"])
    ax.set_yticks([0.5, 1.5], ["Actual: no", "Actual: yes"], rotation=0)
    ax.set(title=title)
fig.suptitle(f"{best_name}: confusion matrices (test set, n = {len(y_test):,})", x=0.01, ha="left",
             fontweight="semibold")
save(fig, "bk_confusion")

perm = permutation_importance(fitted[best_name], X_test, y_test, n_repeats=10, random_state=SEED,
                              scoring="roc_auc", n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(imp.index, imp.values, color=BLUE, height=0.65)
ax.set(title=f"Permutation importance ({best_name})", xlabel="Drop in test ROC-AUC when shuffled")
ax.grid(axis="y", visible=False)
save(fig, "bk_feature_importance")
imp.sort_values(ascending=False).round(4)

# Business view: calling only the top-scored 20% of clients — what share of all subscribers do we reach?
order = np.argsort(-p)
cum = np.cumsum(y_test.values[order]) / y_test.sum()
k = int(0.2 * len(p))
lift20 = y_test.values[order][:k].mean() / y_test.mean()
fig, ax = plt.subplots(figsize=(6.4, 4.2))
xs = np.arange(1, len(p) + 1) / len(p)
ax.plot(xs * 100, cum * 100, color=BLUE, label=best_name)
ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--", label="Random calling")
ax.scatter([20], [cum[k - 1] * 100], s=40, color=SERIES[1], zorder=3)
ax.annotate(f"Top 20% of calls → {cum[k - 1]:.0%} of subscribers", (20, cum[k - 1] * 100),
            xytext=(28, cum[k - 1] * 100 - 14), color=INK_2, fontsize=9.5)
ax.set(title="Cumulative gain: who to call first", xlabel="% of clients called (highest score first)",
       ylabel="% of all subscribers reached")
ax.legend(loc="lower right")
save(fig, "bk_gain_curve")
print(f"Top 20% captures {cum[k - 1]:.1%} of subscribers; lift = {lift20:.2f}x")

b = results.set_index("Model")
summary = {
    "rows": len(df), "features_used": X.shape[1], "train": len(X_train), "test": len(X_test),
    "positive_rate": base_rate * 100, "best_model": best_name, "threshold": best_thr,
    "best_default": b.loc[best_name].to_dict(), "best_tuned": b.loc[tuned_name].to_dict(),
    "baseline": b.loc["Baseline (always 'no')"].to_dict(),
    "logreg_plain": b.loc["Logistic Regression"].to_dict(),
    "top20_capture_pct": cum[k - 1] * 100, "lift_top20": lift20,
    "importance_top": imp.sort_values(ascending=False).head(6).to_dict(),
    "corr_duration_target": corr.loc["duration", "target"],
    "poutcome_success_rate": df.loc[df["poutcome"] == "success", "target"].mean() * 100,
    "month_rates": (df.groupby("month")["target"].mean() * 100).round(1).to_dict(),
    "month_counts": df["month"].value_counts().to_dict(),
    "job_rates": (df.groupby("job")["target"].mean() * 100).round(1).to_dict(),
    "cm_tuned": confusion_matrix(y_test, pred_t).tolist(),
    "cm_default": confusion_matrix(y_test, fitted[best_name].predict(X_test)).tolist(),
}
save_json(summary, "classification_summary")
pd.Series(summary)
