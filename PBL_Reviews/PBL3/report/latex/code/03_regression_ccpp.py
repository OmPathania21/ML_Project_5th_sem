import sys; sys.path.insert(0, "..")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor
from src.viz import *

set_style()
SEED = 42
df_raw = pd.read_excel(RAW / "ccpp" / "CCPP" / "Folds5x2_pp.xlsx", sheet_name="Sheet1")
print(df_raw.shape)
df_raw.head()

print("Missing:", df_raw.isna().sum().sum(), "| Duplicates:", df_raw.duplicated().sum())
df = df_raw.drop_duplicates().reset_index(drop=True)
df.to_csv(PROCESSED / "ccpp_clean.csv", index=False)
df.describe().T.round(2)

# Outliers by IQR rule (reported, not removed: all values are physically plausible readings)
q1, q3 = df.quantile(0.25), df.quantile(0.75)
iqr = q3 - q1
((df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)).sum()

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
labels = {"AT": "Temperature AT (°C)", "V": "Exhaust vacuum V (cm Hg)",
          "AP": "Ambient pressure AP (mbar)", "RH": "Relative humidity RH (%)"}
for ax, c in zip(axes, ["AT", "V", "AP", "RH"]):
    ax.scatter(df[c], df["PE"], s=3, alpha=0.25, color=BLUE, linewidths=0)
    r = df[[c, "PE"]].corr().iloc[0, 1]
    ax.set(xlabel=labels[c], title=f"r = {r:+.2f}")
axes[0].set_ylabel("Energy output PE (MW)")
save(fig, "cc_feature_vs_target")

fig, ax = plt.subplots(figsize=(5.2, 4.3))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, square=True,
            linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.75}, ax=ax)
ax.set(title="Pearson correlation")
ax.grid(False)
save(fig, "cc_correlation")

X = df[["AT", "V", "AP", "RH"]]
y = df["PE"]
Xc = np.column_stack([np.ones(len(X)), StandardScaler().fit_transform(X)])
vif = pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(4)], index=X.columns, name="VIF")
vif.round(2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(X_train.shape, X_test.shape)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge (α=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso (α=0.01)": make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
    "Polynomial (deg 3)": make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression()),
    "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance")),
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=50, epsilon=0.5)),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=4, random_state=SEED),
    "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.9, random_state=SEED, n_jobs=-1),
}

rows, fitted = [], {}
for name, model in models.items():
    cvres = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring=["r2", "neg_root_mean_squared_error"])
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_s = time.time() - t0
    pred = model.predict(X_test)
    fitted[name] = model
    rows.append({
        "Model": name,
        "CV R² (mean)": cvres["test_r2"].mean(), "CV R² (std)": cvres["test_r2"].std(),
        "CV RMSE": -cvres["test_neg_root_mean_squared_error"].mean(),
        "Train R²": r2_score(y_train, model.predict(X_train)),
        "Test MAE": mean_absolute_error(y_test, pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "Test R²": r2_score(y_test, pred),
        "Fit time (s)": fit_s,
    })
results = pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)
results.round(4)

grid = GridSearchCV(
    XGBRegressor(random_state=SEED, n_jobs=-1, subsample=0.8),
    {"n_estimators": [600, 1200], "learning_rate": [0.03, 0.05], "max_depth": [5, 7],
     "colsample_bytree": [0.8, 1.0]},
    cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_, "| CV RMSE:", round(-grid.best_score_, 4))
tuned = grid.best_estimator_
pred = tuned.predict(X_test)
tuned_row = {"Model": "XGBoost (tuned)", "CV R² (mean)": np.nan, "CV R² (std)": np.nan,
             "CV RMSE": -grid.best_score_, "Train R²": r2_score(y_train, tuned.predict(X_train)),
             "Test MAE": mean_absolute_error(y_test, pred),
             "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)), "Test R²": r2_score(y_test, pred),
             "Fit time (s)": np.nan}
cvt = cross_validate(tuned, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
tuned_row["CV R² (mean)"], tuned_row["CV R² (std)"] = cvt["test_score"].mean(), cvt["test_score"].std()
fitted["XGBoost (tuned)"] = tuned
results = pd.concat([results, pd.DataFrame([tuned_row])]).sort_values("Test RMSE").reset_index(drop=True)
results.to_csv(MODELS / "regression_results.csv", index=False)
results.round(4)

plot = results.sort_values("Test RMSE", ascending=False)
best_name = results.iloc[0]["Model"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
colors = [SERIES[1] if m == best_name else BLUE for m in plot["Model"]]
axes[0].barh(plot["Model"], plot["Test RMSE"], color=colors, height=0.7)
for yv, v in enumerate(plot["Test RMSE"]):
    axes[0].text(v + 0.05, yv, f"{v:.2f}", va="center", fontsize=9, color=INK_2)
axes[0].set(title="Test RMSE (MW) — lower is better", xlim=(0, plot["Test RMSE"].max() * 1.15))
axes[1].barh(plot["Model"], plot["Test R²"], color=colors, height=0.7)
for yv, v in enumerate(plot["Test R²"]):
    axes[1].text(v + 0.001, yv, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
axes[1].set(title="Test R² — higher is better", xlim=(0.9, 0.975))
for ax in axes:
    ax.grid(axis="y", visible=False)
save(fig, "cc_model_comparison")

best = fitted[best_name]
pred = best.predict(X_test)
res = y_test - pred
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
axes[0].scatter(y_test, pred, s=5, alpha=0.4, color=BLUE, linewidths=0)
lims = [y.min() - 2, y.max() + 2]
axes[0].plot(lims, lims, color=INK_2, lw=1, ls="--")
axes[0].set(xlim=lims, ylim=lims, xlabel="Actual PE (MW)", ylabel="Predicted PE (MW)", title="Actual vs predicted")
axes[1].scatter(pred, res, s=5, alpha=0.4, color=BLUE, linewidths=0)
axes[1].axhline(0, color=INK_2, lw=1, ls="--")
axes[1].set(xlabel="Predicted PE (MW)", ylabel="Residual (MW)", title="Residuals vs predicted")
axes[2].hist(res, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
axes[2].set(xlabel="Residual (MW)", ylabel="Count", title="Residual distribution")
fig.suptitle(f"{best_name}: test-set diagnostics", x=0.01, ha="left", fontweight="semibold")
save(fig, "cc_best_diagnostics")
print(f"Residual mean {res.mean():.3f}, std {res.std():.3f}; "
      f"{(res.abs() <= 5).mean():.1%} of predictions within ±5 MW")

perm = permutation_importance(best, X_test, y_test, n_repeats=15, random_state=SEED, scoring="r2")
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()
lin = fitted["Linear Regression"][-1]
coef = pd.Series(lin.coef_, index=X.columns)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
axes[0].barh(imp.index, imp.values, color=BLUE, height=0.6)
axes[0].set(title=f"Permutation importance ({best_name})", xlabel="Drop in R² when shuffled")
axes[1].barh(coef.index, coef.values, color=[SERIES[7] if v < 0 else BLUE for v in coef.values], height=0.6)
axes[1].axvline(0, color=AXIS, lw=1)
axes[1].set(title="Linear model coefficients (standardised)", xlabel="MW change per 1 std. increase")
for ax in axes:
    ax.grid(axis="y", visible=False)
save(fig, "cc_feature_importance")
pd.DataFrame({"permutation_importance": imp, "linear_coef": coef}).round(3)

lr = results.set_index("Model").loc["Linear Regression"]
b = results.iloc[0]
summary = {
    "rows": len(df), "train": len(X_train), "test": len(X_test),
    "best_model": best_name, "best_params": grid.best_params_,
    "best_rmse": b["Test RMSE"], "best_mae": b["Test MAE"], "best_r2": b["Test R²"],
    "best_cv_r2": b["CV R² (mean)"], "best_train_r2": b["Train R²"],
    "lr_rmse": lr["Test RMSE"], "lr_mae": lr["Test MAE"], "lr_r2": lr["Test R²"],
    "rmse_improvement_pct": (1 - b["Test RMSE"] / lr["Test RMSE"]) * 100,
    "within_5mw_pct": (res.abs() <= 5).mean() * 100,
    "pe_range": [float(y.min()), float(y.max())], "pe_std": float(y.std()),
    "importance": imp.sort_values(ascending=False).to_dict(), "linear_coef": coef.to_dict(),
    "vif": vif.to_dict(), "corr_with_pe": df.corr()["PE"].drop("PE").to_dict(),
}
save_json(summary, "regression_summary")
pd.Series(summary)
