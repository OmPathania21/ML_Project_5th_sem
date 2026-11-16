import sys; sys.path.insert(0, "..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.viz import *

set_style()
pd.set_option("display.max_columns", 30)
df_raw = pd.read_csv(RAW / "play_store" / "googleplaystore.csv")
print(df_raw.shape)
df_raw.head()

audit = pd.DataFrame({
    "dtype": df_raw.dtypes.astype(str),
    "missing": df_raw.isna().sum(),
    "missing_%": (df_raw.isna().mean() * 100).round(2),
    "unique": df_raw.nunique(),
})
print("Exact duplicate rows:", df_raw.duplicated().sum())
print("Duplicate app names:", df_raw["App"].duplicated().sum())
audit

# Row whose columns are shifted by one (Category = '1.9', Rating = 19)
df_raw[~df_raw["Reviews"].str.isnumeric()]

df = df_raw[df_raw["Reviews"].str.isnumeric()].copy()
df = df.drop_duplicates()
df["Reviews"] = df["Reviews"].astype(int)
df = df.sort_values("Reviews", ascending=False).drop_duplicates("App").reset_index(drop=True)

def size_mb(s):
    if s.endswith("M"):
        return float(s[:-1])
    if s.endswith("k"):
        return float(s[:-1]) / 1024
    return np.nan

df["Size_MB"] = df["Size"].map(size_mb)
df["Installs_n"] = df["Installs"].str.replace(r"[+,]", "", regex=True).astype(int)
df["Price_usd"] = df["Price"].str.replace("$", "", regex=False).astype(float)
df["Last_Updated"] = pd.to_datetime(df["Last Updated"], format="%B %d, %Y")
df["Days_Since_Update"] = (df["Last_Updated"].max() - df["Last_Updated"]).dt.days
df["Type"] = np.where(df["Price_usd"] > 0, "Paid", "Free")
df["Primary_Genre"] = df["Genres"].str.split(";").str[0]
df["Log_Installs"] = np.log10(df["Installs_n"].clip(lower=1))
df["Log_Reviews"] = np.log10(df["Reviews"] + 1)

print(f"Rows: raw {len(df_raw):,} → clean {len(df):,} ({len(df_raw) - len(df):,} removed)")
print("Rating missing after cleaning:", df["Rating"].isna().sum(), f"({df['Rating'].isna().mean():.1%})")
print("Size unknown ('Varies with device'):", df["Size_MB"].isna().sum())
df.to_csv(PROCESSED / "play_store_clean.csv", index=False)
df[["Rating", "Reviews", "Size_MB", "Installs_n", "Price_usd", "Days_Since_Update"]].describe().T.round(2)

missing_by_install = df.assign(missing=df["Rating"].isna()).groupby("Installs_n")["missing"].mean()
print("Share of apps with missing rating, by install bucket:")
print((missing_by_install * 100).round(1).head(10))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["Rating"].dropna(), bins=np.arange(1, 5.1, 0.1), color=BLUE, edgecolor=SURFACE, linewidth=0.6)
med = df["Rating"].median()
ax.axvline(med, color=INK_2, lw=1.2, ls="--")
ax.text(med - 0.45, ax.get_ylim()[1] * 0.93, f"median {med:.1f} →", ha="right", color=INK_2)
ax.set(title="App ratings cluster between 4.0 and 4.6", xlabel="Rating", ylabel="Number of apps")
save(fig, "ps_rating_distribution")

top_cat = df["Category"].value_counts().head(12).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(top_cat.index.str.replace("_", " ").str.title(), top_cat.values, color=BLUE, height=0.7)
for y, v in enumerate(top_cat.values):
    ax.text(v + 15, y, f"{v:,}", va="center", fontsize=9, color=INK_2)
ax.set(title="Family, Game and Tools dominate the store", xlabel="Number of apps")
ax.grid(axis="y", visible=False)
save(fig, "ps_top_categories")

fig, ax = plt.subplots(figsize=(7, 4))
inst = df["Installs"].value_counts()
inst = inst.reindex(sorted(inst.index, key=lambda s: int(s.replace(",", "").replace("+", ""))))
ax.bar(range(len(inst)), inst.values, color=BLUE, width=0.75)
ax.set_xticks(range(len(inst)), inst.index, rotation=60, ha="right", fontsize=8)
ax.set(title="Installs are extremely skewed (log-scale buckets)", ylabel="Number of apps")
ax.grid(axis="x", visible=False)
save(fig, "ps_installs_distribution")

cat_stats = (df.groupby("Category")
               .agg(apps=("App", "size"), median_installs=("Installs_n", "median"),
                    total_installs=("Installs_n", "sum"), mean_rating=("Rating", "mean"))
               .sort_values("total_installs", ascending=False))
cat_stats.head(10)

top_inst = (cat_stats["total_installs"].head(10) / 1e9).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh(top_inst.index.str.replace("_", " ").str.title(), top_inst.values, color=BLUE, height=0.7)
for y, v in enumerate(top_inst.values):
    ax.text(v + 0.3, y, f"{v:.1f}B", va="center", fontsize=9, color=INK_2)
ax.set(title="Games and Communication win on total installs", xlabel="Total installs (billions)")
ax.grid(axis="y", visible=False)
save(fig, "ps_installs_by_category")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
share = df["Type"].value_counts()
axes[0].bar(share.index, share.values, color=[SERIES[0], SERIES[1]], width=0.6)
for i, v in enumerate(share.values):
    axes[0].text(i, v + 100, f"{v:,}\n({v / share.sum():.1%})", ha="center", color=INK_2, fontsize=9)
axes[0].set(title="Number of apps", ylim=(0, share.max() * 1.18))
axes[0].grid(axis="x", visible=False)
sns.boxplot(data=df, x="Type", y="Rating", order=share.index, hue="Type", hue_order=share.index,
            palette=[SERIES[0], SERIES[1]], width=0.5, fliersize=2, linewidth=1, ax=axes[1], legend=False)
axes[1].set(title="Rating", xlabel="")
fig.suptitle("Free vs Paid apps", x=0.02, ha="left", fontweight="semibold")
save(fig, "ps_free_vs_paid")
df.groupby("Type").agg(apps=("App", "size"), mean_rating=("Rating", "mean"),
                       median_installs=("Installs_n", "median"), median_reviews=("Reviews", "median"))

num = df[["Rating", "Log_Reviews", "Log_Installs", "Size_MB", "Price_usd", "Days_Since_Update"]]
corr = num.corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.4, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, square=True,
            linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.75}, ax=ax,
            annot_kws={"fontsize": 9})
ax.set(title="Spearman correlation (numeric features)")
ax.grid(False)
save(fig, "ps_correlation")
corr.round(2)

fig, ax = plt.subplots(figsize=(7, 4.4))
sample = df.sample(3000, random_state=0)
ax.scatter(sample["Reviews"] + 1, sample["Installs_n"].clip(lower=1), s=9, alpha=0.35, color=BLUE, linewidths=0)
ax.set(xscale="log", yscale="log", xlabel="Reviews (log)", ylabel="Installs (log)",
       title="Reviews grow almost linearly with installs")
rho = df[["Reviews", "Installs_n"]].corr(method="spearman").iloc[0, 1]
ax.text(0.03, 0.92, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, color=INK_2)
save(fig, "ps_reviews_vs_installs")

order = (df.groupby("Category")["Rating"].median().sort_values(ascending=False))
sel = list(order.index[:6]) + list(order.index[-6:])
fig, ax = plt.subplots(figsize=(8, 4.6))
sns.boxplot(data=df[df["Category"].isin(sel)], y="Category", x="Rating", order=sel, color="#9ec5f4",
            fliersize=1.5, linewidth=0.9, ax=ax)
ax.set_yticks(range(len(sel)), [s.replace("_", " ").title() for s in sel], fontsize=9)
ax.set(title="Highest vs lowest rated categories (median rating)", ylabel="")
ax.axhline(5.5, color=AXIS, lw=1, ls=":")
save(fig, "ps_rating_by_category")

fig, ax = plt.subplots(figsize=(7, 4))
df["Update_Year"] = df["Last_Updated"].dt.year
upd = df.groupby(pd.cut(df["Days_Since_Update"], [-1, 30, 180, 365, 730, 5000],
                        labels=["≤1 mo", "1–6 mo", "6–12 mo", "1–2 yr", ">2 yr"]), observed=True)["Rating"].mean()
ax.bar(upd.index.astype(str), upd.values, color=BLUE, width=0.6)
for i, v in enumerate(upd.values):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", color=INK_2, fontsize=9)
ax.set(ylim=(3.8, 4.4), title="Recently updated apps are rated higher", xlabel="Time since last update",
       ylabel="Mean rating")
ax.grid(axis="x", visible=False)
save(fig, "ps_rating_vs_update")

paid = df[df["Type"] == "Paid"]
insights = {
    "rows_raw": len(df_raw), "rows_clean": len(df), "removed": len(df_raw) - len(df),
    "rating_missing_pct": df["Rating"].isna().mean() * 100,
    "rating_median": df["Rating"].median(),
    "rating_share_ge4": (df["Rating"] >= 4).mean() / df["Rating"].notna().mean() * 100,
    "free_share": (df["Type"] == "Free").mean() * 100,
    "free_rating": df.loc[df["Type"] == "Free", "Rating"].mean(),
    "paid_rating": paid["Rating"].mean(),
    "free_median_installs": df.loc[df["Type"] == "Free", "Installs_n"].median(),
    "paid_median_installs": paid["Installs_n"].median(),
    "paid_median_price": paid["Price_usd"].median(),
    "top_category_count": df["Category"].value_counts().index[0],
    "top_category_installs": cat_stats.index[0],
    "top_category_installs_B": cat_stats["total_installs"].iloc[0] / 1e9,
    "spearman_reviews_installs": rho,
    "spearman_rating_reviews": corr.loc["Rating", "Log_Reviews"],
    "spearman_rating_size": corr.loc["Rating", "Size_MB"],
    "spearman_rating_update": corr.loc["Rating", "Days_Since_Update"],
    "rating_recent_update": upd.iloc[0], "rating_old_update": upd.iloc[-1],
    "best_category": order.index[0], "worst_category": order.index[-1],
    "missing_rating_low_installs_pct": df.loc[df["Rating"].isna(), "Installs_n"].le(1000).mean() * 100,
}
save_json(insights, "play_store_insights")
pd.Series(insights)
