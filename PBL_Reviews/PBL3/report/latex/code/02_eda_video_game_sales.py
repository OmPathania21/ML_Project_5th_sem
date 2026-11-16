import sys; sys.path.insert(0, "..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.viz import *

set_style()
df_raw = pd.read_csv(RAW / "video_game_sales" / "vgsales.csv")
print(df_raw.shape)
df_raw.head()

print("Duplicate rows:", df_raw.duplicated().sum())
print("Duplicate (Name, Platform, Year):", df_raw.duplicated(["Name", "Platform", "Year"]).sum())
pd.DataFrame({"dtype": df_raw.dtypes.astype(str), "missing": df_raw.isna().sum(),
              "unique": df_raw.nunique()})

df_raw["Year"].value_counts().sort_index().tail(8)

# Do regional columns add up to Global_Sales? (rounding differences expected)
diff = (df_raw[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum(axis=1) - df_raw["Global_Sales"]).abs()
print(f"Max |sum(regions) − global| = {diff.max():.2f}M; rows off by > 0.02M: {(diff > 0.02).sum()}")

df = df_raw.dropna(subset=["Year"]).copy()
df["Year"] = df["Year"].astype(int)
df = df[df["Year"] <= 2016]
df["Publisher"] = df["Publisher"].fillna("Unknown")
REGIONS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
REGION_NAMES = {"NA_Sales": "North America", "EU_Sales": "Europe", "JP_Sales": "Japan", "Other_Sales": "Other"}
print(f"Rows: {len(df_raw):,} → {len(df):,}")
df.to_csv(PROCESSED / "vgsales_clean.csv", index=False)
df[REGIONS + ["Global_Sales"]].describe().T.round(3)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(df["Global_Sales"]), bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
ticks = [0.01, 0.1, 1, 10, 80]
ax.set_xticks(np.log10(ticks), [f"{t:g}M" for t in ticks])
ax.set(title="Most games sell < 0.5M copies; a few sell tens of millions",
       xlabel="Global sales (log scale)", ylabel="Number of games")
save(fig, "vg_sales_distribution")
top1 = df.nlargest(int(len(df) * 0.01), "Global_Sales")["Global_Sales"].sum() / df["Global_Sales"].sum()
print(f"Skewness: {df['Global_Sales'].skew():.1f};  top 1% of games = {top1:.1%} of all sales")

yearly = df.groupby("Year").agg(games=("Name", "size"), sales=("Global_Sales", "sum"))
fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
axes[0].plot(yearly.index, yearly["sales"], color=SERIES[0])
axes[0].fill_between(yearly.index, yearly["sales"], color=SERIES[0], alpha=0.08)
axes[0].set(title="Global sales per release year (millions)")
pk = yearly["sales"].idxmax()
axes[0].annotate(f"peak {pk}: {yearly.loc[pk, 'sales']:.0f}M", (pk, yearly.loc[pk, "sales"]),
                 xytext=(-110, -8), textcoords="offset points", color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=MUTED))
axes[1].bar(yearly.index, yearly["games"], color=SERIES[0], width=0.75)
axes[1].set(title="Games released per year", xlabel="Release year")
axes[1].grid(axis="x", visible=False)
save(fig, "vg_lifecycle")
yearly.tail(12)

genre = df.groupby("Genre").agg(games=("Name", "size"), sales=("Global_Sales", "sum"),
                                avg_sales=("Global_Sales", "mean")).sort_values("sales")
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(genre.index, genre["sales"], color=BLUE, height=0.7)
for y, v in enumerate(genre["sales"]):
    ax.text(v + 15, y, f"{v:,.0f}M", va="center", fontsize=9, color=INK_2)
ax.set(title="Action and Sports sell the most copies overall", xlabel="Global sales (millions)")
ax.grid(axis="y", visible=False)
save(fig, "vg_sales_by_genre")
genre.sort_values("avg_sales", ascending=False).round(3)

reg = df.groupby("Genre")[REGIONS].sum()
reg_share = reg.div(reg.sum(axis=1), axis=0).sort_values("JP_Sales")
fig, ax = plt.subplots(figsize=(8.4, 4.8))
left = np.zeros(len(reg_share))
for i, r in enumerate(REGIONS):
    ax.barh(reg_share.index, reg_share[r], left=left, color=SERIES[i], height=0.72,
            edgecolor=SURFACE, linewidth=1.5, label=REGION_NAMES[r])
    left += reg_share[r].values
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.set(xlim=(0, 1), title="Regional share of sales within each genre")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.22))
ax.grid(False)
save(fig, "vg_region_share_by_genre")
(reg_share * 100).round(1)

region_top = {REGION_NAMES[r]: reg[r].sort_values(ascending=False).index[:3].tolist() for r in REGIONS}
region_top

plat = df.groupby("Platform")["Global_Sales"].sum().nlargest(12).sort_values()
pub = df.groupby("Publisher")["Global_Sales"].sum().nlargest(10).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, s, t in [(axes[0], plat, "Top 12 platforms"), (axes[1], pub, "Top 10 publishers")]:
    ax.barh(s.index, s.values, color=BLUE, height=0.7)
    for y, v in enumerate(s.values):
        ax.text(v + 10, y, f"{v:,.0f}", va="center", fontsize=8.5, color=INK_2)
    ax.set(title=t, xlabel="Global sales (millions)")
    ax.grid(axis="y", visible=False)
save(fig, "vg_platforms_publishers")
top5_pub_share = df.groupby("Publisher")["Global_Sales"].sum().nlargest(5).sum() / df["Global_Sales"].sum()
print(f"Publishers: {df['Publisher'].nunique()}; top-5 share of sales = {top5_pub_share:.1%}")

region_year = df.groupby("Year")[REGIONS].sum().loc[1995:]
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, r in enumerate(REGIONS):
    ax.plot(region_year.index, region_year[r], color=SERIES[i], label=REGION_NAMES[r])
ax.set(title="North America drives the boom and the decline", ylabel="Sales (millions)", xlim=(1995, 2016.5))
ax.legend(loc="upper left")
save(fig, "vg_region_trend")

corr = df[REGIONS + ["Global_Sales"]].rename(columns=lambda c: c.replace("_Sales", "")).corr()
fig, ax = plt.subplots(figsize=(5.6, 4.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, square=True,
            linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.75}, ax=ax)
ax.set(title="Correlation between regional sales")
ax.grid(False)
save(fig, "vg_correlation")
corr.round(2)

insights = {
    "rows_raw": len(df_raw), "rows_clean": len(df),
    "year_min": int(df["Year"].min()), "year_max": int(df["Year"].max()),
    "median_sales": df["Global_Sales"].median(), "max_game": df.loc[df["Global_Sales"].idxmax(), "Name"],
    "max_sales": df["Global_Sales"].max(), "skew": df["Global_Sales"].skew(), "top1pct_share": top1 * 100,
    "peak_year": int(pk), "peak_sales": yearly.loc[pk, "sales"], "peak_games_year": int(yearly["games"].idxmax()),
    "top_genre": genre["sales"].idxmax(), "top_genre_sales": genre["sales"].max(),
    "top_avg_genre": genre["avg_sales"].idxmax(), "top_avg_value": genre["avg_sales"].max(),
    "region_share": (df[REGIONS].sum() / df["Global_Sales"].sum() * 100).round(1).to_dict(),
    "jp_top_genre": reg["JP_Sales"].idxmax(), "jp_rpg_share": reg_share.loc["Role-Playing", "JP_Sales"] * 100,
    "na_shooter_share": reg_share.loc["Shooter", "NA_Sales"] * 100,
    "top_platform": plat.index[-1], "top_platform_sales": plat.iloc[-1],
    "top_publisher": pub.index[-1], "top_publisher_sales": pub.iloc[-1],
    "n_publishers": df["Publisher"].nunique(), "top5_pub_share": top5_pub_share * 100,
    "corr_na_global": corr.loc["NA", "Global"], "corr_jp_global": corr.loc["JP", "Global"],
    "region_top_genres": region_top,
}
save_json(insights, "vgsales_insights")
pd.Series(insights)
