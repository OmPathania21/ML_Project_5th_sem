# --- Colab-friendly setup: installs, paths, dataset download, plot helpers (no repo dependency) ---
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hmmlearn"], check=False)

import os, json, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

RAW, PROCESSED, FIGURES, MODELS = Path("data/raw"), Path("data/processed"), Path("outputs/figures"), Path("outputs/models")
for d in (RAW, PROCESSED, FIGURES, MODELS):
    d.mkdir(parents=True, exist_ok=True)

HAR = RAW / "har" / "UCI HAR Dataset"
if not HAR.exists():
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    zpath = RAW / "har.zip"
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW / "har")
    with zipfile.ZipFile(RAW / "har" / "UCI HAR Dataset.zip") as z:
        z.extractall(RAW / "har")
    print("Downloaded and extracted UCI HAR Dataset")

BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9e2323"])

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 11, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.show()

def save_json(obj, name):
    (MODELS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))

set_style()
ACT = pd.read_csv(HAR / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
ACT_MAP = dict(zip(ACT.id, ACT.activity))
print(ACT_MAP)

def load_signal(split, axis):
    f = HAR / split / "Inertial Signals" / f"total_acc_{axis}_{split}.txt"
    return pd.read_csv(f, sep=r"\s+", header=None).values

def load_split(split):
    sig = np.stack([load_signal(split, a) for a in "xyz"], axis=-1)       # (windows, 128, 3)
    y = pd.read_csv(HAR / split / f"y_{split}.txt", header=None)[0].values
    subj = pd.read_csv(HAR / split / f"subject_{split}.txt", header=None)[0].values
    return sig, y, subj

sig_tr, y_tr, subj_tr = load_split("train")
sig_te, y_te, subj_te = load_split("test")
sig = np.concatenate([sig_tr, sig_te]); y = np.concatenate([y_tr, y_te]); subj = np.concatenate([subj_tr, subj_te])
print(f"Windows: {sig.shape[0]}, samples/window: {sig.shape[1]}, axes: {sig.shape[2]}, subjects: {len(set(subj))}")
print("Missing:", np.isnan(sig).sum(), "| Activities:", pd.Series(y).map(ACT_MAP).value_counts().to_dict())

def window_features(w):
    mag = np.linalg.norm(w, axis=1)
    jerk = np.diff(w, axis=0)
    return {
        "mean_mag": mag.mean(), "std_mag": mag.std(), "energy": (w ** 2).mean(),
        "mean_x": w[:, 0].mean(), "mean_y": w[:, 1].mean(), "mean_z": w[:, 2].mean(),
        "std_x": w[:, 0].std(), "std_y": w[:, 1].std(), "std_z": w[:, 2].std(),
        "jerk_energy": (jerk ** 2).mean(),
    }

feat = pd.DataFrame([window_features(sig[i]) for i in range(sig.shape[0])])
feat["activity_id"] = y
feat["activity"] = pd.Series(y).map(ACT_MAP).values
feat["subject"] = subj
feat.to_csv(PROCESSED / "har_features.csv", index=False)
feat.describe().T.round(3)

fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
s0 = subj_tr[0]
idx = np.where(subj_tr == s0)[0][:40]
flat = sig_tr[idx].reshape(-1, 3)
act_flat = np.repeat(y_tr[idx], sig_tr.shape[1])
axes[0].plot(np.linalg.norm(flat, axis=1), color=BLUE, lw=0.6)
axes[0].set(title=f"Total acceleration magnitude, subject {s0} (first 40 windows)")
axes[1].plot(act_flat, color=SERIES[1], lw=1)
axes[1].set(title="True activity id over the same span (the hidden state we will try to recover)", ylabel="Activity id")
save(fig, "har_signal_and_activity")

order = ACT.activity.tolist()
fig, ax = plt.subplots(figsize=(8, 4.2))
sns.boxplot(data=feat, x="activity", y="mean_mag", order=order, color="#9ec5f4", ax=ax, fliersize=1.5)
ax.set(title="Mean acceleration magnitude by activity", xlabel="")
ax.tick_params(axis="x", rotation=30)
save(fig, "har_feature_by_activity")

fig, ax = plt.subplots(figsize=(8, 4.2))
sns.boxplot(data=feat, x="activity", y="jerk_energy", order=order, color="#9ec5f4", ax=ax, fliersize=1.5)
ax.set(title="Jerk energy (movement dynamics) separates static vs dynamic activities", xlabel="", yscale="log")
ax.tick_params(axis="x", rotation=30)
save(fig, "har_jerk_by_activity")

num_cols = [c for c in feat.columns if c not in ("activity_id", "activity", "subject")]
corr = feat[num_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, cmap=DIVERGING, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, linewidths=1,
            linecolor=SURFACE, cbar_kws={"shrink": 0.75}, annot_kws={"fontsize": 7.5}, ax=ax)
ax.set(title="Feature correlation"); ax.grid(False)
save(fig, "har_feature_correlation")
