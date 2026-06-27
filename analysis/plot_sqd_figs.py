"""保存済み CSV から sqd_compare.png / sqd_convergence.png を再描画する（計算なし）。

compare_sqd_runs.py / compare_convergence.py は SQD を実際に走らせて CSV と図を
出力する「重い」スクリプト。本スクリプトはそれらが残した CSV だけを読み、
図の見た目（軸ラベル・凡例位置・色・タイトル等）の調整を高速に反映する。

入力 CSV（既定の出力先 results_en/.../sqd_compare/ にある想定）:
  - sqd_compare_runs.csv      … 列: system, run, E_*, err_standard, err_en
  - sqd_convergence_runs.csv  … 列: system, run, iter, err_standard, err_en

使い方:
  uv run plot_sqd_figs.py                 # 両方を再描画
  uv run plot_sqd_figs.py --which compare
  uv run plot_sqd_figs.py --which convergence
  uv run plot_sqd_figs.py --tag rerun     # *_rerun.csv / *_rerun.png を対象
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results_en", "hw_variability", "orig_pittsburgh", "sqd_compare")

COLORS = {"standard": "#1f77b4", "en": "#8B0000"}
LABELS = {"standard": "Standard SQD", "en": "AS-SQD (EN)"}
YLABEL = "|E - CASCI|  [Hartree]"
CHEM_ACC = 0.0016


def _systems_in(df):
    order = ["pb2", "pb3", "pb4"]
    present = [s for s in order if s in set(df.system)]
    return present + [s for s in df.system.unique() if s not in order]


def plot_compare(out_dir, tag=""):
    """Run 別最終誤差の散布図（standard vs en）。"""
    csv = os.path.join(out_dir, f"sqd_compare_runs{tag}.csv")
    df = pd.read_csv(csv)
    systems = _systems_in(df)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), 5), squeeze=False)
    for j, sysd in enumerate(systems):
        ax = axes[0, j]
        sub = df[df.system == sysd]
        for xi, meth in enumerate(("standard", "en")):
            ys = sub[f"err_{meth}"].values
            x = np.full(len(ys), xi) + rng.uniform(-0.08, 0.08, len(ys))
            ax.scatter(x, ys, color=COLORS[meth], s=45, alpha=0.8,
                       edgecolor="k", linewidth=0.4)
            ax.scatter([xi], [ys.mean()], color="k", marker="_", s=600)
        ax.axhline(CHEM_ACC, color="#BF5700", ls=":", label="chemical accuracy")
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Standard SQD", "AS-SQD (EN)"])
        ax.set_ylabel(YLABEL)
        ax.set_title(f"{sysd}")
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.99, 0.93))
    fig.suptitle("Standard SQD vs AS-SQD(EN) over 10 HW runs "
                 "(ibm_pittsburgh)", fontsize=12) # original LUCJ circuits
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"sqd_compare{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}  (from {os.path.basename(csv)})")


def plot_compare_energy(out_dir, tag=""):
    """sqd_compare と同構成で、CASCI 差を取る前の生エネルギー E[Hartree] を散布。

    参照 CASCI(self) は変分原理(E >= E_CASCI)より E - err で CSV から復元する。
    縦軸は線形(系ごとに自動スケール)で、Standard が EN より上(高エネルギー)に
    位置することと Run 間バラつきがそのまま読み取れる。
    """
    csv = os.path.join(out_dir, f"sqd_compare_runs{tag}.csv")
    df = pd.read_csv(csv)
    systems = _systems_in(df)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), 5), squeeze=False)
    for j, sysd in enumerate(systems):
        ax = axes[0, j]
        sub = df[df.system == sysd]
        casci = float(np.median(np.concatenate([
            sub["E_standard"].values - sub["err_standard"].values,
            sub["E_en"].values - sub["err_en"].values])))
        for xi, meth in enumerate(("standard", "en")):
            ys = sub[f"E_{meth}"].values
            x = np.full(len(ys), xi) + rng.uniform(-0.08, 0.08, len(ys))
            ax.scatter(x, ys, color=COLORS[meth], s=45, alpha=0.8,
                       edgecolor="k", linewidth=0.4)
            ax.scatter([xi], [ys.mean()], color="k", marker="_", s=600)
        ax.axhline(casci, color="#2ca02c", ls="--", label="CASCI")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Standard SQD", "AS-SQD (EN)"])
        ax.set_ylabel("E  [Hartree]")
        ax.set_title(f"{sysd}")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Total energy (before subtracting CASCI): Standard SQD vs AS-SQD(EN) "
                 "over 10 HW runs (ibm_pittsburgh)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"sqd_compare_energy{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}  (from {os.path.basename(csv)})")


def plot_convergence(out_dir, tag=""):
    """反復 vs |E-CASCI| の収束カーブ（平均 + run min-max バンド）。"""
    csv = os.path.join(out_dir, f"sqd_convergence_runs{tag}.csv")
    df = pd.read_csv(csv)
    systems = _systems_in(df)
    iters = int(df["iter"].max())
    x = np.arange(1, iters + 1)
    fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), 5), squeeze=False)
    for j, sysd in enumerate(systems):
        ax = axes[0, j]
        sub = df[df.system == sysd]
        for meth in ("standard", "en"):
            col = f"err_{meth}"
            M = np.vstack([sub[sub.run == r].sort_values("iter")[col].values
                           for r in sorted(sub.run.unique())])
            mean, lo, hi = M.mean(axis=0), M.min(axis=0), M.max(axis=0)
            c = COLORS[meth]
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=c, marker="o", markersize=4, linewidth=2,
                    label=LABELS[meth])
        ax.axhline(CHEM_ACC, color="#BF5700", ls=":", label="chemical accuracy")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(YLABEL)
        ax.set_title(f"{sysd}")
        ax.set_xticks(x[::max(1, iters // 10)])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Energy convergence: Standard SQD vs AS-SQD(EN)   "
                 "[ line = mean , shaded band = min to max over 10 HW runs ]  (ibm_pittsburgh)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"sqd_convergence{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}  (from {os.path.basename(csv)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["compare", "energy", "convergence", "both", "all"],
                    default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    if args.which in ("compare", "both", "all"):
        plot_compare(args.out_dir, tag)
    if args.which in ("energy", "all"):
        plot_compare_energy(args.out_dir, tag)
    if args.which in ("convergence", "both", "all"):
        plot_convergence(args.out_dir, tag)


if __name__ == "__main__":
    main()
