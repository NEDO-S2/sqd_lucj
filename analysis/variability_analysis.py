"""繰り返し実機測定(ibm_pittsburgh, 元のトランスパイル済み LUCJ 回路)の
Run間バラつきを、Hamiltonian を使わずに解析するスクリプト。

各系(pb2/pb3/pb4)で「同一回路」を R=10 回独立に 1万ショット測定した。本スクリプトは
生の測定分布がRun毎にどれだけ変動するかを定量化する。活性空間 Hamiltonian を必要としない
ため、活性空間・エネルギー基準の曖昧さの影響を受けない。

ビット列の規約(sqd_pb_en.py と同一): 整数キー = int(ビット列, 2)。
α = 下位 norb ビット、β = 上位 norb ビット。
「電子数保存の配置」= popcount(α)==na かつ popcount(β)==nb。

指標
----
Runごと:
  - n_unique            出現した相異なるビット列の数
  - n_unique_phys       出現したビット列のうち電子数が正しいビット列の数
  - p_phys              電子数が正しいビット列のの割合(ハードウェア品質)
  - top1                最頻出のビット列が出現した全ビット列に占める割合
Run間(電子数保存の配置で再正規化した分布に対して):
  - pairwise TVD        全Runペアの全変動距離 (Total Variation Distance)
  - top-K Jaccard       上位 K 配置のRun間の重なり
                        (= SQD が対角化する部分空間の再現性)

結果は results_en/hw_variability/orig_pittsburgh/ 以下に CSV と4パネル図として出力する。
"""
import os
import json
import glob
import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = {
    "pb2": dict(norb=8, na=2, nb=2),
    "pb3": dict(norb=12, na=3, nb=3),
    "pb4": dict(norb=16, na=4, nb=4),
}
TOPK = [50, 100, 300, 1000]
COLORS = {"pb2": "#1f77b4", "pb3": "#ff7f0e", "pb4": "#2ca02c"}


def load_runs(sysd, shots=10000, base="../extracted"):
    """指定系の run*.json をすべて読み込み、{整数キー: 出現数} のリストとパス一覧を返す。"""
    d = os.path.join(HERE, base, sysd, "hw_runs", f"orig_shot{shots}")
    paths = sorted(p for p in glob.glob(os.path.join(d, "run*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            raw = json.load(f)
        # ビット列(2進文字列)を整数キーに変換
        counts = {int(k.replace(" ", ""), 2): int(v) for k, v in raw.items()}
        runs.append(counts)
    return runs, paths


def physical_mask(keys, norb, na, nb):
    """各配置キーが電子数保存かどうかの真偽配列を返す。"""
    keys = np.asarray(keys, dtype=np.int64)
    amask = (1 << norb) - 1            # 下位 norb ビットを取り出すマスク
    alpha = keys & amask               # α スピンの占有(下位 norb ビット)
    beta = (keys >> norb) & amask      # β スピンの占有(上位 norb ビット)
    pca = np.array([bin(int(x)).count("1") for x in alpha])  # α の 1 の個数
    pcb = np.array([bin(int(x)).count("1") for x in beta])   # β の 1 の個数
    return (pca == na) & (pcb == nb)


def run_metrics(counts, norb, na, nb):
    """1Run分の指標(ユニーク数・物理ユニーク数・p_phys・top1)を計算。"""
    keys = np.fromiter(counts.keys(), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64)
    total = vals.sum()                          # 通常は 10,000
    phys = physical_mask(keys, norb, na, nb)
    p_phys = float(vals[phys].sum() / total)    # 電子数保存ショットの割合
    return dict(
        n_unique=len(keys),
        n_unique_phys=int(phys.sum()),
        p_phys=p_phys,
        top1=float(vals.max() / total),
    )


def phys_distribution(counts, norb, na, nb):
    """電子数保存の配置のみを取り出し、和が1になるよう再正規化した分布 {キー: 確率} を返す。"""
    keys = np.fromiter(counts.keys(), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64)
    phys = physical_mask(keys, norb, na, nb)
    k = keys[phys]
    v = vals[phys]
    s = v.sum()
    if s == 0:
        return {}
    return dict(zip(k.tolist(), (v / s).tolist()))


def tvd(p, q):
    """2つの確率分布 p, q の全変動距離 TVD = 0.5 * Σ|p_i - q_i| を計算。"""
    keys = set(p) | set(q)             # 両Runに現れた配置の和集合
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def topk_set(counts, norb, na, nb, k):
    """電子数保存の配置の中で出現数の多い上位 K 配置の集合を返す。"""
    keys = np.fromiter(counts.keys(), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64)
    phys = physical_mask(keys, norb, na, nb)
    k_arr = keys[phys]
    v_arr = vals[phys]
    if len(k_arr) == 0:
        return set()
    order = np.argsort(-v_arr)[:k]     # 出現数の降順に並べて上位 K 個
    return set(k_arr[order].tolist())


def jaccard(a, b):
    """2集合の Jaccard 係数(重なり度) |a∩b| / |a∪b| を返す。"""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    out_dir = os.path.join(HERE, "results_en", "hw_variability", "orig_pittsburgh")
    os.makedirs(out_dir, exist_ok=True)

    per_run_rows = []
    summary_rows = []
    tvd_by_sys = {}
    jacc_by_sys = {}

    for sysd, cfg in SYS.items():
        norb, na, nb = cfg["norb"], cfg["na"], cfg["nb"]
        runs, paths = load_runs(sysd)
        if not runs:
            print(f"{sysd}: no runs found, skipping")
            continue
        # Runごとの指標
        ms = []
        for i, c in enumerate(runs):
            m = run_metrics(c, norb, na, nb)
            ms.append(m)
            per_run_rows.append([sysd, i + 1] + [m[k] for k in
                                ("n_unique", "n_unique_phys", "p_phys", "top1")])
        # Run間距離(全ペアの TVD)
        dists = [phys_distribution(c, norb, na, nb) for c in runs]
        tvds = [tvd(dists[i], dists[j]) for i, j in itertools.combinations(range(len(runs)), 2)]
        tvd_by_sys[sysd] = np.array(tvds)
        # 上位 K 配置の Jaccard 重なり(全ペア)
        jacc_by_sys[sysd] = {}
        for k in TOPK:
            sets = [topk_set(c, norb, na, nb, k) for c in runs]
            js = [jaccard(sets[i], sets[j]) for i, j in itertools.combinations(range(len(runs)), 2)]
            jacc_by_sys[sysd][k] = np.array(js)

        def col(name):
            return np.array([m[name] for m in ms])

        summary_rows.append([
            sysd, f"{na+nb}e{norb}o", len(runs),
            col("n_unique").mean(), col("n_unique").std(),
            col("n_unique_phys").mean(), col("n_unique_phys").std(),
            col("p_phys").mean(), col("p_phys").std(),
            col("top1").mean(), col("top1").std(),
            np.mean(tvds), np.std(tvds), np.min(tvds), np.max(tvds),
        ])
        print(f"{sysd}: R={len(runs)}  p_phys={col('p_phys').mean():.3f}"
              f"±{col('p_phys').std():.3f}  n_uniq={col('n_unique').mean():.0f}"
              f"±{col('n_unique').std():.0f}  TVD={np.mean(tvds):.3f}±{np.std(tvds):.3f}",
              flush=True)

    # CSV を出力
    with open(os.path.join(out_dir, "per_run_metrics.csv"), "w") as f:
        f.write("system,run,n_unique,n_unique_phys,p_phys,top1\n")
        for row in per_run_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    with open(os.path.join(out_dir, "variability_summary.csv"), "w") as f:
        f.write("system,active_space,R,"
                "n_unique_mean,n_unique_std,n_unique_phys_mean,n_unique_phys_std,"
                "p_phys_mean,p_phys_std,top1_mean,top1_std,"
                "tvd_mean,tvd_std,tvd_min,tvd_max\n")
        for row in summary_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # 図(5パネル, 2x3 レイアウトで6枠目は非表示)を作成。
    # 図中の文字は日本語フォント非依存にするため英語のまま
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    syslist = list(tvd_by_sys.keys())

    def scatter_per_system(ax, col_idx, marker, ylabel, title):
        """各系の per-Run 値(per_run_rows の col_idx 列)を散布図+平均棒で描く共通処理。"""
        for xi, s in enumerate(syslist):
            ys = [r[col_idx] for r in per_run_rows if r[0] == s]
            x = np.full(len(ys), xi) + np.random.uniform(-0.08, 0.08, len(ys))
            ax.scatter(x, ys, color=COLORS[s], s=40, alpha=0.8, marker=marker,
                       edgecolor="k", linewidth=0.4)
            ax.scatter([xi], [np.mean(ys)], color="k", marker="_", s=600)
        ax.set_xticks(range(len(syslist)))
        ax.set_xticklabels([f"{s}\n({SYS[s]['na']+SYS[s]['nb']}e{SYS[s]['norb']}o)"
                            for s in syslist])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    # パネルA: Runごとの電子数保存ショット率 p_phys (per_run_rows の 4 列目)
    scatter_per_system(axes[0, 0], 4, "o",
                       "fraction of shots with correct electron number  (p_phys)",
                       "(A) particle-number preservation per run")

    # パネルB1: Runごとの全ユニーク配置数 (per_run_rows の 2 列目)
    scatter_per_system(axes[0, 1], 2, "o",
                       "unique bitstrings / run (total)",
                       "(B1) num.of configurations -- total")

    # パネルB2: Runごとの電子数保存ユニーク配置数 (per_run_rows の 3 列目)
    scatter_per_system(axes[0, 2], 3, "^",
                       "unique bitstrings / run (physical)",
                       "(B2) num.of configurations -- physical configs")

    # パネルC: Run間 TVD の分布(箱ひげ)
    ax = axes[1, 0]
    data = [tvd_by_sys[s] for s in syslist]
    bp = ax.boxplot(data, tick_labels=syslist, showmeans=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], syslist):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.5)
    ax.set_ylabel("pairwise TVD (physical distribution)")
    ax.set_title("(C) run-to-run distribution distance")
    ax.grid(True, axis="y", alpha=0.3)

    # パネルD: 上位 K 配置の Jaccard 重なり(部分空間の再現性) vs K
    ax = axes[1, 1]
    for s in syslist:
        means = [jacc_by_sys[s][k].mean() for k in TOPK]
        stds = [jacc_by_sys[s][k].std() for k in TOPK]
        ax.errorbar(TOPK, means, yerr=stds, marker="o", color=COLORS[s], capsize=3, label=s)
    ax.set_xscale("log")
    ax.set_xlabel("subspace size K (most probable physical configs)")
    ax.set_ylabel("mean pairwise Jaccard overlap")
    ax.set_title("(D) reproducibility of selected subspace")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6枠目は使わないので非表示
    axes[1, 2].axis("off")

    fig.suptitle("Run-to-run variability of repeated HW measurements "
                 "(ibm_pittsburgh, original LUCJ circuits, R=10 x 10k shots)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(out_dir, "variability_overview.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")
    print(f"wrote {os.path.join(out_dir, 'per_run_metrics.csv')}")
    print(f"wrote {os.path.join(out_dir, 'variability_summary.csv')}")


if __name__ == "__main__":
    main()
