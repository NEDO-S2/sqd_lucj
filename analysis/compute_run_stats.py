"""繰り返し測定データ（ibm_pittsburgh, 元LUCJ回路）のRun間統計を計算するスクリプト。

pb2 / pb3 / pb4 それぞれについて、
  extracted/<sys>/hw_runs/orig_shot10000/run*.json
から各Runの測定カウントを読み込み、以下の4指標の「平均値」と「標準偏差」を求める。

  1. ユニーク配置数        … 10,000ショット中に現れた相異なるビット列の数
  2. 電子数保存確率 p_phys … 正しい粒子数（活性空間の電子数）になっていたショットの割合
  3. top-1確率             … 最も多く出たビット列1個が全ショットに占める割合
  4. Run間バラつき TVD    … 2Runの分布間の全変動距離（全ペアの平均・標準偏差）(Total Variation Distance)

ビット列の規約（sqd_pb_en.py と同一）:
  整数キー = int(ビット列, 2)。下位 norb ビット = α スピン、上位 norb ビット = β スピン。
  「電子数保存（物理）」= popcount(α)==Na かつ popcount(β)==Nb。

使い方:
  python compute_run_stats.py
"""
import os
import json
import glob
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 各系の活性空間: norb=空間軌道数, na/nb=α/β スピンの電子数
SYS = {
    "pb2": dict(norb=8, na=2, nb=2),
    "pb3": dict(norb=12, na=3, nb=3),
    "pb4": dict(norb=16, na=4, nb=4),
}


def load_runs(sysd, shots=10000, base="../extracted"):
    """指定系の run*.json をすべて読み込み、{整数キー: 出現数} のリストを返す。"""
    d = os.path.join(HERE, base, sysd, "hw_runs", f"orig_shot{shots}")
    paths = sorted(glob.glob(os.path.join(d, "run*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            raw = json.load(f)
        # ビット列(2進文字列)を整数キーに変換
        counts = {int(k.replace(" ", ""), 2): int(v) for k, v in raw.items()}
        runs.append(counts)
    return runs


def physical_mask(keys, norb, na, nb):
    """各配置キーが電子数保存（物理セクター）かどうかの真偽配列を返す。"""
    keys = np.asarray(keys, dtype=np.int64)
    amask = (1 << norb) - 1            # 下位 norb ビットを取り出すマスク
    alpha = keys & amask               # α スピンの占有
    beta = (keys >> norb) & amask      # β スピンの占有
    # 各キーの 1 の個数（= 占有電子数）を数える
    pca = np.array([bin(int(x)).count("1") for x in alpha])
    pcb = np.array([bin(int(x)).count("1") for x in beta])
    return (pca == na) & (pcb == nb)


def run_basic_metrics(counts, norb, na, nb):
    """1Run分の「ユニーク配置数・電子数保存確率・top-1確率」を計算。"""
    keys = np.fromiter(counts.keys(), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.float64)
    total = vals.sum()                 # 通常は 10,000
    phys = physical_mask(keys, norb, na, nb)
    n_unique = len(keys)               # ユニーク配置数
    p_phys = vals[phys].sum() / total  # 電子数保存確率
    top1 = vals.max() / total          # top-1 確率
    return n_unique, p_phys, top1


def phys_distribution(counts, norb, na, nb):
    """物理セクターのみを取り出し、和が1になるよう再正規化した確率分布 {キー: 確率} を返す。"""
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


def pairwise_tvd(runs, norb, na, nb):
    """全Runペア (R choose 2) の TVD を計算してリストで返す。"""
    dists = [phys_distribution(c, norb, na, nb) for c in runs]
    return [tvd(dists[i], dists[j])
            for i, j in itertools.combinations(range(len(runs)), 2)]


def main():
    # 結果をCSVにも保存する
    out_path = os.path.join(HERE, "run_stats_summary.csv")
    header = ("system,active_space,R,"
              "n_unique_mean,n_unique_std,"
              "p_phys_mean,p_phys_std,"
              "top1_mean,top1_std,"
              "tvd_mean,tvd_std")
    lines = [header]

    print(f"{'系':<5}{'活性空間':<10}{'R':>3}  "
          f"{'ユニーク配置 (平均±SD)':<22}{'電子数保存 (平均±SD)':<22}"
          f"{'top-1 (平均±SD)':<20}{'Run間TVD (平均±SD)':<22}")
    print("-" * 120)

    for sysd, cfg in SYS.items():
        norb, na, nb = cfg["norb"], cfg["na"], cfg["nb"]
        runs = load_runs(sysd)
        if not runs:
            print(f"{sysd}: run*.json が見つかりません（スキップ）")
            continue

        # 各Runの基本3指標
        n_unique, p_phys, top1 = [], [], []
        for c in runs:
            nu, pp, t1 = run_basic_metrics(c, norb, na, nb)
            n_unique.append(nu)
            p_phys.append(pp)
            top1.append(t1)
        n_unique = np.array(n_unique, dtype=float)
        p_phys = np.array(p_phys)
        top1 = np.array(top1)

        # Run間 TVD（全ペア）
        tvds = np.array(pairwise_tvd(runs, norb, na, nb))

        R = len(runs)
        aspace = f"{na+nb}e{norb}o"
        print(f"{sysd:<5}{aspace:<10}{R:>3}  "
              f"{n_unique.mean():7.1f} ± {n_unique.std():5.1f}     "
              f"{p_phys.mean():.4f} ± {p_phys.std():.4f}     "
              f"{top1.mean():.4f} ± {top1.std():.4f}     "
              f"{tvds.mean():.4f} ± {tvds.std():.4f}")

        lines.append(
            f"{sysd},{aspace},{R},"
            f"{n_unique.mean():.4f},{n_unique.std():.4f},"
            f"{p_phys.mean():.6f},{p_phys.std():.6f},"
            f"{top1.mean():.6f},{top1.std():.6f},"
            f"{tvds.mean():.6f},{tvds.std():.6f}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("-" * 120)
    print(f"集計結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
