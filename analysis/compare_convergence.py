"""Standard SQD と AS-SQD(EN) の「エネルギー収束」を反復回数に対して比較する。

compare_sqd_runs.py が最終誤差の Run 間バラつきを示すのに対し、本スクリプトは
各 Run の反復ごとのエネルギー履歴を集め、|E - CASCI(self)| が反復に伴って
どう下がるか（収束の速さ・到達精度）を Standard と EN で比較する。

各系(pb2/pb3/pb4)について:
  - extracted/<sys>/hw_runs/orig_shot<N>[_<tag>]/run*.json の各 Run で
    Standard SQD と AS-SQD(EN) を実行し、反復ごとの |E - CASCI| を記録、
  - Run 間の平均線と min〜max バンドを描く（3 パネル）。
参照は ham_cache の自己無撞着 CASCI(casci_s0)。

使い方:
  uv run compare_convergence.py --systems pb2,pb3,pb4 --shots 10000 \
      --iters 20 --badd 600 --kinit 100 --samples 100 --batches 5 [--runs N] [--tag rerun]
"""
import os
import glob
import time
import argparse

import numpy as np
import pandas as pd

from sqd_pb_en import load_hamiltonian
from compare_sqd_runs import load_run_counts, run_standard, run_en
from plot_sqd_figs import plot_convergence

HERE = os.path.dirname(os.path.abspath(__file__))


def pad_to(hist, n):
    """履歴を長さ n に最終値でパディング（早期収束で反復が短い場合に整列）。"""
    h = list(hist)
    if len(h) < n:
        h = h + [h[-1]] * (n - len(h))
    return np.array(h[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="pb2,pb3,pb4")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--badd", type=int, default=600)
    ap.add_argument("--kinit", type=int, default=100)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--runs", type=int, default=0)
    ap.add_argument("--base", default="../extracted")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    cache_dir = os.path.join(HERE, "ham_cache")
    out_dir = os.path.join(HERE, "results_en", "hw_variability", "orig_pittsburgh", "sqd_compare")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sysd in systems:
        hcore, eri, e_nuc, norb, nelec, casci_s0 = load_hamiltonian(cache_dir, sysd)
        ref = casci_s0
        dirname = f"orig_shot{args.shots}" + (f"_{args.tag}" if args.tag else "")
        run_dir = os.path.join(args.base, sysd, "hw_runs", dirname)
        paths = sorted(glob.glob(os.path.join(run_dir, "run*.json")))
        if args.runs:
            paths = paths[:args.runs]
        print(f"\n=== {sysd} ({nelec[0]+nelec[1]}e{norb}o)  ref=CASCI(self)={ref:.6f} Ha "
              f"  runs={len(paths)} ===", flush=True)

        for i, p in enumerate(paths):
            counts = load_run_counts(p)
            t0 = time.time()
            _, h_std = run_standard(hcore, eri, counts, norb, nelec, e_nuc,
                                    args.iters, args.samples, args.batches)
            _, h_en = run_en(hcore, eri, counts, norb, nelec, e_nuc,
                             args.iters, args.badd, args.kinit)
            err_std = np.abs(pad_to(h_std, args.iters) - ref)
            err_en = np.abs(pad_to(h_en, args.iters) - ref)
            for it in range(args.iters):
                rows.append(dict(system=sysd, run=i + 1, iter=it + 1,
                                 err_standard=err_std[it], err_en=err_en[it]))
            print(f"  run{i+1:02d}: std final err {err_std[-1]:.5e} | "
                  f"en final err {err_en[-1]:.5e}  [{time.time()-t0:.1f}s]", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"sqd_convergence_runs{tag}.csv"), index=False)
    print(f"\nwrote {os.path.join(out_dir, f'sqd_convergence_runs{tag}.csv')}")

    # 図は plot_sqd_figs に一元化（CSV から描画）
    plot_convergence(out_dir, tag)


if __name__ == "__main__":
    main()
