"""Standard SQD と AS-SQD(EN) を、繰り返し実機測定(10 Run)の各カウントに対して
実行し、最終エネルギー誤差の Run 間バラつきを比較するスクリプト。

sqd_pb_en.py の確立済みロジック(FermionicAS_SQD と qiskit-addon-sqd の
diagonalize_fermionic_hamiltonian)をそのまま使い、入力カウントだけを
extracted/<sys>/hw_runs/orig_shot<N>[_<tag>]/run*.json に差し替える。

参照エネルギーは ham_cache の自己無撞着 CASCI(casci_s0)。これにより、活性空間が
元回路と一致しない問題に依存せず、Standard と EN を同一 Hamiltonian 上で公平に
比較できる(絶対誤差は casci_s0 基準)。ソルバの乱数シードは固定するので、Run 間の
差は測定データ由来のみになる。

使い方:
  python compare_sqd_runs.py --systems pb2,pb4 --shots 10000 \
      --iters 20 --badd 600 --kinit 100 --samples 100 --batches 5 [--runs N] [--tag rerun]
"""
import os
import json
import glob
import time
import argparse

import numpy as np
import pandas as pd

from qiskit_addon_sqd.counts import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, SCIResult
from sqd_pb_en import FermionicAS_SQD, load_hamiltonian
from plot_sqd_figs import plot_compare, plot_compare_energy

HERE = os.path.dirname(os.path.abspath(__file__))


def load_run_counts(path):
    """run*.json を {整数キー: 出現数} に変換して返す。"""
    with open(path) as f:
        raw = json.load(f)
    counts = {}
    for bitstr, c in raw.items():
        key = int(bitstr.replace(" ", ""), 2)
        counts[key] = counts.get(key, 0) + int(c)
    return counts


def run_standard(hcore, eri, counts_int, norb, nelec, e_nuc,
                 iters, samples, batches, seed=42):
    """Standard SQD(qiskit-addon-sqd)を1ラン分実行し、最終エネルギーと履歴を返す。"""
    bit_array = BitArray.from_counts(counts_int, num_bits=2 * norb)
    hist = []

    def cb(results):
        hist.append(results[0].energy + e_nuc)

    diagonalize_fermionic_hamiltonian(
        hcore, eri, bit_array,
        samples_per_batch=samples, norb=norb, nelec=nelec,
        num_batches=batches, energy_tol=1e-5, occupancies_tol=1e-5,
        max_iterations=iters, callback=cb, seed=np.random.default_rng(seed),
    )
    return hist[-1], hist


def run_en(hcore, eri, counts_int, norb, nelec, e_nuc,
           iters, badd, kinit, tau=1e-3, seed=42):
    """AS-SQD(EN)を1ラン分実行し、最終エネルギーと履歴を返す。"""
    solver = FermionicAS_SQD(
        hcore, eri, counts_int,
        K_init=kinit, num_orbs=norb, nelec_tuple=nelec,
        method="en", apply_pt2=False, e_nuc=e_nuc,
    )
    solver.run(iterations=iters, B_add=badd, tau_dom=tau, rng_seed=seed, verbose=False)
    return solver.history[-1], solver.history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="pb2,pb4")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--tag", default="")           # run ディレクトリ接尾辞(measure_qpy.py と合わせる)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--badd", type=int, default=600)
    ap.add_argument("--kinit", type=int, default=100)
    ap.add_argument("--samples", type=int, default=100)   # Standard SQD: samples_per_batch
    ap.add_argument("--batches", type=int, default=5)      # Standard SQD: num_batches
    ap.add_argument("--runs", type=int, default=0)         # 0 = 全ラン
    ap.add_argument("--base", default="../extracted")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    cache_dir = os.path.join(HERE, "ham_cache")
    out_dir = os.path.join(HERE, "results_en", "hw_variability", "orig_pittsburgh", "sqd_compare")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    summary = []
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
            e_std, _ = run_standard(hcore, eri, counts, norb, nelec, e_nuc,
                                    args.iters, args.samples, args.batches)
            e_en, _ = run_en(hcore, eri, counts, norb, nelec, e_nuc,
                             args.iters, args.badd, args.kinit)
            err_std = abs(e_std - ref)
            err_en = abs(e_en - ref)
            dt = time.time() - t0
            print(f"  run{i+1:02d}: standard E={e_std:.5f} (err {err_std:.5f}) | "
                  f"en E={e_en:.5f} (err {err_en:.5f})  [{dt:.1f}s]", flush=True)
            rows.append(dict(system=sysd, run=i + 1,
                             E_standard=e_std, err_standard=err_std,
                             E_en=e_en, err_en=err_en))

        # 系ごとの集計
        sub = [r for r in rows if r["system"] == sysd]
        for meth in ("standard", "en"):
            errs = np.array([r[f"err_{meth}"] for r in sub])
            summary.append(dict(system=sysd, method=meth, R=len(sub),
                                err_mean=errs.mean(), err_std=errs.std(),
                                err_min=errs.min(), err_max=errs.max()))
            print(f"  [{sysd} {meth:8s}] err mean={errs.mean():.5f} std={errs.std():.5f} "
                  f"min={errs.min():.5f} max={errs.max():.5f}", flush=True)

    df = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary)
    tag = f"_{args.tag}" if args.tag else ""
    df.to_csv(os.path.join(out_dir, f"sqd_compare_runs{tag}.csv"), index=False)
    df_sum.to_csv(os.path.join(out_dir, f"sqd_compare_summary{tag}.csv"), index=False)
    print(f"\nwrote {os.path.join(out_dir, f'sqd_compare_runs{tag}.csv')}")
    print(f"wrote {os.path.join(out_dir, f'sqd_compare_summary{tag}.csv')}")

    # 図は plot_sqd_figs に一元化（CSV から描画）
    plot_compare(out_dir, tag)
    plot_compare_energy(out_dir, tag)


if __name__ == "__main__":
    main()
