#!/usr/bin/env python3
"""Pb 系 AS-SQD の獲得関数アブレーション / パラメータ分析

保存済み hw_runs ビットストリングを再利用し、獲得関数 A/B/C のアブレーションと
ハイパーパラメータ感度を計算する。各 (系, run, 実験条件) はデフォルトで逐次実行する。

Usage:
  uv run python additional-calc-for-paper/ablation_pb_as_sqd.py \\
      --systems pb2,pb3,pb4 --runs 10 --iters 20

オプション ``--workers N`` (N>1) を付けると ProcessPool で並列実行できるが、
本リポジトリで報告した結果は上記の逐次実行によるものである。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

# BLAS 過購読を防ぐ（ワーカー側でも再設定）
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ANALYSIS = os.path.join(REPO, "analysis")
sys.path.insert(0, ANALYSIS)

from qiskit_addon_sqd.counts import BitArray  # noqa: E402
from qiskit_addon_sqd.fermion import (  # noqa: E402
    diagonalize_fermionic_hamiltonian,
    solve_fermion,
    SCIResult,
)
from sqd_pb_en import FermionicAS_SQD, load_hamiltonian  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 140

CHEM_ACC = 0.0016
BASE_B_ADD = 600
BASE_GAMMA = 1.0
BASE_TAU = 1e-3
SYS_WEIGHT = {"pb4": 3, "pb3": 2, "pb2": 1}


# ======================================================================
# AS-SQD ablation core
# ======================================================================
class FermionicAS_SQD_Ablation(FermionicAS_SQD):
    def __init__(
        self,
        hcore,
        eri,
        counts_dict,
        K_init,
        num_orbs,
        nelec_tuple,
        e_nuc,
        score_type: str = "ABC",
    ):
        super().__init__(
            hcore,
            eri,
            counts_dict,
            K_init,
            num_orbs,
            nelec_tuple,
            method="en",
            e_nuc=e_nuc,
            apply_pt2=False,
        )
        self.score_type = score_type
        self.counts_dict = counts_dict
        self.total_shots = sum(counts_dict.values()) if counts_dict else 1

    def _generate_candidates_only(self, dom_basis):
        """行列要素なしで singles+doubles 候補だけ列挙 (B/C 系の高速パス)。"""
        cands = set()
        L = self.num_orbs
        mask_lo = (1 << L) - 1
        for j_int in dom_basis:
            alpha_j = j_int & mask_lo
            beta_j = (j_int >> L) & mask_lo
            occ_a = [p for p in range(L) if (alpha_j >> p) & 1]
            occ_b = [p for p in range(L) if (beta_j >> p) & 1]
            vir_a = [p for p in range(L) if not ((alpha_j >> p) & 1)]
            vir_b = [p for p in range(L) if not ((beta_j >> p) & 1)]
            for a in occ_a:
                for r in vir_a:
                    cands.add(((alpha_j & ~(1 << a)) | (1 << r)) | (beta_j << L))
            for a in occ_b:
                for r in vir_b:
                    cands.add(alpha_j | (((beta_j & ~(1 << a)) | (1 << r)) << L))
            for a in occ_a:
                for r in vir_a:
                    k_alpha = (alpha_j & ~(1 << a)) | (1 << r)
                    for b in occ_b:
                        for s in vir_b:
                            cands.add(k_alpha | (((beta_j & ~(1 << b)) | (1 << s)) << L))
            for i_a in range(len(occ_a)):
                a = occ_a[i_a]
                for i_b in range(i_a + 1, len(occ_a)):
                    b = occ_a[i_b]
                    for i_r in range(len(vir_a)):
                        r = vir_a[i_r]
                        for i_s in range(i_r + 1, len(vir_a)):
                            s = vir_a[i_s]
                            k_alpha = (alpha_j & ~(1 << a) & ~(1 << b)) | (1 << r) | (1 << s)
                            cands.add(k_alpha | (beta_j << L))
            for i_a in range(len(occ_b)):
                a = occ_b[i_a]
                for i_b in range(i_a + 1, len(occ_b)):
                    b = occ_b[i_b]
                    for i_r in range(len(vir_b)):
                        r = vir_b[i_r]
                        for i_s in range(i_r + 1, len(vir_b)):
                            s = vir_b[i_s]
                            k_beta = (beta_j & ~(1 << a) & ~(1 << b)) | (1 << r) | (1 << s)
                            cands.add(alpha_j | (k_beta << L))
        cands -= self.subspace
        return list(cands)

    def run(self, iterations, B_add, tau_dom, gamma=1.0, eps_denom=1e-6, verbose=False):
        t0 = time.time()
        st = self.score_type
        inv_shots = 1.0 / self.total_shots
        # スコアに必要な量だけ計算 (A系=ν, B系=Hkk, C系=p_meas)
        need_nu = st in ("A", "AB", "AC", "ABC")
        need_diag = st in ("A", "B", "AB", "BC", "AC", "ABC")
        need_c = st in ("C", "BC", "AC", "ABC")

        for it in range(iterations):
            basis_list = list(self.subspace)
            bool_matrix = np.array(
                [self._int_to_bool_array(b) for b in basis_list], dtype=bool
            )
            energy, sci_state, _, _ = solve_fermion(
                bool_matrix, self.hcore, self.eri, open_shell=False, spin_sq=0
            )

            amps = sci_state.amplitudes
            strs_a_q = sci_state.ci_strs_a
            strs_b_q = sci_state.ci_strs_b
            probs = amps ** 2
            mask = probs > tau_dom
            ia_arr, ib_arr = np.where(mask)
            if len(ia_arr) == 0:
                ia, ib = np.unravel_index(int(np.argmax(probs)), probs.shape)
                ia_arr, ib_arr = np.array([ia]), np.array([ib])

            dom_basis, dom_coeffs = [], []
            for ia, ib in zip(ia_arr, ib_arr):
                bit_int = self._qiskit_pair_to_bit_int(strs_a_q[ia], strs_b_q[ib])
                dom_basis.append(bit_int)
                dom_coeffs.append(float(amps[ia, ib]))

            self.history.append(float(energy) + self.e_nuc)
            if verbose:
                print(
                    f"  AS-SQD[{st:3s}] Iter {it+1}: |S|={len(self.subspace):5d} "
                    f"E={self.history[-1]:.6f} Ha",
                    flush=True,
                )
            if it == iterations - 1:
                break

            if need_nu:
                cand_list, nu_dict = self._generate_candidates_and_nu(dom_basis, dom_coeffs)
            else:
                cand_list = self._generate_candidates_only(dom_basis)
                nu_dict = {}
            if not cand_list:
                continue

            n = len(cand_list)
            if need_c:
                pm = np.fromiter(
                    (self.counts_dict.get(k, 0) * inv_shots for k in cand_list),
                    dtype=np.float64,
                    count=n,
                )
                C = np.log1p(pm + 1e-8)
            if need_diag:
                Hkk = np.fromiter(
                    (self._diagonal_energy(k) for k in cand_list),
                    dtype=np.float64,
                    count=n,
                )
                B = np.exp(-gamma * np.maximum(Hkk - energy, 0.0))
            if need_nu:
                nu2 = np.fromiter(
                    (abs(nu_dict[k]) ** 2 for k in cand_list),
                    dtype=np.float64,
                    count=n,
                )
                denom = np.maximum(np.abs(energy - Hkk), eps_denom)
                A = np.maximum(nu2 / denom, 1e-12)

            if st == "A":
                scores = A
            elif st == "B":
                scores = B
            elif st == "C":
                scores = C
            elif st == "AB":
                scores = A * B
            elif st == "BC":
                scores = B * C
            elif st == "AC":
                scores = A * C
            elif st == "ABC":
                scores = A * B * C
            else:
                raise ValueError(f"Unknown score_type: {st}")

            k_top = min(B_add, n - 1)
            top_idx = np.argpartition(-scores, k_top)[:B_add]
            for i in top_idx:
                self.subspace.add(cand_list[int(i)])

        elapsed = time.time() - t0
        while len(self.history) < iterations:
            self.history.append(self.history[-1] if self.history else 0.0)
        return self.history[:iterations], elapsed


def run_standard_sqd(hcore, eri, counts_dict, norb, nelec, e_nuc, iterations, samples, batches):
    bit_array = BitArray.from_counts(counts_dict, num_bits=2 * norb)
    hist = []

    def callback(results: list[SCIResult]):
        hist.append(float(results[0].energy) + e_nuc)

    t0 = time.time()
    diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=samples,
        norb=norb,
        nelec=nelec,
        num_batches=batches,
        energy_tol=1e-5,
        occupancies_tol=1e-5,
        max_iterations=iterations,
        callback=callback,
        seed=np.random.default_rng(42),
    )
    elapsed = time.time() - t0
    while len(hist) < iterations:
        hist.append(hist[-1] if hist else 0.0)
    return hist[:iterations], elapsed


def load_run_counts(path: str) -> dict[int, int]:
    with open(path) as f:
        raw = json.load(f)
    out: dict[int, int] = {}
    for bitstr, count in raw.items():
        key = int(bitstr.replace(" ", ""), 2)
        out[key] = out.get(key, 0) + int(count)
    return out


def build_experiments():
    exps = [{"method": "Standard_SQD"}]
    for st in ("A", "B", "C", "AB", "BC", "AC", "ABC"):
        exps.append(
            {
                "method": "AS-SQD",
                "score_type": st,
                "B_add": BASE_B_ADD,
                "gamma": BASE_GAMMA,
                "tau_dom": BASE_TAU,
            }
        )
    for b in (200, 1200):
        exps.append(
            {
                "method": "AS-SQD",
                "score_type": "ABC",
                "B_add": b,
                "gamma": BASE_GAMMA,
                "tau_dom": BASE_TAU,
            }
        )
    for g in (0.1, 5.0):
        exps.append(
            {
                "method": "AS-SQD",
                "score_type": "ABC",
                "B_add": BASE_B_ADD,
                "gamma": g,
                "tau_dom": BASE_TAU,
            }
        )
    for t in (1e-4, 1e-2):
        exps.append(
            {
                "method": "AS-SQD",
                "score_type": "ABC",
                "B_add": BASE_B_ADD,
                "gamma": BASE_GAMMA,
                "tau_dom": t,
            }
        )
    return exps


def get_exp_name(exp: dict) -> str:
    if exp["method"] == "Standard_SQD":
        return "Standard_SQD"
    if exp["score_type"] != "ABC":
        return f"Score_{exp['score_type']}"
    if exp["B_add"] != BASE_B_ADD:
        return f"B_add_{exp['B_add']}"
    if exp["gamma"] != BASE_GAMMA:
        return f"gamma_{exp['gamma']}"
    if exp["tau_dom"] != BASE_TAU:
        return f"tau_dom_{exp['tau_dom']}"
    return "Proposed_Baseline(ABC)"


def plot_results(df, filter_exps, title, filename, n_iters):
    fig, ax = plt.subplots(figsize=(10, 7))
    x_iters = range(1, n_iters + 1)
    for exp_name in filter_exps:
        sub = df[df["Experiment_Name"] == exp_name]
        if sub.empty:
            continue
        mean_err = [sub[f"Iter_{i}_Error"].mean() for i in x_iters]
        std_err = [sub[f"Iter_{i}_Error"].std() for i in x_iters]
        lw = 3 if "Proposed" in exp_name else 1.5
        mk = "D" if "Proposed" in exp_name else "o"
        ax.errorbar(
            list(x_iters),
            mean_err,
            yerr=std_err,
            marker=mk,
            capsize=4,
            label=exp_name,
            linewidth=lw,
            alpha=0.85,
        )
    ax.axhline(y=CHEM_ACC, color="#BF5700", linestyle="--", linewidth=2, label="Chemical Accuracy")
    ax.set_yscale("log")
    ax.set_xticks(list(x_iters))
    ax.set_xlabel("Iteration Index")
    ax.set_ylabel("Absolute Energy Error (Hartree)")
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def write_outputs(sysd: str, df_raw: pd.DataFrame, n_iters: int):
    out_res = os.path.join(HERE, "results", sysd)
    out_fig = os.path.join(HERE, "figures", sysd)
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)

    raw_path = os.path.join(out_res, "ablation_results_raw.csv")
    df_raw.to_csv(raw_path, index=False)

    summary_cols = ["Time_sec"] + [f"Iter_{i+1}_Error" for i in range(n_iters)]
    df_summary = df_raw.groupby("Experiment_Name")[summary_cols].agg(
        ["mean", "var", "std", "median"]
    )
    sum_path = os.path.join(out_res, "ablation_results_summary.csv")
    df_summary.to_csv(sum_path)

    score_exps = [
        "Standard_SQD",
        "Score_A",
        "Score_B",
        "Score_C",
        "Score_AB",
        "Score_BC",
        "Score_AC",
        "Proposed_Baseline(ABC)",
    ]
    plot_results(
        df_raw,
        score_exps,
        f"{sysd}: Score Function Ablation\nA=EN Energy, B=Penalty, C=Sampling",
        os.path.join(out_fig, "ablation_score_functions.png"),
        n_iters,
    )
    param_exps = [
        "Proposed_Baseline(ABC)",
        "B_add_200",
        "B_add_1200",
        "gamma_0.1",
        "gamma_5.0",
        "tau_dom_0.0001",
        "tau_dom_0.01",
    ]
    plot_results(
        df_raw,
        param_exps,
        f"{sysd}: AS-SQD Hyperparameter Sensitivity (score=ABC)",
        os.path.join(out_fig, "ablation_hyperparameters.png"),
        n_iters,
    )
    print(f"[{sysd}] wrote {raw_path} and figures/", flush=True)


# ======================================================================
# Job runners (default: sequential; optional ProcessPool via --workers)
# ======================================================================
@lru_cache(maxsize=8)
def _cached_ham(cache_dir: str, sysd: str):
    return load_hamiltonian(cache_dir, sysd)


def _job_sort_key(job: dict):
    """実行順: 系の重さ → run 番号 → B_add 大 → tau 小。"""
    exp = job["exp"]
    badd = exp.get("B_add", 0) if isinstance(exp.get("B_add"), (int, float)) else 0
    tau = exp.get("tau_dom", BASE_TAU) if isinstance(exp.get("tau_dom"), (int, float)) else BASE_TAU
    try:
        run_n = int("".join(ch for ch in job["run_id"] if ch.isdigit()) or "0")
    except ValueError:
        run_n = 0
    return (-SYS_WEIGHT.get(job["sysd"], 0), run_n, -badd, tau)


def _init_worker():
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = "1"


def run_one_job(job: dict) -> dict:
    """1 (system, run, experiment) を実行する。"""
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = "1"

    sysd = job["sysd"]
    hcore, eri, e_nuc, norb, nelec, ref = _cached_ham(job["cache_dir"], sysd)
    counts = load_run_counts(job["run_path"])
    exp = job["exp"]
    name = get_exp_name(exp)
    iters = job["iters"]

    if exp["method"] == "Standard_SQD":
        hist, elapsed = run_standard_sqd(
            hcore,
            eri,
            counts,
            norb,
            nelec,
            e_nuc,
            iters,
            job["samples"],
            job["batches"],
        )
    else:
        solver = FermionicAS_SQD_Ablation(
            hcore,
            eri,
            counts,
            K_init=job["kinit"],
            num_orbs=norb,
            nelec_tuple=nelec,
            e_nuc=e_nuc,
            score_type=exp["score_type"],
        )
        hist, elapsed = solver.run(
            iterations=iters,
            B_add=exp["B_add"],
            tau_dom=exp["tau_dom"],
            gamma=exp["gamma"],
            verbose=False,
        )

    errors = [abs(e - ref) for e in hist]
    row = {
        "System": sysd,
        "Run": job["run_id"],
        "Experiment_Name": name,
        "Method": exp["method"],
        "Score_Type": exp.get("score_type", "N/A"),
        "B_add": exp.get("B_add", "N/A"),
        "gamma": exp.get("gamma", "N/A"),
        "tau_dom": exp.get("tau_dom", "N/A"),
        "Time_sec": elapsed,
    }
    for i in range(iters):
        row[f"Iter_{i+1}_Energy"] = hist[i]
        row[f"Iter_{i+1}_Error"] = errors[i]
    return row


def build_jobs(systems, args, experiments) -> list[dict]:
    cache_dir = os.path.join(ANALYSIS, "ham_cache")
    jobs = []
    for sysd in systems:
        run_dir = os.path.join(args.base, sysd, "hw_runs", f"orig_shot{args.shots}")
        run_files = sorted(glob.glob(os.path.join(run_dir, "run*.json")))
        if not run_files:
            raise FileNotFoundError(f"No run*.json under {run_dir}")
        if args.runs > 0:
            run_files = run_files[: args.runs]
        # ham 存在確認
        _cached_ham(cache_dir, sysd)
        for rpath in run_files:
            for exp in experiments:
                jobs.append(
                    {
                        "sysd": sysd,
                        "run_path": rpath,
                        "run_id": os.path.basename(rpath),
                        "exp": exp,
                        "cache_dir": cache_dir,
                        "iters": args.iters,
                        "kinit": args.kinit,
                        "samples": args.samples,
                        "batches": args.batches,
                    }
                )
    jobs.sort(key=_job_sort_key)
    return jobs


def main():
    ap = argparse.ArgumentParser(
        description="Pb AS-SQD acquisition-function ablation using saved bitstrings"
    )
    ap.add_argument("--systems", default="pb2,pb3,pb4")
    ap.add_argument("--base", default=os.path.join(REPO, "extracted"))
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--runs", type=int, default=10, help="0 = all run*.json")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--kinit", type=int, default=100)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="同時実行プロセス数 (default: 1=逐次。2以上で並列)",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-list", action="store_true")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    experiments = build_experiments()

    if args.dry_list:
        print("=== 実験マトリクス ===")
        for exp in experiments:
            print(f"  {get_exp_name(exp):30s}  {exp}")
        jobs = build_jobs(systems, args, experiments)
        print(f"\n総ジョブ数: {len(jobs)}  workers={args.workers}")
        return

    jobs = build_jobs(systems, args, experiments)
    n_jobs = len(jobs)
    workers = max(1, min(args.workers, n_jobs))
    print(
        f"Ablation start: systems={systems}  jobs={n_jobs}  "
        f"workers={workers}  iters={args.iters}",
        flush=True,
    )
    t_wall0 = time.time()
    results: list[dict] = []
    done = 0

    if workers == 1:
        for job in jobs:
            row = run_one_job(job)
            results.append(row)
            done += 1
            if done % 10 == 0 or done == n_jobs:
                elapsed = time.time() - t_wall0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_jobs - done) / rate if rate > 0 else float("nan")
                print(
                    f"  [{done}/{n_jobs}] {row['System']} {row['Run']} "
                    f"{row['Experiment_Name']}  ({row['Time_sec']:.1f}s)  "
                    f"ETA {eta/60:.1f} min",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {ex.submit(run_one_job, job): job for job in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                results.append(row)
                done += 1
                if done % 5 == 0 or done == n_jobs:
                    elapsed = time.time() - t_wall0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_jobs - done) / rate if rate > 0 else float("nan")
                    print(
                        f"  [{done}/{n_jobs}] {row['System']} {row['Run']} "
                        f"{row['Experiment_Name']}  ({row['Time_sec']:.1f}s)  "
                        f"wall {elapsed/60:.1f}m  ETA {eta/60:.1f}m",
                        flush=True,
                    )

    wall = time.time() - t_wall0
    df_all = pd.DataFrame(results)
    for sysd in systems:
        sub = df_all[df_all["System"] == sysd].copy()
        # 安定した順序
        sub = sub.sort_values(["Run", "Experiment_Name"]).reset_index(drop=True)
        write_outputs(sysd, sub, args.iters)

    print(f"\nDone. wall={wall/60:.2f} min  jobs={n_jobs}  workers={workers}", flush=True)


if __name__ == "__main__":
    main()
