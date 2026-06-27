"""Epstein-Nesbet AS-SQD for the perovskite systems pb2 / pb3 / pb4.

Adapted from ``development/source/en/n2_boston.py``. The acquisition-function
AS-SQD core (``FermionicAS_SQD``) is reused almost verbatim; only the data
plumbing changes:

  * active-space integrals (hcore, eri, e_nuc) are loaded from a cached ``.npz``
    produced by ``build_pb_hamiltonian.py`` (instead of being built inline);
  * device counts are read from the pb shot folders
    (``extracted/<sys>/shot-XXX/sqd_counts_*.json``);
  * the reference (exact) energy is the README CASCI S0 value.

Acquisition functions (Epstein-Nesbet decomposition, Slater-Condon exact MEs):
    en       : |nu_k|^2 / |E_S - H_kk|   (full Epstein-Nesbet)
    coupling : |nu_k|^2
    denom    : 1 / |E_S - H_kk|
    diag     : -H_kk
plus baselines ``standard`` (qiskit-addon-sqd) and ``random``, and the
``en+pt2`` variant (en subspace + signed 2nd-order PT2 correction on the
reported energy).

Usage:
    python sqd_pb_en.py <BASE> <sys> [--shot 10k] [--iters 20] [--badd 600]
                        [--kinit 100] [--methods en,en+pt2,standard,random]
"""
import os
import sys
import glob
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit_addon_sqd.counts import BitArray
from qiskit_addon_sqd.fermion import (
    diagonalize_fermionic_hamiltonian,
    solve_fermion,
    SCIResult,
)

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 140

HARTREE2EV = 27.211386245988

# README CASCI references (S0 ground, S1 first singlet excited), Hartree.
REF = {
    "pb2": {"scf": -4317.376693057786, "s0": -4317.38043865596, "s1": -4317.18705227103},
    "pb3": {"scf": -6279.87232750353, "s0": -6279.87843005372, "s1": -6279.68734502148},
    "pb4": {"scf": -7850.022480822798, "s0": -7850.03052542641, "s1": -7849.83681307442},
}


# ======================================================================
# AS-SQD core (ported from n2_boston.py FermionicAS_SQD)
# ======================================================================
class FermionicAS_SQD:
    """Fermionic AS-SQD (closed-shell singlet) with selectable acquisition function.

    hcore, eri : active-space 1-/2-electron integrals (chemist's notation).
    counts_dict : dict[int, int] device counts, integer key = alpha|beta bitstring.
    K_init : number of top-frequency (particle-number-filtered) configs in S0.
    num_orbs : active orbital count L.
    nelec_tuple : (Na, Nb).
    method : 'random' | 'en' | 'coupling' | 'denom' | 'diag'.
    e_nuc : core/nuclear energy added to reported energy.
    apply_pt2 : add E_PT2 to the reported energy each iteration.
    """

    def __init__(self, hcore, eri, counts_dict, K_init, num_orbs, nelec_tuple,
                 method, e_nuc, apply_pt2: bool = False):
        self.hcore = np.array(hcore, dtype=float)
        self.eri = np.array(eri, dtype=float)
        self.num_orbs = num_orbs
        self.method = method
        self.apply_pt2 = apply_pt2
        self.e_nuc = e_nuc
        self.nelec_alpha = nelec_tuple[0]
        self.nelec_beta = nelec_tuple[1]
        self.pt2_history = []

        n = num_orbs
        self.h_diag = np.diag(self.hcore).copy()
        self.J = np.zeros((n, n))   # (pp|qq)
        self.K = np.zeros((n, n))   # (pq|qp)
        for p in range(n):
            for q in range(n):
                self.J[p, q] = self.eri[p, p, q, q]
                self.K[p, q] = self.eri[p, q, q, p]
        self._diag_cache = {}

        filtered = {}
        for bit_int, count in counts_dict.items():
            occ_a, occ_b = self._get_occs(bit_int)
            if len(occ_a) == self.nelec_alpha and len(occ_b) == self.nelec_beta:
                filtered[bit_int] = count
        top_k = sorted(filtered.items(), key=lambda x: -x[1])[:K_init]
        self.subspace = {k for k, _ in top_k}
        self.history = []

    # ----------------- bit representation -----------------
    def _int_to_bool_array(self, bit_int):
        arr = np.zeros(2 * self.num_orbs, dtype=bool)
        for p in range(self.num_orbs):
            if (bit_int >> p) & 1:
                arr[p] = True
            if (bit_int >> (self.num_orbs + p)) & 1:
                arr[self.num_orbs + p] = True
        return arr

    def _bit_reverse(self, s):
        out = 0
        for i in range(self.num_orbs):
            if s & (1 << i):
                out |= (1 << (self.num_orbs - 1 - i))
        return out

    def _qiskit_pair_to_bit_int(self, ci_str_a, ci_str_b):
        alpha_lsb = self._bit_reverse(int(ci_str_a))
        beta_lsb = self._bit_reverse(int(ci_str_b))
        return alpha_lsb | (beta_lsb << self.num_orbs)

    def _get_occs(self, bit_int):
        alpha_int = bit_int & ((1 << self.num_orbs) - 1)
        beta_int = (bit_int >> self.num_orbs) & ((1 << self.num_orbs) - 1)
        occ_a = [i for i in range(self.num_orbs) if (alpha_int >> i) & 1]
        occ_b = [i for i in range(self.num_orbs) if (beta_int >> i) & 1]
        return occ_a, occ_b

    # ----------------- phase factors -----------------
    @staticmethod
    def _phase_single_sector(sec, a, r):
        n = bin(sec & ((1 << a) - 1)).count("1")
        sec_after = sec & ~(1 << a)
        n += bin(sec_after & ((1 << r) - 1)).count("1")
        return -1 if (n & 1) else 1

    @staticmethod
    def _phase_double_same_sector(sec, a, b, r, s):
        n = bin(sec & ((1 << a) - 1)).count("1")
        sec = sec & ~(1 << a)
        n += bin(sec & ((1 << b) - 1)).count("1")
        sec = sec & ~(1 << b)
        n += bin(sec & ((1 << r) - 1)).count("1")
        sec = sec | (1 << r)
        n += bin(sec & ((1 << s) - 1)).count("1")
        return -1 if (n & 1) else 1

    # ----------------- diagonal energy (Slater rule D0) -----------------
    def _diagonal_energy(self, bit_int):
        if bit_int in self._diag_cache:
            return self._diag_cache[bit_int]
        occ_a, occ_b = self._get_occs(bit_int)
        e = self.h_diag[occ_a].sum() + self.h_diag[occ_b].sum()
        for i in range(len(occ_a)):
            for j in range(i + 1, len(occ_a)):
                p, q = occ_a[i], occ_a[j]
                e += self.J[p, q] - self.K[p, q]
        for i in range(len(occ_b)):
            for j in range(i + 1, len(occ_b)):
                p, q = occ_b[i], occ_b[j]
                e += self.J[p, q] - self.K[p, q]
        for p in occ_a:
            for q in occ_b:
                e += self.J[p, q]
        self._diag_cache[bit_int] = e
        return e

    # ----------------- candidate generation + nu_k aggregation -----------------
    def _generate_candidates_and_nu(self, dom_basis, dom_coeffs):
        nu_dict = {}
        L = self.num_orbs
        mask_lo = (1 << L) - 1

        for j_int, c_j in zip(dom_basis, dom_coeffs):
            alpha_j = j_int & mask_lo
            beta_j = (j_int >> L) & mask_lo
            occ_a = [p for p in range(L) if (alpha_j >> p) & 1]
            occ_b = [p for p in range(L) if (beta_j >> p) & 1]
            vir_a = [p for p in range(L) if not ((alpha_j >> p) & 1)]
            vir_b = [p for p in range(L) if not ((beta_j >> p) & 1)]

            # alpha single
            for a in occ_a:
                for r in vir_a:
                    h_eff = self.hcore[a, r]
                    for p in occ_a:
                        if p != a:
                            h_eff += self.eri[a, r, p, p] - self.eri[a, p, p, r]
                    for p in occ_b:
                        h_eff += self.eri[a, r, p, p]
                    phase = self._phase_single_sector(alpha_j, a, r)
                    k_alpha = (alpha_j & ~(1 << a)) | (1 << r)
                    k_int = k_alpha | (beta_j << L)
                    nu_dict[k_int] = nu_dict.get(k_int, 0.0) + c_j * phase * h_eff

            # beta single
            for a in occ_b:
                for r in vir_b:
                    h_eff = self.hcore[a, r]
                    for p in occ_b:
                        if p != a:
                            h_eff += self.eri[a, r, p, p] - self.eri[a, p, p, r]
                    for p in occ_a:
                        h_eff += self.eri[a, r, p, p]
                    phase = self._phase_single_sector(beta_j, a, r)
                    k_beta = (beta_j & ~(1 << a)) | (1 << r)
                    k_int = alpha_j | (k_beta << L)
                    nu_dict[k_int] = nu_dict.get(k_int, 0.0) + c_j * phase * h_eff

            # opposite-spin double: alpha a->r, beta b->s
            for a in occ_a:
                for r in vir_a:
                    phase_a = self._phase_single_sector(alpha_j, a, r)
                    k_alpha = (alpha_j & ~(1 << a)) | (1 << r)
                    for b in occ_b:
                        for s in vir_b:
                            phase_b = self._phase_single_sector(beta_j, b, s)
                            mel = phase_a * phase_b * self.eri[a, r, b, s]
                            if mel == 0.0:
                                continue
                            k_beta = (beta_j & ~(1 << b)) | (1 << s)
                            k_int = k_alpha | (k_beta << L)
                            nu_dict[k_int] = nu_dict.get(k_int, 0.0) + c_j * mel

            # same-spin alpha-alpha double
            for i_a in range(len(occ_a)):
                a = occ_a[i_a]
                for i_b in range(i_a + 1, len(occ_a)):
                    b = occ_a[i_b]
                    for i_r in range(len(vir_a)):
                        r = vir_a[i_r]
                        for i_s in range(i_r + 1, len(vir_a)):
                            s = vir_a[i_s]
                            mel = self.eri[a, r, b, s] - self.eri[a, s, b, r]
                            if mel == 0.0:
                                continue
                            phase = self._phase_double_same_sector(alpha_j, a, b, r, s)
                            k_alpha = (alpha_j & ~(1 << a) & ~(1 << b)) | (1 << r) | (1 << s)
                            k_int = k_alpha | (beta_j << L)
                            nu_dict[k_int] = nu_dict.get(k_int, 0.0) + c_j * phase * mel

            # same-spin beta-beta double
            for i_a in range(len(occ_b)):
                a = occ_b[i_a]
                for i_b in range(i_a + 1, len(occ_b)):
                    b = occ_b[i_b]
                    for i_r in range(len(vir_b)):
                        r = vir_b[i_r]
                        for i_s in range(i_r + 1, len(vir_b)):
                            s = vir_b[i_s]
                            mel = self.eri[a, r, b, s] - self.eri[a, s, b, r]
                            if mel == 0.0:
                                continue
                            phase = self._phase_double_same_sector(beta_j, a, b, r, s)
                            k_beta = (beta_j & ~(1 << a) & ~(1 << b)) | (1 << r) | (1 << s)
                            k_int = alpha_j | (k_beta << L)
                            nu_dict[k_int] = nu_dict.get(k_int, 0.0) + c_j * phase * mel

        for k in self.subspace:
            nu_dict.pop(k, None)
        return list(nu_dict.keys()), nu_dict

    # ----------------- main loop -----------------
    def run(self, iterations, B_add, tau_dom, eps_denom=1e-6, rng_seed=42, verbose=True):
        rng = np.random.default_rng(rng_seed)
        need_nu = self.method in ("en", "coupling") or self.apply_pt2
        need_diag = self.method in ("en", "denom", "diag") or self.apply_pt2

        for it in range(iterations):
            basis_list = list(self.subspace)
            bool_matrix = np.array([self._int_to_bool_array(b) for b in basis_list])
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

            need_candidates = (it < iterations - 1) or self.apply_pt2
            if need_candidates:
                cand_list, nu_dict = self._generate_candidates_and_nu(dom_basis, dom_coeffs)
            else:
                cand_list, nu_dict = [], {}

            E_pt2 = 0.0
            if self.apply_pt2 and cand_list:
                for k in cand_list:
                    H_kk = self._diagonal_energy(k)
                    denom_s = energy - H_kk
                    if abs(denom_s) > eps_denom:
                        E_pt2 += abs(nu_dict[k]) ** 2 / denom_s
            self.pt2_history.append(E_pt2)
            E_report = energy + E_pt2
            self.history.append(E_report + self.e_nuc)
            if verbose:
                log_pt2 = f" + E_PT2={E_pt2:+.5f}" if self.apply_pt2 else ""
                print(
                    f"AS-SQD [{self.method:8s}{'*' if self.apply_pt2 else ' '}] "
                    f"Iter {it+1}: |S|={len(self.subspace):5d}, |prod|={amps.size:6d}, "
                    f"E_SQD={energy + self.e_nuc:.5f}{log_pt2} "
                    f"=> {self.history[-1]:.5f} Ha", flush=True
                )

            if it == iterations - 1:
                break
            if not cand_list:
                continue

            if self.method == "random":
                if len(cand_list) > B_add:
                    pick = rng.choice(len(cand_list), B_add, replace=False)
                    top_cands = [cand_list[i] for i in pick]
                else:
                    top_cands = cand_list
            else:
                scores = np.empty(len(cand_list))
                for idx, k in enumerate(cand_list):
                    nu_k = nu_dict[k] if need_nu else 0.0
                    H_kk = self._diagonal_energy(k) if need_diag else 0.0
                    denom = max(abs(energy - H_kk), eps_denom) if need_diag else 1.0
                    if self.method == "en":
                        scores[idx] = abs(nu_k) ** 2 / denom
                    elif self.method == "coupling":
                        scores[idx] = abs(nu_k) ** 2
                    elif self.method == "denom":
                        scores[idx] = 1.0 / denom
                    elif self.method == "diag":
                        scores[idx] = -H_kk
                    else:
                        raise ValueError(f"Unknown method: {self.method}")
                k_top = min(B_add, len(scores) - 1)
                top_idx = np.argpartition(-scores, k_top)[:B_add]
                top_cands = [cand_list[i] for i in top_idx]

            for k in top_cands:
                self.subspace.add(k)


# ======================================================================
# pb data plumbing
# ======================================================================
def load_hamiltonian(cache_dir, sysd):
    path = os.path.join(cache_dir, f"{sysd}_ham.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"  python build_pb_hamiltonian.py <BASE> {sysd} [--df]"
        )
    d = np.load(path)
    hcore = d["hcore"]
    eri = d["eri"]
    e_nuc = float(d["e_nuc"])
    nelec_act = int(d["nelec_act"])
    norb_act = int(d["norb_act"])
    # self-consistent CASCI S0 for these integrals (correct AS-SQD reference)
    casci_s0 = float(d["casci_s0"]) if "casci_s0" in d.files else None
    na = nelec_act // 2
    nelec = (na, nelec_act - na)
    return hcore, eri, e_nuc, norb_act, nelec, casci_s0


def load_counts(base, sysd, shot):
    pattern = f"{base}/{sysd}/shot-{shot}/*.json"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No counts json at {pattern}")
    with open(matches[0]) as f:
        raw = json.load(f)
    counts_int = {}
    for bitstr, count in raw.items():
        key = int(bitstr.replace(" ", ""), 2)
        counts_int[key] = counts_int.get(key, 0) + int(count)
    return counts_int, matches[0]


STYLE_MAP = {
    "standard": {"color": "#1f77b4", "marker": "o", "ls": "--"},
    "random":   {"color": "#7f7f7f", "marker": "v", "ls": "--"},
    "en":       {"color": "#8B0000", "marker": "X", "ls": "-"},
    "en+pt2":   {"color": "#d62728", "marker": "s", "ls": "-"},
    "coupling": {"color": "#ff7f0e", "marker": "D", "ls": "-"},
    "denom":    {"color": "#2ca02c", "marker": "^", "ls": "-"},
    "diag":     {"color": "#9467bd", "marker": "P", "ls": "-"},
}

METHOD_SPEC = {
    "standard": None,
    "random":   {"method": "random",   "apply_pt2": False},
    "en":       {"method": "en",       "apply_pt2": False},
    "en+pt2":   {"method": "en",       "apply_pt2": True},
    "coupling": {"method": "coupling", "apply_pt2": False},
    "denom":    {"method": "denom",    "apply_pt2": False},
    "diag":     {"method": "diag",     "apply_pt2": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("sys", choices=["pb2", "pb3", "pb4"])
    ap.add_argument("--shot", default="10k")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--badd", type=int, default=600)
    ap.add_argument("--kinit", type=int, default=100)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--methods", default="standard,random,en,en+pt2,coupling,denom,diag")
    ap.add_argument("--std-batches", type=int, default=5)
    ap.add_argument("--std-samples", type=int, default=100)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, "ham_cache")
    out_dir = os.path.join(here, "results_en", args.sys)
    os.makedirs(out_dir, exist_ok=True)

    hcore, eri, e_nuc, norb, nelec, casci_s0 = load_hamiltonian(cache_dir, args.sys)
    counts_int, counts_path = load_counts(args.base, args.sys, args.shot)
    # use the self-consistent CASCI of these integrals as the exact reference;
    # fall back to the README value if the cache predates the casci_s0 field.
    exact_energy = casci_s0 if casci_s0 is not None else REF[args.sys]["s0"]
    ref_src = "CASCI(self)" if casci_s0 is not None else "README"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"=== {args.sys}  ({nelec[0]+nelec[1]}e{norb}o)  shot={args.shot} ===")
    print(f"  counts file : {os.path.basename(counts_path)}")
    print(f"  unique keys : {len(counts_int):,}  total shots: {sum(counts_int.values()):,}")
    print(f"  e_nuc(core) : {e_nuc:.6f}   exact ref ({ref_src}): {exact_energy:.6f}")
    print(f"  methods     : {methods}\n", flush=True)

    bit_array = BitArray.from_counts(counts_int, num_bits=2 * norb)
    all_histories = {}
    pt2_terms = {}

    # ---- standard SQD ----
    if "standard" in methods:
        print("--- Standard SQD (qiskit-addon-sqd) ---", flush=True)
        std_history = []

        def std_callback(results: list[SCIResult]):
            std_history.append(results[0].energy + e_nuc)
            print(f"  Iter {len(std_history)}: E={std_history[-1]:.5f} Ha", flush=True)

        diagonalize_fermionic_hamiltonian(
            hcore, eri, bit_array,
            samples_per_batch=args.std_samples, norb=norb, nelec=nelec,
            num_batches=args.std_batches, energy_tol=1e-5, occupancies_tol=1e-5,
            max_iterations=args.iters, callback=std_callback,
            seed=np.random.default_rng(42),
        )
        all_histories["standard"] = std_history

    # ---- AS-SQD acquisition-function variants ----
    for label in methods:
        if label == "standard":
            continue
        cfg = METHOD_SPEC[label]
        print(f"\n--- AS-SQD '{label}' (method={cfg['method']}, pt2={cfg['apply_pt2']}) ---",
              flush=True)
        t0 = time.time()
        solver = FermionicAS_SQD(
            hcore, eri, counts_int,
            K_init=args.kinit, num_orbs=norb, nelec_tuple=nelec,
            method=cfg["method"], apply_pt2=cfg["apply_pt2"], e_nuc=e_nuc,
        )
        solver.run(iterations=args.iters, B_add=args.badd, tau_dom=args.tau)
        all_histories[label] = solver.history
        pt2_terms[label] = solver.pt2_history
        print(f"  ({label}) elapsed: {time.time()-t0:.1f}s", flush=True)

    # ---- save CSV ----
    rows = []
    for m, hist in all_histories.items():
        pt2 = pt2_terms.get(m, [0.0] * len(hist))
        for i, e in enumerate(hist):
            rows.append({
                "system": args.sys, "shot": args.shot, "method": m, "iter": i + 1,
                "energy": e, "E_pt2": pt2[i] if i < len(pt2) else 0.0,
                "abs_err": abs(e - exact_energy),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"ASSQD_{args.sys}_shot{args.shot}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults -> {csv_path}", flush=True)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    max_iter = 1
    for m, hist in all_histories.items():
        errs = [abs(e - exact_energy) for e in hist]
        x = list(range(1, len(errs) + 1))
        max_iter = max(max_iter, len(errs))
        s = STYLE_MAP.get(m, {})
        ax.plot(x, errs, label=m, linewidth=2, color=s.get("color"),
                marker=s.get("marker"), linestyle=s.get("ls", "-"), markersize=7)
    ax.axhline(y=0.0016, color="#BF5700", linestyle=":", label="Chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute Energy Error (Hartree)")
    ax.set_title(f"{args.sys} ({nelec[0]+nelec[1]}e{norb}o, shot={args.shot}) -- EN AS-SQD")
    ax.set_xticks(list(range(1, max_iter + 1)))
    ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.95)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"ASSQD_{args.sys}_shot{args.shot}.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Plot    -> {fig_path}", flush=True)

    print("\n=== Final absolute errors (Ha) ===")
    for m, hist in all_histories.items():
        print(f"  {m:9s}: {abs(hist[-1] - exact_energy):.5f}")


if __name__ == "__main__":
    main()
