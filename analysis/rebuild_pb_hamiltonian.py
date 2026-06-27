"""pb2 / pb3 / pb4 の活性空間ハミルトニアンを「正しい軌道選択」で再構築する。

背景（ポスター Poster1_4.pdf より判明）:
  - 最小CAS は Pb 1 個あたり 2e4o = Pb(6s + 6p) に相当。
      pb2(Pb×2)=4e8o, pb3(Pb×3)=6e12o, pb4(Pb×4)=8e16o。
  - クラスターモデルでは CH3NH3(メチルアンモニウム)に局在した軌道が
    HOMO/LUMO 近傍に現れるが、これらは電子相関に寄与しない「傍観者」。
    そのため単純なエネルギー順 HOMO/LUMO 窓では相関をほぼ拾えない
    （pb2 で実測 +3.14 mHa, ほぼ SCF）。
  - 正しい活性空間は次で構成する:
      占有側 = 価電子帯端 = 正準 HF の上位 N_Pb 占有軌道（Pb-I 相互作用軌道）
      仮想側 = 伝導帯 = AVAS('Pb 6p') が返す Pb 6p 反結合軌道（3 × N_Pb 本）
    これにより CH3NH3 局在軌道を自動的に除外できる。

検証（pb2）: 上記構成の DF-CASCI S0 = -4317.381131,
            README(-4317.380439) との差 -0.69 mHa（化学精度内）。

注意:
  - S0（基底状態エネルギー）は化学精度内で再現するが、S1（バンドギャップ）の
    厳密再現には軌道最適化(SA-CASSCF)が必要（この系では計算量が非現実的）。
    本スクリプトは SQD のエネルギー比較用に S0 基準のハミルトニアンを供給する。
  - 積分は DF（密度フィッティング）。補助基底を変えても CASCI は不変であることを
    確認済み（DF 誤差 <0.001 mHa）。

使い方:
  uv run rebuild_pb_hamiltonian.py <BASE> <sys>            # 例: ../extracted pb2
  uv run rebuild_pb_hamiltonian.py <BASE> <sys> --s1       # S1 も計算（任意）
  uv run rebuild_pb_hamiltonian.py <BASE> <sys> --out <path.npz>
"""
import os, sys, time
import numpy as np
import pyscf.scf, pyscf.mcscf, pyscf.ao2mo
from pyscf.mcscf import avas

# (活性電子数, 活性軌道数) と Pb 個数。占有=N_Pb本, 仮想=3*N_Pb本。
SPEC = {"pb2": (4, 8, 2), "pb3": (6, 12, 3), "pb4": (8, 16, 4)}
REF = {
    "pb2": {"scf": -4317.376693057786, "s0": -4317.38043865596, "s1": -4317.18705227103},
    "pb3": {"scf": -6279.87232750353,  "s0": -6279.87843005372, "s1": -6279.68734502148},
    "pb4": {"scf": -7850.022480822798, "s0": -7850.03052542641, "s1": -7849.83681307442},
}


def complement(S, space, picked, n_keep):
    """metric S の下で space から picked 成分を除去し、S-正規直交な n_keep 本を返す。"""
    r = space - picked @ (picked.T @ S @ space)
    M = r.T @ S @ r
    w, v = np.linalg.eigh(M)
    idx = np.argsort(w)[::-1][:n_keep]
    k = r @ v[:, idx]
    return k / np.sqrt(np.einsum('pi,pq,qi->i', k, S, k))


def build(base, sysd, want_s1=False, out=None, mem=20000):
    nelec_act, norb_act, n_pb = SPEC[sysd]
    n_occ_act = nelec_act // 2          # 占有活性軌道数 = N_Pb
    n_virt_act = norb_act - n_occ_act   # 仮想活性軌道数 = 3 * N_Pb

    chk = f"{base}/{sysd}/scf.chk"
    mol = pyscf.scf.chkfile.load_mol(chk); mol.max_memory = mem
    sd = pyscf.scf.chkfile.load(chk, "scf")
    mf = pyscf.scf.RHF(mol).density_fit(); mf.__dict__.update(sd); mf.max_memory = mem
    C = np.asarray(sd["mo_coeff"]); S = mol.intor_symmetric("int1e_ovlp")
    nocc = mol.nelectron // 2

    # 占有活性 = 価電子帯端（上位 N_Pb 占有軌道）
    occ_idx = list(range(nocc - n_occ_act, nocc))
    Aocc = C[:, occ_idx]

    # 仮想活性 = AVAS('Pb 6p') の仮想ブロック（Pb 6p 反結合 = 伝導帯）
    ncas, ne, mo6p = avas.avas(mf, ["Pb 6p"], threshold=0.2, canonicalize=True, verbose=0)
    no = ne // 2; nc = nocc - no
    virt6p = mo6p[:, nc + no: nc + ncas]
    if virt6p.shape[1] < n_virt_act:
        raise RuntimeError(f"AVAS Pb6p は仮想 {virt6p.shape[1]} 本しか返さず、必要 {n_virt_act} 本に不足")
    Avirt = virt6p[:, :n_virt_act]

    # 完全な正規直交 MO 集合を構成: core | active_occ | active_virt | rest
    occ_hf, virt_hf = C[:, :nocc], C[:, nocc:]
    core = complement(S, occ_hf, Aocc, nocc - n_occ_act)
    rest = complement(S, virt_hf, Avirt, virt_hf.shape[1] - n_virt_act)
    mo = np.hstack([core, Aocc, Avirt, rest])
    err = np.abs(mo.T @ S @ mo - np.eye(mo.shape[1])).max()
    assert err < 1e-8, f"orthonormality broken: {err}"

    print(f"===== {sysd}  ({nelec_act}e{norb_act}o, Pb×{n_pb})  nao={mol.nao_nr()} nocc={nocc} =====", flush=True)
    print(f"  占有活性(価電子帯端) = 正準MO {occ_idx}", flush=True)
    print(f"  仮想活性(Pb 6p 伝導帯) = AVAS {n_virt_act} 本", flush=True)

    cas = pyscf.mcscf.CASCI(mf, norb_act, nelec_act)
    hcore, e_nuc = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb_act)

    cas.fix_spin_(ss=0)
    cas.fcisolver.nroots = 2 if want_s1 else 1
    t = time.time(); cas.kernel(mo); dt = time.time() - t
    e = np.atleast_1d(cas.e_tot)
    r = REF[sysd]
    dm = cas.fcisolver.make_rdm1(cas.ci[0] if want_s1 else cas.ci, norb_act, nelec_act)
    print(f"  CASCI [{dt:.0f}s]  占有数(対角)= {np.round(dm.diagonal(), 3)}", flush=True)
    print(f"  S0: {e[0]:.8f}  README {r['s0']:.8f}  d={(e[0]-r['s0'])*1e3:+.3f} mHa", flush=True)
    if want_s1 and len(e) > 1:
        print(f"  S1: {e[1]:.8f}  README {r['s1']:.8f}  d={(e[1]-r['s1'])*1e3:+.3f} mHa", flush=True)
        print(f"  gap: calc {(e[1]-e[0])*27.211386:.4f} eV  README {(r['s1']-r['s0'])*27.211386:.4f} eV", flush=True)

    if out is None:
        out = os.path.join(os.path.dirname(__file__), "ham_cache", f"{sysd}_ham.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, hcore=hcore, eri=eri, e_nuc=e_nuc,
             nelec_act=nelec_act, norb_act=norb_act,
             occ_active=np.array(occ_idx), casci_s0=float(e[0]), used_df=True,
             recipe="occ=valence-edge canonical (top N_Pb) + virt=AVAS Pb6p conduction")
    print(f"  saved -> {out}", flush=True)
    return float(e[0])


if __name__ == "__main__":
    base, sysd = sys.argv[1], sys.argv[2]
    want_s1 = "--s1" in sys.argv
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    build(base, sysd, want_s1=want_s1, out=out)
