"""選択した活性空間軌道を Avogadro 用に書き出す（Molden + cube）。

各系（pb2/pb3/pb4）について:
  - 活性空間の軌道（占有=価電子帯端 Pb-I, 仮想=Pb 6p 伝導帯）を再構成し、
  - 比較用に「除外した CH3NH3 局在の正準 LUMO」も数本含めて、
  - 1) Molden ファイル（全活性軌道をまとめて Avogadro で対話的に閲覧）
    2) cube ファイル（各軌道の等値面。Avogadro で確実に描画可能）
  を出力する。さらに各軌道の Pb/I/有機(org) population を表示する。

使い方:
  uv run export_orbitals.py <BASE> <sys> [--cube] [--grid 80]
    --cube  : cube も生成（やや時間がかかる）。無指定だと Molden と population のみ。
出力先: analysis/orbitals_avogadro/
"""
import os, sys
import numpy as np
import pyscf.scf
from pyscf.mcscf import avas
from pyscf.tools import molden, cubegen

SPEC = {"pb2": (4, 8, 2), "pb3": (6, 12, 3), "pb4": (8, 16, 4)}


def complement(S, space, picked, n_keep):
    r = space - picked @ (picked.T @ S @ space)
    M = r.T @ S @ r
    w, v = np.linalg.eigh(M)
    idx = np.argsort(w)[::-1][:n_keep]
    k = r @ v[:, idx]
    return k / np.sqrt(np.einsum('pi,pq,qi->i', k, S, k))


def frag_pop(mol, S, mo):
    """各軌道の Pb / I / org population(%) を返す。"""
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    ao_atom = np.array([lab[0] for lab in mol.ao_labels(fmt=False)])
    ao_frag = np.array(["Pb" if sym[a] == "Pb" else "I" if sym[a] == "I" else "org"
                        for a in ao_atom])
    SCmo = S @ mo
    out = []
    for i in range(mo.shape[1]):
        p = mo[:, i] * SCmo[:, i]; tot = p.sum()
        out.append({f: p[ao_frag == f].sum() / tot * 100 for f in ["Pb", "I", "org"]})
    return out


def run(base, sysd, do_cube=False, grid=80):
    nelec_act, norb_act, n_pb = SPEC[sysd]
    n_occ_act = nelec_act // 2
    n_virt_act = norb_act - n_occ_act
    chk = f"{base}/{sysd}/scf.chk"
    mol = pyscf.scf.chkfile.load_mol(chk)
    sd = pyscf.scf.chkfile.load(chk, "scf")
    mf = pyscf.scf.RHF(mol).density_fit(); mf.__dict__.update(sd)
    C = np.asarray(sd["mo_coeff"]); moe = np.asarray(sd["mo_energy"])
    S = mol.intor_symmetric("int1e_ovlp"); nocc = mol.nelectron // 2

    # 占有活性（価電子帯端）と仮想活性（AVAS Pb 6p）
    occ_idx = list(range(nocc - n_occ_act, nocc))
    Aocc = C[:, occ_idx]
    ncas, ne, mo6p = avas.avas(mf, ["Pb 6p"], threshold=0.2, canonicalize=True, verbose=0)
    no = ne // 2; nc = nocc - no
    Avirt = mo6p[:, nc + no: nc + ncas][:, :n_virt_act]
    # 比較用: 除外した正準 LUMO（CH3NH3 局在）を n_pb 本
    contrast_idx = list(range(nocc, nocc + n_pb))
    Acon = C[:, contrast_idx]

    # 直交性確保のため core/rest は不要（書き出すのは選択軌道のみ）
    blocks = [("occ", Aocc, occ_idx), ("virt(Pb6p)", Avirt, [None] * n_virt_act),
              ("excluded LUMO", Acon, contrast_idx)]
    mo_all = np.hstack([b[1] for b in blocks])
    labels = []
    for name, arr, idxs in blocks:
        for j, ix in enumerate(idxs):
            labels.append(f"{name}" + (f" (canon MO {ix})" if ix is not None else f" #{j+1}"))

    pops = frag_pop(mol, S, mo_all)
    outdir = os.path.join(os.path.dirname(__file__), "orbitals_avogadro")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== {sysd}  ({nelec_act}e{norb_act}o, Pb×{n_pb}) =====", flush=True)
    print(f"{'#':>2} {'role':18} {'Pb%':>6} {'I%':>6} {'org%':>6}", flush=True)
    for i, (lab, p) in enumerate(zip(labels, pops)):
        print(f"{i+1:>2} {lab:18} {p['Pb']:6.1f} {p['I']:6.1f} {p['org']:6.1f}", flush=True)

    # Molden（活性軌道 + 比較LUMO をまとめて1ファイル）
    occv = np.array([2.0] * n_occ_act + [0.0] * n_virt_act + [0.0] * n_pb)
    enev = np.concatenate([moe[occ_idx], np.zeros(n_virt_act), moe[contrast_idx]])
    mpath = os.path.join(outdir, f"{sysd}_active.molden")
    molden.from_mo(mol, mpath, mo_all, occ=occv, ene=enev)
    print(f"  molden -> {mpath}  (MO 1..{mo_all.shape[1]} = 上表の順)", flush=True)

    # cube（任意）
    if do_cube:
        for i, lab in enumerate(labels):
            tag = lab.replace(" ", "_").replace("(", "").replace(")", "").replace("#", "n")
            cpath = os.path.join(outdir, f"{sysd}_{i+1:02d}_{tag}.cube")
            cubegen.orbital(mol, cpath, mo_all[:, i], nx=grid, ny=grid, nz=grid)
            print(f"  cube -> {cpath}", flush=True)


if __name__ == "__main__":
    base, sysd = sys.argv[1], sys.argv[2]
    do_cube = "--cube" in sys.argv
    grid = int(sys.argv[sys.argv.index("--grid") + 1]) if "--grid" in sys.argv else 80
    run(base, sysd, do_cube=do_cube, grid=grid)
