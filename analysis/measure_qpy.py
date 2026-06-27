"""元のトランスパイル済み LUCJ 回路(.qpy)を IBM 実機へ R 回投入し、
バラつき解析用に独立な測定データ(ショットカウント)を収集するスクリプト。

.qpy 回路は ibm_pittsburgh の 156 量子ビット配置に合わせて既にトランスパイル
(ISA 化)されており、終端の測定も含まれている。そのため再トランスパイル・再レイアウト
はせず、そのまま投入する。各Runは独立したジョブとして「先にまとめて投入」することで
キュー待ち時間を重ね合わせる。job_id は投入直後にファイルへ保存するので、結果回収が
途中で中断しても(例: トークン失効)後から job_id 経由で再取得できる
(回収専用スクリプト fetch_qpy_results.py を参照)。

カウントは {ビット列: 出現数} の JSON として保存する
(アーカイブ extracted/<sys>/shot-*/sqd_counts_*.json と同形式)。保存先:
  extracted/<sys>/hw_runs/orig_shot<N>/run01..R.json

実行方法(環境変数 QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE が必要):
  python measure_qpy.py <qpy_path> <sys> --shots 10000 --runs 10 --backend ibm_pittsburgh
"""
import os
import json
import argparse

from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qpy_path")                       # 投入する .qpy 回路のパス
    ap.add_argument("sys", choices=["pb2", "pb3", "pb4"])  # 対象系
    ap.add_argument("--shots", type=int, default=10000)    # 1Runあたりのショット数
    ap.add_argument("--runs", type=int, default=10)        # 繰り返し数(Run数) R
    ap.add_argument("--backend", default="ibm_pittsburgh") # 投入先バックエンド
    ap.add_argument("--base", default="../extracted")      # 保存先のベースディレクトリ
    ap.add_argument("--tag", default="")                   # 保存先ディレクトリ名に付ける接尾辞(再測定の区別用)
    args = ap.parse_args()

    # .qpy 回路を読み込む(1ファイルに1回路の前提)
    with open(args.qpy_path, "rb") as fh:
        circ = qpy.load(fh)[0]
    creg = circ.cregs[0].name  # 測定先の古典レジスタ名(カウント取得時に使用)
    print(f"loaded {args.qpy_path}: {circ.num_qubits}q, measures {circ.num_clbits} "
          f"-> creg '{creg}', depth={circ.depth()}", flush=True)

    # 保存先ディレクトリ: extracted/<sys>/hw_runs/orig_shot<N>[_<tag>]/
    dirname = f"orig_shot{args.shots}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.base, args.sys, "hw_runs", dirname)
    os.makedirs(out_dir, exist_ok=True)

    # IBM Quantum へ接続(トークンが環境変数にあればそれを使い、無ければ保存済み資格情報)
    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    service = (QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
               if token else QiskitRuntimeService())
    backend = service.backend(args.backend)
    print(f"backend: {args.backend} (#qubits={backend.num_qubits})", flush=True)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = args.shots

    # まず R 本のジョブを「すべて」投入する(キュー待ちを並走させるため)
    jobs = []
    for r in range(args.runs):
        job = sampler.run([circ], shots=args.shots)
        jobs.append(job)
        print(f"  submitted run{r+1:02d}  job_id={job.job_id()}", flush=True)

    # job_id 等のメタ情報を即保存(結果回収が中断しても後から再取得できるように)
    meta = {"backend": args.backend, "shots": args.shots, "runs": args.runs,
            "qpy": os.path.basename(args.qpy_path), "creg": creg,
            "job_ids": [j.job_id() for j in jobs]}
    with open(os.path.join(out_dir, "jobs.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  job ids -> {os.path.join(out_dir, 'jobs.meta.json')}", flush=True)

    # 各ジョブの結果(カウント)を回収して runNN.json に保存
    for r, job in enumerate(jobs):
        res = job.result()
        counts = getattr(res[0].data, creg).get_counts()
        path = os.path.join(out_dir, f"run{r+1:02d}.json")
        with open(path, "w") as f:
            json.dump(counts, f, indent=2)
        print(f"  run{r+1:02d}: unique={len(counts)} sum={sum(counts.values())} -> {path}",
              flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
