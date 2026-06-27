"""投入済みの IBM ジョブを job_id(jobs.meta.json に保存)から再取得する回収専用スクリプト。

長時間のポーリング中に結果回収が落ちた(例: トークンのセッション失効)ものの、ジョブ自体は
プラットフォーム上で完了している場合に使う。新しい接続を張り直し、job_id でジョブを開いて
結果(カウント)を取得する。

まだ保存されていない runNN.json だけを、meta ファイルと同じディレクトリへ保存する
(既に存在する run はスキップ)。

実行方法:
  python fetch_qpy_results.py <sys> --shots 10000 --base ../extracted
"""
import os
import json
import argparse

from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sys", choices=["pb2", "pb3", "pb4"])  # 対象系
    ap.add_argument("--shots", type=int, default=10000)    # 投入時のショット数(保存先の特定に使用)
    ap.add_argument("--base", default="../extracted")      # 保存先のベースディレクトリ
    ap.add_argument("--tag", default="")                   # 保存先ディレクトリ名の接尾辞(measure_qpy.py と合わせる)
    args = ap.parse_args()

    # 投入時に保存した jobs.meta.json から job_id と古典レジスタ名を読み込む
    dirname = f"orig_shot{args.shots}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.base, args.sys, "hw_runs", dirname)
    with open(os.path.join(out_dir, "jobs.meta.json")) as f:
        meta = json.load(f)
    creg = meta["creg"]          # カウント取得に使う古典レジスタ名
    job_ids = meta["job_ids"]    # 投入済みジョブの job_id 一覧

    # IBM Quantum へ接続(トークンが環境変数にあればそれを使う)
    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    service = (QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
               if token else QiskitRuntimeService())

    # 各 job_id を開いて結果を回収し、未保存の runNN.json だけ書き出す
    for r, jid in enumerate(job_ids):
        path = os.path.join(out_dir, f"run{r+1:02d}.json")
        if os.path.exists(path):
            # 既に取得済みのランは再取得しない
            print(f"  run{r+1:02d}: exists, skip", flush=True)
            continue
        job = service.job(jid)
        st = job.status()
        print(f"  run{r+1:02d} {jid} status={st}", flush=True)
        res = job.result()
        counts = getattr(res[0].data, creg).get_counts()
        with open(path, "w") as f:
            json.dump(counts, f, indent=2)
        print(f"  run{r+1:02d}: unique={len(counts)} sum={sum(counts.values())} -> {path}",
              flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
