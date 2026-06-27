# Run-to-run variability of repeated hardware measurements

**Backend:** `ibm_pittsburgh` &nbsp;|&nbsp; **Circuits:** original transpiled LUCJ
circuits (`pb*_sqd_circuit_transpiled_ibm_pittsburgh_*.qpy`), submitted as-is (no
re-transpilation) &nbsp;|&nbsp; **Replicates:** R = 10 independent jobs per system
&nbsp;|&nbsp; **Shots:** 10,000 per run.

This analysis quantifies how the **raw sampling output** of the *same* circuit
varies from run to run. It is **Hamiltonian-free**, so it is unaffected by the
open question about the active-space energy reference — it measures only the
quantum-measurement / sampling variability that feeds SQD.

Bit convention (matches `sqd_pb_en.py`): integer key = `int(bitstring, 2)`;
α = lower `norb` bits, β = upper `norb` bits. A configuration is **physical**
when `popcount(α) = Na` and `popcount(β) = Nb`.

## Summary (mean ± std over 10 runs)

| system | active space | unique configs | physical fidelity `p_phys` | top-1 prob | pairwise TVD |
|--------|--------------|----------------|----------------------------|------------|--------------|
| pb2 | 4e, 8o  | 1385 ± 43  | **0.481 ± 0.007** | 0.347 ± 0.010 | 0.045 ± 0.014 |
| pb3 | 6e, 12o | 5164 ± 286 | **0.211 ± 0.038** | 0.105 ± 0.037 | 0.364 ± 0.206 |
| pb4 | 8e, 16o | 8376 ± 105 | **0.107 ± 0.004** | 0.033 ± 0.003 | 0.468 ± 0.024 |

- **`p_phys`** = fraction of shots that land in the correct particle-number
  sector (the rest are discarded / repaired by SQD configuration recovery). It is
  the single best hardware-quality indicator and drops steeply with circuit size
  (deeper circuit, more qubits measured → more noise): 48% → 21% → 11%.
- **pairwise TVD** = total-variation distance between the (physical, renormalized)
  empirical distributions of two runs, averaged over all 45 run-pairs. Larger =
  less reproducible sampling.

## Metrics explained

- **unique configs** — number of distinct bitstrings seen in 10k shots. Grows
  with system size; its run-to-run std is the simplest variability signal.
- **physical-sector fidelity `p_phys`** — what fraction of the 10k shots is
  usable by SQD. The complement is noise that violated particle-number
  conservation.
- **top-1 probability** — weight of the single most-frequent bitstring (the HF /
  dominant determinant). Falls as the wavefunction spreads over more configs.
- **pairwise TVD** — run-to-run distance of the physical distributions.
- **top-K Jaccard** — overlap of the K most-probable physical configurations
  between two runs = how reproducible the **subspace SQD would diagonalize** is.

## Figure: `variability_overview.png` (5 panels)

- **(A) particle-number preservation** — `p_phys` per run. pb2 is tight
  (±0.007); pb4 is low but tight; pb3 is the noisiest *and* most variable
  (one outlier run at 0.12 vs the cluster at ~0.21).
- **(B1) configuration coverage — total** — number of distinct bitstrings per
  run (correct or not): ≈1385 (pb2) → 5164 (pb3) → 8376 (pb4).
- **(B2) configuration coverage — physical configs** — number of distinct
  *particle-number-conserving* bitstrings per run, on its own scale: ≈106 (pb2)
  → 346 (pb3) → 490 (pb4). It grows with system size (it only looks flat if
  plotted on the total scale of B1), but is a tiny fraction of both the total
  unique strings and the combinatorially huge physical space
  (C(norb,Na)² = 784 / 48,400 / 3.3M for pb2/pb3/pb4), i.e. 10k shots sample only
  a sliver of the larger systems.
- **(C) run-to-run distribution distance** — pairwise TVD. pb2 distributions are
  nearly identical across runs (TVD≈0.045); pb3/pb4 differ substantially
  (≈0.36–0.47), i.e. each 10k-shot run probes a noticeably different sample.
- **(D) reproducibility of the selected subspace** — mean top-K Jaccard vs K.
  For a small subspace (K=50) pb2 reproduces ~71% of its top configs run-to-run,
  pb3 ~55%, pb4 ~40%. The overlap decreases for larger K and for larger systems.

(Panel layout: 2×3 grid; A / B1 / B2 on the top row, C / D on the bottom row,
the sixth cell is hidden.)

## Key findings

1. **Variability grows strongly with system size.** pb2 is essentially
   reproducible (TVD≈0.045, top-50 Jaccard≈0.71); pb4 is the most variable
   (TVD≈0.47, top-50 Jaccard≈0.40). This is the run-to-run dispersion that
   motivates the paper's "median of several realizations" protocol.
2. **Hardware noise dominates the dispersion.** The usable (physical) fraction
   collapses from 48% (pb2) to 11% (pb4); the discarded majority is
   particle-number-violating noise, and the part that survives still differs
   between runs.
3. **pb3 has the largest *relative* run-to-run scatter** (`p_phys` std 0.038 on a
   0.211 mean, TVD std 0.206), driven by one low-quality run (run 05). This is a
   concrete example of why a single shot-set can mislead and repeated runs are
   needed.
4. **The subspace SQD selects is only partially reproducible**, especially for
   larger K and larger systems — directly explaining run-to-run energy spread in
   the downstream SQD step.

## pb3 outlier (run 05): a transient device fluctuation, not a circuit/calibration issue

pb3's outsized TVD *spread* in panel (C) is driven almost entirely by a single
run. Per-run diagnostics (mean pairwise TVD to the other 9 runs):

| run | p_phys | top-1 | mean TVD to others |
|-----|--------|-------|--------------------|
| 1–4, 6–9 | 0.19–0.27 | 0.09–0.17 | 0.29–0.35 (normal cluster) |
| **5** | **0.121** | **0.032** | **0.714** (anomaly) |
| 10 | 0.211 | 0.068 | 0.452 (mild) |

Dropping run 05 collapses pb3's TVD from mean 0.364 / std 0.206 to
mean 0.277 / std 0.110 (max 0.784 → 0.493).

Investigation via the IBM job ids (`jobs.meta.json`) **rules out** a circuit or
calibration-generation cause:

- **Same circuit** is reused for all 10 runs — a circuit defect would affect all
  runs systematically (as in pb4's uniform behaviour), not just one.
- **Same time window**: all 10 pb3 jobs executed back-to-back within
  16:33:22–16:37:10 UTC (2026-06-20, ~4 minutes). run 05 ran at 16:34:33, right
  in the middle — no special queue/time slot.
- **Same calibration**: the calibration in effect was last updated at
  16:20 UTC, ~13 min before the batch; no recalibration occurred mid-batch.

So run 05 caught a **transient in-session hardware degradation** during its ~63 s
execution (e.g. a TLS defect drifting onto a qubit, a momentary readout/2-qubit-gate
fault) that calibration metrics do not capture. This *strengthens* the central
message: even within one calibration cycle and a 4-minute span, the device can
emit a markedly worse realization — exactly why several realizations + median are
needed. (A fresh pb3 re-measurement is stored under
`extracted/pb3/hw_runs/orig_shot10000_rerun/` for comparison.)

## Data layout

```
results_en/hw_variability/orig_pittsburgh/
├── README.md                  (this file)
├── per_run_metrics.csv        per-run metrics (30 rows)
├── variability_summary.csv    per-system mean/std summary
└── variability_overview.png   5-panel figure

extracted/<sys>/hw_runs/orig_shot10000/
├── jobs.meta.json             backend, shots, creg, the 10 job_ids
└── run01..run10.json          raw {bitstring: count}, 10k shots each

extracted/pb3/hw_runs/orig_shot10000_rerun/   (pb3 re-measurement, for comparison)
├── jobs.meta.json
└── run01..run10.json
```

Regenerate the figure/CSVs with: `python analysis/variability_analysis.py`

Related scripts:
- `analysis/measure_qpy.py`        submit a .qpy circuit R times (`--tag` for re-runs)
- `analysis/fetch_qpy_results.py`  re-fetch results by job_id if collection was interrupted
- `analysis/compute_run_stats.py`  quick per-system mean/std of the 4 key metrics
