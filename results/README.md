# Results

This directory stores benchmark results for each algorithm-suite combination.

## Directory Layout

```
results/
├── bbob/
│   ├── cma-es/
│   ├── de/
│   └── pso/
├── cec2017/
│   ├── cma-es/
│   └── ...
└── real-world/
```

## Result Entry Format

Each algorithm-suite result directory should contain:

- `metadata.yaml` — hardware, software versions, date, random seeds
- `raw_data.csv` — one row per evaluation: `[run_id, feval, best_so_far, timestamp_ms]`
- `energy.csv` — one row per run: `[run_id, total_joules, measurement_method]` (optional but encouraged)
- `summary.md` — human-readable summary table of key metrics

## Metrics Reported

| Metric | Description |
|---|---|
| **Success rate** | % of runs reaching target quality `f* + 1e-8` |
| **Median FEs** | Median function evaluations to reach target (over successful runs) |
| **ERT** | Expected Running Time — accounts for restarts |
| **Energy/FE** | Mean joules per function evaluation |
| **Energy to target** | Total joules to reach target quality (median over runs) |
| **AUC** | Area under convergence curve — anytime performance |

## Measurement Guidelines

### Energy Measurement

Preferred methods (in order of accuracy):
1. **Hardware power meter** (e.g., Yokogawa WT series) on wall outlet
2. **RAPL** (Running Average Power Limit) — CPU+DRAM, available on Intel/AMD Linux via `perf stat -e power/energy-pkg/`
3. **nvidia-smi** for GPU power (if GPU is used)
4. **Cloud provider billing** — CPU-hours × TDP as last resort

Always record: hardware model, CPU TDP, whether idle draw is subtracted, OS, and any other concurrent processes.

### Reproducibility

- Fix random seeds and report them
- Report Python/library versions (`pip freeze > requirements_snapshot.txt`)
- Report OS, CPU model, RAM, and whether NUMA effects are relevant
