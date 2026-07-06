# Benchmark Suites

This directory documents the test suites used to evaluate algorithms in this library.

## BBOB / COCO

The **Black-Box Optimization Benchmarking** framework (via the [COCO platform](https://github.com/numbbo/coco)) provides 24 noiseless real-parameter test functions in 5 groups:

| Group | Functions | Characteristics |
|---|---|---|
| Separable | f1–f5 | Each variable independent |
| Low/moderate conditioning | f6–f9 | Moderate ill-conditioning |
| High conditioning / unimodal | f10–f14 | Strongly ill-conditioned |
| Multimodal, adequate structure | f15–f19 | Multiple basins, structured |
| Multimodal, weak structure | f20–f24 | Highly deceptive |

Standard dimensions: 2, 3, 5, 10, 20, 40.

**How to run:** Use the COCO Python or C interface. Results are post-processed with `cocopp` to generate ECDF plots and data profiles.

## CEC Suites

IEEE Congress on Evolutionary Computation releases annual benchmark suites:

- **CEC 2013** — 28 functions, d=5/10/30
- **CEC 2017** — 30 functions, d=10/30/50/100 (widely used)
- **CEC 2022** — 12 functions, d=10/20

CEC suites include shifted, rotated, and hybrid compositions to test generalization.

## Real-World Problem Sets

| Problem Set | Domain | Dimensions | Source |
|---|---|---|---|
| HPOB | Hyperparameter optimization | varied | [hpob-benchmark](https://github.com/releaunifreiburg/HPO-B) |
| SVM tuning | ML | 2–8 | OpenML |
| Engineering design problems | Mechanical/electrical | 3–30 | Literature |

## Adding a New Benchmark

Create a new `.md` file in this directory following the pattern:
- Description and origin
- Number of functions and dimension range
- Standard budget (function evaluations)
- How to run / obtain the suite
- Citation
