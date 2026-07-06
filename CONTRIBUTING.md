# Contributing

Thank you for helping expand this library. Contributions of any size are welcome — fixing a typo, adding energy data for an existing entry, or documenting a new algorithm.

## Adding a New Algorithm

1. Copy [`algorithms/_template.md`](algorithms/_template.md) to `algorithms/<acronym-lowercase>.md`
2. Fill in all sections. Leave cells as `—` rather than omitting rows.
3. Add the algorithm to the table in [`README.md`](README.md)
4. Open a pull request with title: `Add: <Algorithm Name>`

## Adding Benchmark Results

1. Create the directory `results/<suite>/<algorithm>/`
2. Include at minimum: `metadata.yaml` and `raw_data.csv`
3. See [`results/README.md`](results/README.md) for format details
4. Update the convergence table in the algorithm's `.md` file

## Updating Energy Data

Energy data is the hardest to collect and the most valuable. If you have access to power measurement hardware or RAPL readings, please contribute them even for algorithms already documented.

## Style Guidelines

- Keep algorithm pages factual and citation-backed
- Use SI units: joules (J) for energy, seconds (s) for time
- Dimension is always `d`, population size is `n` or algorithm-specific notation
- Complexity expressions use standard O-notation
- Avoid promotional language — document tradeoffs honestly

## Code of Conduct

Be constructive. Disagreements about benchmarking methodology are welcome; personal attacks are not.
