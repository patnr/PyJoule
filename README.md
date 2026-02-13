# PyJoule

<img src="icon.png" alt="icon" width="200">

Named after rigorous experimentalist J. Joule,
this package is for managing a large number of numerical experiments in Python.
It helps you run experiments in parallel and remotely (including the push/pull),
and track, save and load the parameters and results, with minimal code (mental) overhead:

```py
# Replace
results = [experiment(**kws) for kws in params]
# by
data_dir = dispatch(experiment, params, host)
```

This also helps in the configuration and generation of experimental settings,
and the analysis (processing and presentation, including tabulation/plotting) of the results.

It leverages well-know python libraries:
`pathos` for multiprocessing and result storage,
`rsync (ssh)` for remote execution,
`pandas` for tabulation,
and sparse `xarray` for post-processing.

## Alternatives

- **PyJoule**:
  Python-native, manages parameters and results, minimal code overhead, supports remote/HPC execution (via rsync/ssh), no dashboard, low-medium setup.
- **Weights & Biases / MLflow / Neptune.ai**:
  Advanced experiment/parameter tracking, rich dashboards, Python APIs (extra config needed), remote execution for ML (not HPC), medium-high setup.
- **Ray / Dask**:
  Distributed parameter sweeps, web dashboards, Python-native, remote/cloud/cluster execution, high setup complexity.
- **Snakemake / Nextflow**:
  Workflow-based param management (config files), some visualization, Snakemake is Pythonic, strong HPC/remote support, medium-high setup.
- **Sacred / Hydra**:
  Hierarchical config management, minimal dashboard (Sacred only), Python-native, no remote/HPC, low-medium setup.
- **Joblib**:
  Simple param mapping, no dashboard, Python-native, local parallel only, low setup.
- **GNU Parallel / xargs + tmux/screen**:
  Manual param management, no dashboard, not Python-native, remote via manual SSH, low setup, shell skills needed.

[Tabular comparison within the ML domain.](https://dagshub.com/blog/how-to-compare-ml-experiment-tracking-tools-to-fit-your-data-science-workflow/)

## Motivation

When developing and testing numerical methods and algorithms, it is common practice to evaluate them on prototype, toy, or simplified problems. These problems are chosen to balance simplicity (for speed and transparency) with representativeness of the real-world scenarios being targeted.

Although the principle of varying only one parameter at a time is valuable, in practice, we quickly encounter a large number of parameters to consider. Each of the following aspects introduces at least one additional parameter or dimension:

1. **Algorithms/methods (context):** Different algorithms or methods should be compared.
2. **Tuning parameters (fairness):** Each method may require its own set of parameters to be optimized for a fair comparison.
3. **Problem selection (relevance):** A variety of problems should be considered to ensure relevance.
4. **Problem parameters (generality):** Each problem may have its own parameters, which can differ between methods.
5. **Random seed (reliability):** Results should be averaged over multiple random seeds to ensure reliability.

This leads to a great number of experiments to be run and results to be processed,
with the complication that many parameters only apply for some methods/experiments.
As mentioned [here](https://www.youtube.com/watch?v=EeqhOSvNX-A)
hand crafted solutions are error prone, and often an after-thought.

### QoL details

- SSH multiplexing
- Progress bars
- Error handling
- per-project venv
- `threadpoolctl.threadpool_limits(1)`
- Batching for `SLURM`

## Development

### Testing

**Run all tests (excluding slow tests):**
```bash
pytest
```

**Run all tests including slow tests:**
```bash
pytest -m "slow or not slow"
# or simply
pytest -m ""
```

**Run only slow tests (remote host integration tests):**
```bash
pytest -m slow
```

**Run specific host dispatch test:**
```bash
# By parameter name (uses -k for pattern matching)
pytest -m slow -k "subprocess" -v
pytest -m slow -k "localhost" -v
pytest -m slow -k "hpc-cluster" -v

# By full test path
pytest tests/test_example.py::TestIntegration::test_host_dispatch[subprocess] -v -m ""
```

**View all available tests:**
```bash
pytest --collect-only -m ""
```

The test suite includes:
- **Fast tests**: Unit tests for core functionality (run by default)
- **Slow tests**: Integration tests with SSH/remote execution (marked with `@pytest.mark.slow`, skipped by default)
  - Each host appears as a separate parameterized test
  - Results are compared against reference (direct execution)
  - Unreachable hosts are automatically skipped

### Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check formatting
ruff format --check .

# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable linting issues
ruff check --fix .

# Run both (all files)
ruff check --fix . && ruff format .
```
