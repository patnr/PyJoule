"""Measure total duration for jobs running on different servers.

Each individual job hogs 1 CPU and a given amount of memory, for a given amount of time.
"""

import time

import numpy as np
import pandas as pd

from xp import dict_prod, dispatch, load_data


def experiment(MB=1, seconds=10, job_nr=3000):
    """Hog computing resources."""
    start_time = time.time()
    np.random.seed(job_nr)

    # Memory
    data = np.random.rand(int(MB * 1024**2 / 8))

    # Compute
    nIter = 0
    y = 0.0
    while time.time() < (start_time + seconds):
        nIter += 1
        x = np.random.rand(1000)
        y += np.sum(np.sin(x) * np.cos(x))

    return {"nIter": nIter, "mval": np.mean(data)}


if __name__ == "__main__":
    xps = dict_prod(
        MB=[1],
        seconds=[10],
        job_nr=list(range(3 * 1024)),
    )

    # Uncomment to run locally
    # from tqdm import tqdm
    # res = [experiment(**kwargs) for kwargs in tqdm(xps)]

    # Configure host for dispatch
    # host = None  # Local execution
    # host = "localhost"
    # host = "cno-0001" 3:20
    host = "login-1.hpc.intra.norceresearch.no"  # 0:17

    dir = dispatch(
        experiment,
        xps,
        host,
        nBatch=60,
    )
    res = load_data(dir / "res")

    # * Want nBatch > 5x nNodes (to enable efficient load balancing by SLURM)
    # * Want nBatch * nCPU (or --cpus-per-task) == nCPUs_available_in_total
    #   In principle should not be a problem to use 5x nCPUs_available_in_total,
    #   but there appears to be significant SLURM overhead such that it is best if no jobs are pending,
    #   i.e. if LHS == RHS - epsilon
    # * Must keep nBatch < 60 due to queue system limit

    df = pd.concat([pd.DataFrame(xps), pd.DataFrame.from_records(res)], axis=1)
    print(df)
