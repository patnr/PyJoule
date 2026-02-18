import time

import numpy as np
from tqdm import tqdm

from xp import dict_prod, dispatch, load_data


def experiment(memory_mb=1, duration_sec=10, seed=3000):
    """Configurable experiment for testing resource usage.

    Args:
        memory_mb: Memory to allocate in MB
        duration_sec: How long to run computation in seconds
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    start_time = time.time()
    results = {}

    # Memory
    bytes_per_num = 8
    bytes_per_mb = 2**20
    array_size = int(memory_mb * bytes_per_mb / bytes_per_num)
    data = np.random.rand(array_size)
    results["mval"] = float(np.mean(data))

    # Compute
    elapsed = 0
    iterations = 0
    x = 0.0
    while elapsed < duration_sec:
        temp_array = np.random.rand(1000)
        x += np.sum(np.sin(temp_array) * np.cos(temp_array))
        iterations += 1
        elapsed = time.time() - start_time

    results["iterations"] = iterations
    results["checksum"] = float(x)

    # results["actual_duration_sec"] = time.time() - start_time

    return results


if __name__ == "__main__":
    xps = dict_prod(
        memory_mb=[1],
        duration_sec=[10],
        seed=list(range(3000)),
    )

    # Uncomment to run locally
    # res = [experiment(**kwargs) for kwargs in tqdm(xps)]

    # Configure host for dispatch
    # host = None  # Local execution
    # host = "localhost"
    # host = "cno-0001"
    host = "login-1.hpc.intra.norceresearch.no"

    dir = dispatch(
        experiment,
        xps,
        host,
        nBatch=12
    )
    res = load_data(dir / "res")

    import pandas as pd
    df = pd.concat([pd.DataFrame(xps), pd.DataFrame.from_records(res)], axis=1)
    # df = df.sort_values(["memory_mb", "duration_sec"])
    print(df)
