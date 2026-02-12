import itertools
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import dill
from tqdm.auto import tqdm

from . import uplink

timestamp = "%Y-%m-%d_at_%H-%M-%S"
bar_frmt = "{l_bar}|{bar}| {n_fmt}/{total_fmt}, ⏱️ {elapsed} ⏳{remaining}, {rate_fmt}{postfix}"
responsive = {"check": True, "capture_output": True, "text": True}


def dict_prod(**kwargs):
    """Product of `kwargs` values."""
    # PS: the first keys in `kwargs` are the slowest to increment.
    return [dict(zip(kwargs, x, strict=True)) for x in itertools.product(*kwargs.values())]


def progbar(*args, **kwargs):
    return tqdm(*args, bar_format=bar_frmt, **kwargs)


def load_data(pth, pbar=True):
    pbar = progbar if pbar else (lambda x: x)
    data = []
    for r in pbar(sorted(pth.iterdir(), key=lambda p: int(p.name))):
        try:
            data.extend(dill.loads(r.read_bytes()))
        except Exception as e:
            print(f"Warning: Failed to load {r}: {e}")
    return data


def find_latest_run(root: Path):
    """Find the latest experiment (dir containing many)"""
    lst = []
    for f in root.iterdir():
        try:
            f = datetime.strptime(f.name, timestamp)
        except ValueError:
            pass
        else:
            lst.append(f)
    f = max(lst)
    f = datetime.strftime(f, timestamp)
    return f


def git_dir():
    """Get project (.git) root dir and HEAD 'sha'."""
    git_dir = subprocess.run(["git", "rev-parse", "--show-toplevel"], **responsive).stdout.strip()
    return Path(git_dir)


def git_sha():
    """Get project HEAD 'sha'."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], **responsive).stdout.strip()


def mk_data_dir(
    data_dir,
    tags=(),  # Whatever you want, e.g. "v1"
    mkdir=True,  # Make dirs, including xps/ and res/
):
    """Add timestamp/tag and mkdir for data storage."""
    if tags:
        data_dir /= tags
    else:
        data_dir /= datetime.now().strftime(timestamp)

    if mkdir:
        data_dir.mkdir(parents=True)
        (data_dir / "xps").mkdir()
        (data_dir / "res").mkdir()

    return data_dir


def find_proj_dir(script: Path):
    """Find python project's root dir.

    Returns the (shallowest) parent below `script`
    of first found among some common root markers.
    """
    markers = ["pyproject.toml", "requirements.txt", "setup.py", ".git"]
    for d in script.resolve().parents:
        for marker in markers:
            candidate = d / marker
            if candidate.exists():
                return d


def save(xps, data_dir, nBatch):
    print(f"Saving {len(xps)} xp's to", data_dir)
    ceil_division = lambda a, b: (a+b-1)//b  # noqa: E731
    batch_size = ceil_division(len(xps), nBatch)
    nBatch = ceil_division(len(xps), batch_size)

    def save_batch(i):
        xp_batch = xps[i * batch_size : (i + 1) * batch_size]
        (data_dir / "xps" / str(i)).write_bytes(dill.dumps(xp_batch))

    # saving can be slow ⇒ mp
    # from .local_mp import mp
    # mp(save_batch, range(nBatch))
    for i in tqdm(list(range(nBatch))):
        save_batch(i)

    # List resulting paths
    paths_xps = sorted((data_dir / "xps").iterdir(), key=lambda p: int(p.name))
    assert paths_xps, f"No files found in {data_dir}"
    return paths_xps


def get_cluster_resources(remote: uplink.Uplink):
    # Columns: [Partition, CPUS(A/I/O/T), NODES(A/I)]
    resources = remote.cmd('sinfo -o "%P %C %A"').stdout
    for line in resources.strip().splitlines()[1:]:  # skip header
        partition, nCPUS, nNODES = line.split()
        if partition.startswith("comp"):
            cpus = map(int, nCPUS.split("/"))
            nodes = map(int, nNODES.split("/"))
            cpus = dict(zip(["allocated", "idle", "other", "total"], cpus))
            nodes = dict(zip(["allocated", "idle"], nodes))
            return cpus, nodes


def submit_and_monitor_slurm(remote, cmd, nCPU, remote_dir, script, paths_xps):
    # Send job submission script
    with NamedTemporaryFile(mode="w+t", delete_on_close=False) as sbatch:
        txt = (Path(__file__).parent / "slurm_script.sbatch").read_text()
        txt = eval(f"f'''{txt}'''", {}, locals())  # interpolate f-strings inside {txt}
        sbatch.write(txt)
        sbatch.close()
        remote.rsync(sbatch.name, remote_dir / "job_script.sbatch")

    # Submit
    job_id = remote.cmd(f"command cd {remote_dir}; sbatch job_script.sbatch")
    print(job_id.stdout, end="")
    job_id = int(re.search(r"job (\d*)", job_id.stdout).group(1))

    # Monitor job progress
    nJobs = len(paths_xps)
    try:
        with tqdm(total=nJobs, desc="Jobs") as pbar:
            unfinished = nJobs
            while unfinished:
                time.sleep(1)  # dont clog the ssh uplink
                new = f"squeue -j {job_id} -r -h -t pending,running,completing | wc -l"
                new = int(remote.cmd(new).stdout)
                inc = unfinished - new
                pbar.update(inc)
                unfinished = new
    except KeyboardInterrupt:
        print(f"\nCancelling job {job_id}...")
        remote.cmd(f"scancel {job_id}")
        raise

    # Provide error summary
    # NOTE: Most errors will be caught (and logged) already by `local_mp.py`
    failed = f"sacct -j {job_id} --format=JobID,State,ExitCode,NodeList | grep -E FAILED"
    failed = remote.cmd(failed, check=False).stdout.splitlines()
    if failed:
        regex = r"_(\d+).*(node-\d+) *$"
        nodes = {int((m := re.search(regex, ln)).group(1)): m.group(2) for ln in failed}
        for task in nodes:
            print(f" Error for job {job_id}_{task} on {nodes[task]} ".center(70, "="))
            print(remote.cmd(f"cat {remote_dir}/error/{task}").stdout)
        raise RuntimeError(f"Task(s) {list(nodes)} had errors, see printout above.")


# Note on development
# Would like to develop `dispatch` through an *editable* install,
# but this does not easily work when transposed to *remote*,
# because the local path is of course invalid (and hard to replicate).
# Flagging the package as `--dev` or optional (and not installing it on remote) does not work,
# because `uv` will still check if it's present.
# Solutions:
#
# - Sync it via github. Seems overkill, and has complications:
#   * Use `uv sync --upgrade-package xp` to upgrade in lockfile.
#   * May need to clear cache to pick up latest commit.
# - Copy `src/xp/` dir to PWD. I.e. no install.
#   ⇒ must disable `xp` among dependencies, but copy over sub-dependencies.
# - The above duplicate is quite likely to cause confusion.
#   ⇒ Symlink instead. Requires `-L` flag to `rsync` (append to `-azh`).


def dispatch(
    fun: callable,
    xps: list,
    host: str = "SUBPROCESS",
    script: Path = None,
    nCPU: int = None,
    nBatch: int = None,
    proj_dir: Path = None,
    data_root: Path = Path.home() / "data",
    data_root_on_remote: Path = None,
):
    """
    Execute function over parameter sets on remote hosts/clusters (or locally).

    Essentially: `[fun(**kwargs) for kwargs in xps]`.

    This presumes that the jobs be embarrasingly parallelizable.

    Parameters
    ----------
    fun : callable
        Function to apply to each experiment. Must accept `**kwargs`.
    xps : list
        Job array, i.e. list of parameter dictionaries to pass to `fun`.
    host : str, optional
        Remote server, e.g. "cno-006".
        Can also be an `ssh/.config` alias, and supports wildcards, e.g., "my-gcp*".
        See `xp/setup-compute-node.sh` for instructions on setting up a Google cloud VM.
        Default is `"SUBPROCESS"`, i.e. local execution.
        Another value commonly used for testing is `"localhost"`.
    script : Path, optional
        Path to script containing `fun`, auto-detected if `None`.
        Used to import "by name" and thus avoid pickling `fun`, which often contains deep references,
        and would consume excessive storage/bandwidth (especially if saved with each experiment).
    nCPU : int, optional
        Number of CPUs used by python's multiprocessing (locally, on a given server, or cluster node).
        Defaults to `None` ⇒ auto-detect.
    nBatch : int, optional
        Number of batches to split `xps` job array into. Useful for SLURM clusters.
        Note: this enables *nested* multiprocessing (SLURM + python).
        * Let `N` be the total available CPUs, and suppose `len(xps) >> N` for simplicity.
          Example: NORCE HPC cluster has 3584 CPUs distributed as 14 nodes * 256 CPUs/node.
        * Maybe don't want to hog all available CPUs? Not an important consideration if using `--nice`.
        * Want `nBatch * nCPU == n N` for some integer `n > 0` to make use of all CPUs.
          If instead `n` is slightly above integer, e.g. 5.01,
          then only a single batch will be running towards the end of the total job
          (assuming uniformity of experiment duration and nodes).
        * It might seem that you could set `nCPU=1` and use `nBatch=N`, however
          - Must keep `nBatch < 1000` due to queue system limit.
          - SLURM is significantly slower in distributing jobs than py multiprocessing.
          - Saving many `xps` is slow (even though total data is same), even w/ multiprocessing.
        * Still, want at least `nBatch > 4x nNodes`, to get some load balancing by SLURM.

        Defaults: `56` for NORCE HPC, `1` for local/other.
        Also see: `get_cluster_resources`
    proj_dir : Path, optional
        Project root directory. Gets copied into (and so uploaded with) `data_dir`.
        Must be parent of `script`. Auto-detected via git if `None`.
        PS: using "." may seem reasonable, but is bad practice
        (promotes dependence on whatever happens to be `cwd`)
        Still, the `cwd` is preserved on remote if possible, i.e. if child of `proj_dir`.
    data_root : Path, optional
        Local root for experiment data. Default: `~/data`
    data_root_on_remote : Path, optional
        Remote root for data. Auto-set: `${USERWORK}` (NORCE HPC) or `${HOME}/data` (other).

    Returns
    -------
    Path
        Path to local data directory containing experiment inputs and results.

    Notes
    -----
    On use  of execution wrapper `launch_xps.py`:

    When working remotely, why not just `xargs -P`?
    Script takes care of infrastructure wrapping:
    - Load serialized experiment parameters from disk
    - Import the user's function from the script (supporting sibling imports)
    - Execute with local multiprocessing (`mp`)
    - Save serialized results back to disk.

    While subprocessing when working locally can be avoided, it
    ensures uniform execution across all hosts (local subprocess, SSH, SLURM).
    Both local and remote execution use the same code path for consistency.
    """
    # Validate inputs before expensive operations
    if not callable(fun):
        raise TypeError(f"fun must be callable, got {type(fun)}")
    if not xps:
        raise ValueError("xps list cannot be empty")

    # script
    if script is None:
        # Use `co_filename` because `fun.__module__` is sometimes "__main__" and sometimes relative
        script = fun.__code__.co_filename
    script = Path(script)

    # Place launch script in same dir as script
    shutil.copy(Path(__file__).parent / "launch_xps.py", script.parent)

    # Find proj_dir (code to upload)
    if proj_dir is None:
        proj_dir = find_proj_dir(script)
    if len(proj_dir.relative_to(Path.home()).parts) <= 2:
        msg = f"The `proj_dir` ({proj_dir}) should be uploaded, but is too close to home dir."
        raise RuntimeError(msg)

    # Save to data_dir
    data_dir = data_root / proj_dir.stem / script.relative_to(proj_dir).stem
    data_dir = mk_data_dir(data_dir)

    def launch_script(py, cwd=False, string=False):
        args = [py, script.parent / "launch_xps.py", script.stem, fun.__name__, nCPU]
        args = [str(x) for x in args]
        if cwd:
            # PS: A well-crafted script should be independend of cwd,
            # but not necessarily {script.parent} (i.e. sys.path[0])
            args.insert(0, f"cd {cwd} &&")
        if string:
            args = " ".join(args)
        return args

    # Run locally
    if host in ["SUBPROCESS", None]:
        paths_xps = save(xps, data_dir, nBatch or 1)
        for xp in paths_xps:
            try:
                subprocess.run(launch_script(sys.executable) + [xp], check=True, cwd=Path.cwd())
            except subprocess.CalledProcessError:
                raise

    # Run remotely -- largely an exercise in path management
    else:
        # Host alias "globbing"
        if host.endswith("*"):
            host = uplink.resolve_host_glob(host)

        remote = uplink.Uplink(host)

        # Save xps -- partitioned (for node distribution)
        if "hpc.intra.norceresearch" in host:
            if nBatch is None:
                nBatch = 55
            nBatch = min(1000, nBatch)  # formal queue limit
            if nCPU is None:
                nCPU = 64
        elif nBatch is None:
                nBatch = 1
        paths_xps = save(xps, data_dir, nBatch)

        # data_root_on_remote
        if data_root_on_remote is None:
            if "hpc.intra.norceresearch" in host:
                data_root_on_remote = "${USERWORK}"
            else:
                data_root_on_remote = "${HOME}/data"
        # Evaluate
        # PS: not strictly necessary since ${some_envar} should work when later invoked
        # by cmd/rsync, but seems more robust and future-proof (for complex commands)
        data_root_on_remote = remote.cmd("echo " + data_root_on_remote).stdout.splitlines()[0]

        remote_dir = Path(data_root_on_remote) / data_dir.relative_to(data_root)
        paths_xps = [remote_dir / xp.relative_to(data_dir) for xp in paths_xps]

        # cwd -- (try to make!) such that the relative path of the script is same as locally
        try:
            cwd = Path.cwd().relative_to(proj_dir)
        except ValueError:
            print("Warning: The cwd is outside of project path ⇒ do not rely on it.")
            cwd = Path(".")
        finally:
            cwd = remote_dir / proj_dir.stem / cwd
        script = remote_dir / proj_dir.stem / script.relative_to(proj_dir)

        with remote.sym_sync(remote_dir, data_dir, proj_dir):
            # Install (potentially outdated) deps (from lockfile)
            # PS: Pre-install `uv` using `wget -qO- https://astral.sh/uv/install.sh | sh`
            venv = f"~/.cache/venvs/{proj_dir.stem}"
            remote.cmd(
                f"command cd {remote_dir / proj_dir.stem}; UV_PROJECT_ENVIRONMENT={venv} uv sync",
                capture_output=False,  # simply print
            )
            py = f"{venv}/bin/python"

            if "hpc.intra.norceresearch" in host:
                # Run on NORCE HPC cluster with SLURM queueing system
                cmd = launch_script(py, cwd=cwd, string=True)
                submit_and_monitor_slurm(remote, cmd, nCPU, remote_dir, script, paths_xps)

            else:
                # Run (`launch_xps.py` uses `mp` ⇒ no point parallelising this loop)
                for xp in paths_xps:
                    remote.cmd(launch_script(py, cwd=cwd) + [xp], capture_output=False)
    return data_dir
