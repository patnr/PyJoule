#!/usr/bin/env bash
# The contents of this file within braces get interpolated by python f-strings.
# Double braces {{}} escape this (become single braces after interpolation).

#SBATCH --job-name={script.stem}      # Job name
#SBATCH --account=energytech          # Unnecessary it seems
#SBATCH --partition=comp              # Type of nodes?
#SBATCH --qos=normal                  # Only one available I think
#SBATCH --nice=1000                   # High value ⇒ low priority in queue
#SBATCH --array=0-{len(paths_xps)-1}  # list of job/batch indices
#SBATCH --output=output/%a            # StdOut (separate files per array task)
#SBATCH --error=error/%a              # StdErr
#SBATCH --time=01:00:00               # Max runtime (HH:MM:SS)
#SBATCH --mem-per-cpu=60M             # Max memory (per array task)
#SBATCH --cpus-per-task={nCPU}        # Max CPUs (per array task)

# NOTE:
# - options `--ntasks` and `--nodes` seem relevant for MPI jobs,
#   not the embarrasingly parallelisable jobs that we handle
# - try `--requeue --max-requeue=3` if venv not found (even though nodes share file system)
# Also see https://documentation.sigma2.no/software/userinstallsw/conda.html

set -eu -o pipefail # NB: no -f (need filename expansion/globbing)
module --quiet reset  # Reset the modules to the system default


DATASETS=({remote_dir}/xps/*)
INPUT=${{DATASETS[$SLURM_ARRAY_TASK_ID]}}
{cmd} "$INPUT" 
