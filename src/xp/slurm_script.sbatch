#!/usr/bin/env bash
# The contents of this file within braces get interpolated by python f-strings.
# Double braces {{}} escape this (become single braces after interpolation).
#
# Also see https://documentation.sigma2.no/software/userinstallsw/conda.html
#SBATCH --qos=normal                  # Only one available I think
#SBATCH --account=energytech          # Unnecessary it seems
#SBATCH --job-name={script.stem}      # Job name
#SBATCH --output=output/%a     # StdOut (separate files per array task)
#SBATCH --error=error/%a       # StdErr
#SBATCH --time=00:01:15               # Maximum runtime (HH:MM:SS)
#SBATCH --partition=comp              # Type of nodes?
#SBATCH --cpus-per-task=58           # should be min(batch_size, actual_cpu_count) # TODO
#SBATCH --array=0-{len(paths_xps)-1}  # job/batch indices
#SBATCH --mem-per-cpu=60M
# try `--requeue --max-requeue=3` if venv not found (even though nodes share file system)

# TODO: sbatch: warning: can't run 1 processes on 12 nodes, setting nnodes to 1

set -eu -o pipefail # NB: no -f (need filename expansion/globbing)
module --quiet reset  # Reset the modules to the system default


DATASETS=({remote_dir}/xps/*)
INPUT=${{DATASETS[$SLURM_ARRAY_TASK_ID]}}
{cmd} "$INPUT" 
