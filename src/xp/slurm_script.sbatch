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
#SBATCH --time=00:05:00               # Maximum runtime (HH:MM:SS)
#SBATCH --partition=comp              # Type of nodes?
#SBATCH --ntasks=1                    # Only useful with `srun` ?
#SBATCH --nodes=1                     # Only useful with `srun` ?
#SBATCH --cpus-per-task=255           # max(batch_size, actual_cpu_count)
#SBATCH --array=0-{len(paths_xps)-1}  # job/batch indices
# try `--requeue --max-requeue=3` if venv not found (even though nodes share file system)


set -eu -o pipefail # NB: no -f (need filename expansion/globbing)
module --quiet reset  # Reset the modules to the system default


DATASETS=({remote_dir}/xps/*)
INPUT=${{DATASETS[$SLURM_ARRAY_TASK_ID]}}
{cmd} "$INPUT" 
