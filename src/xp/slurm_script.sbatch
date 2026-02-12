#!/usr/bin/env bash
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

# NB: don't include f option, which disables filename expansion (globbing)
set -eu -o pipefail

# The contents of this file within braces get interpolated by python f-strings.
DATASETS=({data_dir_remote}/xps/*)
# Double braces {{}} escape this (become single braces after interpolation).
INPUT=${{DATASETS[$SLURM_ARRAY_TASK_ID]}}

{launch_script} "$INPUT" 
