#!/bin/bash
# Helper script to prepare and submit SLURM job array
# Automatically calculates array size from joblist.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if joblist exists
JOBLIST="joblist.txt"
if [[ ! -f "$JOBLIST" ]]; then
    echo "ERROR: $JOBLIST not found. Generating it now..."
    python generate_joblist.py
fi

# Count jobs
NJOBS=$(wc -l < "$JOBLIST")
echo "Found $NJOBS jobs in $JOBLIST"

# Load config to get max concurrent jobs
if command -v python3 &> /dev/null; then
    MAX_CONCURRENT=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('slurm', {}).get('max_concurrent_jobs', 20))
" 2>/dev/null || echo "20")
else
    MAX_CONCURRENT=20
fi

echo "Max concurrent jobs: $MAX_CONCURRENT"

# Create a temporary SLURM script with correct array size
TMP_SBATCH="submit_rl_auto.sbatch"
sed "s/--array=1-100%20/--array=1-${NJOBS}%${MAX_CONCURRENT}/" submit_rl.sbatch > "$TMP_SBATCH"

echo ""
echo "Submitting SLURM job array:"
echo "  Total jobs: $NJOBS"
echo "  Concurrent: $MAX_CONCURRENT"
echo "  Array: 1-${NJOBS}%${MAX_CONCURRENT}"
echo ""

# Submit the job
sbatch "$TMP_SBATCH"

echo ""
echo "Job submitted! Check status with: squeue -u \$USER"
echo "Monitor logs in: logs/"

