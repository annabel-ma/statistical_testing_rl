# RL Experiments - SLURM Setup

This directory contains scripts to run reinforcement learning experiments in parallel on SLURM.

## Initial Setup (One-time)

**1. Create virtual environment and install packages:**
```bash
cd rl_final_proj
./setup_venv.sh
```

This creates a `venv/` directory with all required packages using `uv`.

## Testing a Single Experiment

Before submitting all jobs, test with one experiment:

**Activate the virtual environment and run:**
```bash
cd rl_final_proj
source venv/bin/activate
python train_rl.py --task "Hopper-v5" --algorithm "SAC" --seed 0
```

This will:
- Train one agent (Hopper-v5 with SAC, seed 0)
- Save results to `rl_experiments/runs/` and `rl_experiments/models/`
- Append to `rl_experiments/final_eval_returns.csv`

**For a quick test with fewer timesteps**, you can temporarily modify `config.yaml`:
```yaml
timesteps_per_task:
  "Hopper-v5": 10000  # Reduced from 1000000 for quick test
```

## Setup

1. **Configure experiments (optional):**
   Edit `config.yaml` to customize tasks, algorithms, seeds, and other settings.
   You can also adjust `slurm.max_concurrent_jobs` to control parallelism.

2. **Generate the job list:**
   ```bash
   python generate_joblist.py
   ```
   This creates `joblist.txt` with all combinations of tasks, algorithms, and seeds from `config.yaml`.

3. **Submit the SLURM job array (EASY WAY):**
   ```bash
   chmod +x prepare_submit.sh
   ./prepare_submit.sh
   ```
   This automatically calculates the array size and submits with the correct parallelism settings.

   **OR manually:**
   ```bash
   # Check number of jobs
   wc -l joblist.txt
   # Edit submit_rl.sbatch to update --array=1-N%M where N=total jobs, M=concurrent
   sbatch submit_rl.sbatch
   ```

## Parallel Execution

**YES, this runs in parallel!** The SLURM job array (`--array=1-N%M`) means:
- **N** = total number of jobs (one per task/algorithm/seed combination)
- **M** = maximum concurrent jobs (default: 20)
- Jobs run independently and in parallel

For example, with 100 jobs and `%20`:
- 20 jobs start immediately
- As jobs finish, new ones start automatically
- All 100 jobs will complete (just 20 at a time)

Adjust `max_concurrent_jobs` in `config.yaml` to change parallelism.

## Configuration

All experiment settings are in `config.yaml`:
- **Tasks**: List of environments to run
- **Algorithms**: List of RL algorithms
- **Seeds**: List of random seeds
- **Timesteps**: Per-task training timesteps
- **Directories**: Output paths
- **Algorithm settings**: Algorithm-specific parameters

Default configuration:
- **Tasks**: Hopper-v5, Walker2d-v5, HalfCheetah-v5, Ant-v5, Humanoid-v5
- **Algorithms**: SAC, TD3, DDPG, PPO
- **Seeds**: 0-4 (5 seeds)
- **Total jobs**: 5 tasks × 4 algorithms × 5 seeds = 100 jobs

## Output Structure

All results are saved in `rl_experiments/`:
- `runs/`: Individual run JSON files (`{task}_{algo}_seed{seed}.json`)
- `models/`: Trained model files (`{task}_{algo}_seed{seed}.zip`)
- `final_eval_returns.csv`: Final evaluation results
- `learning_curves.csv`: Learning curves during training

## SLURM Settings

- **Time limit**: 48 hours per job
- **Memory**: 8GB per job
- **Concurrent jobs**: 20 (adjust `%20` in submit_rl.sbatch if needed)
- **Logs**: `logs/rl_{JOB_ID}_{ARRAY_ID}.out` and `.err`

## Notes

- Jobs will skip if results already exist (cached runs)
- Each job runs one (task, algorithm, seed) combination
- Results are automatically appended to CSV files
- Models are saved after training completes

