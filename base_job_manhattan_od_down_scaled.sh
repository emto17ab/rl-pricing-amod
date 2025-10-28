#!/bin/bash
#BSUB -q hpc
#BSUB -J diagnostics_run_clamped_logprob_v3
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/diagnostics_run_clamped_logprob_v3%J.out
#BSUB -e logs/diagnostics_run_clamped_logprob_v3%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 40105 --actor_clip 5 --critic_clip 5 --use_od_prices --q_lr 1e-4 --p_lr 1e-4 --city "nyc_manhattan" --checkpoint_path diagnostics_run_clamped_logprob_v3