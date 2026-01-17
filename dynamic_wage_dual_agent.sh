#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_agent_nyc_man_south_dynamic_wage_mode2"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 72:00
#BSUB -o logs/dual_agent_nyc_man_south_dynamic_wage_mode2_%J.out
#BSUB -e logs/dual_agent_nyc_man_south_dynamic_wage_mode2_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Mode 2 only
MODE=2

python main_a2c_multi_agent.py --reward_scalar 10000 --critic_warmup_episodes 50 --mode $MODE --city "nyc_man_south" --q_lr 0.0005 --p_lr 0.0003 --actor_clip 1 --critic_clip 1000 --max_episodes 1000000 --use_od_prices --use_dynamic_wage_man_south --checkpoint_path dual_agent_nyc_man_south_dynamic_wage_mode${MODE}
