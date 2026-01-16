#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_agent_nyc_man_south_vehicle_split[1-4]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 72:00
#BSUB -o logs/dual_agent_nyc_man_south_vehicle_split_ratio%I_%J.out
#BSUB -e logs/dual_agent_nyc_man_south_vehicle_split_ratio%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Map job index to agent0_vehicle_ratio
case $LSB_JOBINDEX in
    1)
        RATIO=0.1
        ;;
    2)
        RATIO=0.2
        ;;
    3)
        RATIO=0.3
        ;;
    4)
        RATIO=0.4
        ;;
esac

# Always use mode 2
MODE=2

python main_a2c_multi_agent.py --reward_scalar 10000 --critic_warmup_episodes 50 --mode $MODE --city "nyc_man_south" --q_lr 0.0005 --p_lr 0.0003 --actor_clip 1 --critic_clip 1000 --max_episodes 1000000 --use_od_prices --agent0_vehicle_ratio $RATIO --checkpoint_path dual_agent_nyc_man_south_vehicle_split_ratio${RATIO}_mode${MODE}
