#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_nyc_man_south_fleet_experiment_v2[1-8]"
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=800MB]"
#BSUB -W 72:00
#BSUB -o logs/dual_nyc_man_south_fleet_experiment_v2_cars%I_%J.out
#BSUB -e logs/dual_nyc_man_south_fleet_experiment_v2_cars%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Map job index to vehicle fleet size
case $LSB_JOBINDEX in
    1)
        CARS=450
        ;;
    2)
        CARS=850
        ;;
    3)
        CARS=1050
        ;;
    4)
        CARS=1250
        ;;
    5)
        CARS=1450
        ;;
    6)
        CARS=1650
        ;;
    7)
        CARS=1850
        ;;
    8)
        CARS=2050
        ;;
esac

# Use mode 2 (joint control)
MODE=2

python main_a2c_multi_agent.py --reward_scalar 2000 --critic_warmup_episodes 50 --mode $MODE --city "nyc_man_south" --q_lr 0.0006 --p_lr 0.0002 --actor_clip 1000 --critic_clip 1000 --max_episodes 100000 --use_od_prices --no_share_info --total_vehicles $CARS --checkpoint_path dual_nyc_man_south_cars_${CARS}_v2_mode${MODE}
