# Graph-RL for joint rebalancing and pricing in AMoD
Official implementation of 'Learning joint rebalancing and dynamic pricing policies for Autonomous Mobility-on-Demand'. The paper is curently under review and the link to the paper will be added as soon as it becomes publicly available.

## Prerequisites

You will need to have a working IBM CPLEX installation. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students).

The code is built with python 3.9.18. Different python version might result in error of running the code. To install all required dependencies, run
```
pip install -r requirements.txt
```
It is recommended to create a virtual environment before installing the packages. If you are using Anaconda or miniconda, run
```
conda create --name {env_name} python==3.9
conda activaye {env_name}
pip install -r requirements.txt
```

## Contents

* `src/algos/sac.py`: PyTorch implementation of Graph Networks for A2C.
* `src/algos/IQL.py`: PyTorch implementation of Graph Networks for IQL.
* `src/algos/CQL.py`: PyTorch implementation of Graph Networks for CQL.
* `src/algos/reb_flow_solver.py`: thin wrapper around CPLEX formulation of the Minimum Rebalancing Cost problem.
* `src/envs/amod_env.py`: AMoD simulator.
* `src/cplex_mod/`: CPLEX formulation of Rebalancing and Matching problems.
* `src/misc/`: helper functions.
* `data/`: json files for different cities in the simulator.

## Experiments

To train an agent online, `main_SAC.py` accepts the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test          activates agent evaluation mode (default: False)
    --city          city to train on (default: san_francisco)
    --mode          training policy. 0:rebalancing, 1:pricing, 2:joint. (default: 1)
    --max_episodes  number of episodes within each trial (default: 10000)
    --max_steps     number of steps per episode (default: T=20)
    --hidden_size   node embedding dimension (default: 256)
    --clip          vector magnitude used to clip gradient
    --no-cuda       disables CUDA training (default: True, i.e. run on CPU)
    --directory     defines directory where to log files (default: saved_files)
    --batch_size      defines the batch size (default: 100)
    --alpha           entropy coefficient (default: 0.3)
    --p_lr            Actor learning reate (default 1e-3)
    --q_lr            Critic learning rate (default: 1e-3)
    --checkpoint_path name of the model checkpoint file to load from/save to. The checkpoint file will be saved to/load from path: ckpt/{checkpoint_path}.
    --city            which city to train on 
    --rew_scale       reward scaling (default 0.01, for SF 0.1)
    --critic_version  defined critic version to use (default: 4)
    
simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --json_tsetp    (default: 3)
```
To train an agent offline by CQL, `main_CQL.py` accepts the following arguments additional to `main_SAC.py`. Attention: Do not change the default setting for `mode` in offline training. The offline training only accpets training for the joint policy.
```bash
    
model arguments:
    --memory_path     city name for which the offline dataset is saved (default: nyc_brooklyn)
    --min_q_weight    conservative coefficient (default: 5)
    --samples_buffer  number of samples to take from the dataset (max 10000)
    --lagrange_tresh  lagrange treshhold tau for autonamtic tuning of eta 
    --st              whether to standardize data (default: False)
    --sc              whether to scale (max-min) the data (default: Fasle)     
```

To train an agent offline by IQL, `main_IQL.py` accepts the foolowing arguments additional to `main_SAC.py`. Attention: Do not change the default setting for `mode`.
```bash
    
model arguments:
    --memory_path     city name for which the offline dataset is saved (default: nyc_brooklyn)
    --quantile        quantile of expetile regression(default: 0.8)
    --temperature     weight of advantange(default: 3.0)
    --samples_buffer  number of samples to take from the dataset (max 10000)
```

***Important***: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```
### Training and simulating an agent
Before starting training, remeber to create a folder with the same name as specified by the `directory` arguement (default saved_files) to store intermediate and log files, and a folder named `ckpt` for the checkpoint weight files. All checkpoints will be saved under `ckpt/`.
#### Training and simulating an agent online
1. To train an agent online for the joint policy:
```
python main_SAC.py --city {city_name} --mode 2
```
2. To evaluate a pretrained agent run the following:
```
python main_SAC.py --city {city_name} --test True --checkpoint_path {checkpoint_name}
```
When evaluating the checkpoint, there is no need to include `ckpt/` in the checkpoint_name. Same rule applies to the following offline evaluation.
#### Training and simulating an agent offline
***Important***: Before training agent offline, download offline training dataset from [this link](https://www.dropbox.com/scl/fi/daeuygfz5z2tlmvh4foia/Replaymemories.zip?rlkey=nr16hfc3bk29741w2mq33f0zb&st=ljjvkdja&dl=0), unzip it, and put it to the root of the repository.
1. To train an agent offline by CQL with heuristic, 75%, or 90% datasets:
```
python main_CQL.py --city {city_name} --memory_path {dataset_name}
```
e.g. to train an agent offline on the heuristic dataset on NYC brooklyn: 
```
python main_CQL.py --city nyc_brooklyn --memory_path nyc_brooklyn_heuristic
```
Similarly, to train an agent offline IQL for heuristic, 75%, or 90% datasets:
```
python main_IQL.py --city {city_name} --memory_path {dataset_name}
```
e.g. to train an agent offline on the heuristic dataset on NYC brooklyn: 
```
python main_IQL.py --city nyc_brooklyn --memory_path nyc_brooklyn_iql_heuristic
```
2. To evaluate a pretrained agent run the following:
```
python main_CQL.py --city {city_name} --test True --checkpoint_path {checkpoint_name}
```
or
```
python main_IQL.py --city {city_name} --test True --checkpoint_path {checkpoint_name}
```

## Credits
This work was conducted as a joint effort with [Carolin Schmidt*](https://scholar.google.com/citations?user=-0zHX8oAAAAJ&hl=en), [Daniele Gammelli'](https://danielegammelli.github.io/), and [Filipe Rodrigues*](http://fprodrigues.com/), at Technical University of Denmark* and Stanford University'.

----------
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at xinli831@mit.edu.