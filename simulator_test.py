import numpy as np
from tqdm import trange
from src.envs.amod_env import Scenario, AMoD
from src.algos.reb_flow_solver import solveRebFlow


# Define calibrated simulation parameters
demand_ratio = {'san_francisco': 2, 'washington_dc': 4.2, 'chicago': 1.8, 'nyc_man_north': 1.8, 'nyc_man_middle': 1.8,
                'nyc_man_south': 1.8, 'nyc_brooklyn': 9, 'porto': 4, 'rome': 1.8, 'shenzhen_baoan': 2.5,
                'shenzhen_downtown_west': 2.5, 'shenzhen_downtown_east': 3, 'shenzhen_north': 3
                }
json_hr = {'san_francisco': 19, 'washington_dc': 19, 'chicago': 19, 'nyc_man_north': 19, 'nyc_man_middle': 19,
           'nyc_man_south': 19, 'nyc_brooklyn': 19, 'porto': 8, 'rome': 8, 'shenzhen_baoan': 8,
           'shenzhen_downtown_west': 8, 'shenzhen_downtown_east': 8, 'shenzhen_north': 8
           }
beta = {'san_francisco': 0.2, 'washington_dc': 0.5, 'chicago': 0.5, 'nyc_man_north': 0.5, 'nyc_man_middle': 0.5,
        'nyc_man_south': 0.5, 'nyc_brooklyn': 0.5, 'porto': 0.1, 'rome': 0.1, 'shenzhen_baoan': 0.5,
        'shenzhen_downtown_west': 0.5, 'shenzhen_downtown_east': 0.5, 'shenzhen_north': 0.5}

test_tstep = {'san_francisco': 3,
              'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3}

# Define AMoD Simulator Environment
seed = 42
json_tstep = 3
city = 'washington_dc'
max_steps = 20
cplexpath = "C:/Program Files/IBM/ILOG/CPLEX_Studio201/opl/bin/x64_win64/"
scenario = Scenario(json_file=f"data/scenario_{city}.json",
                    demand_ratio=demand_ratio[city], json_hr=json_hr[city], sd=seed, json_tstep=json_tstep, tf=max_steps)
env = AMoD(scenario, beta=beta[city])

d_t = False
# initialize episode-level book-keeping variables
episode_reward = 0
episode_served_demand = 0
episode_rebalancing_cost = 0
episode_ext_reward = np.zeros(env.nregion)
episode_ext_paxreward = np.zeros(env.nregion)
episode_ext_rebreward = np.zeros(env.nregion)

while not d_t:
    # take matching step
    o_t, paxreward, d_t, info, ext_paxreward, ext_done = env.match_step_simple()
    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
    totalAcc = sum([env.acc[i][0] for i in env.region])
    desiredAcc = {env.region[i]: 1/len(env.region)
                  for i in range(len(env.region))}
    # solve minimum rebalancing distance problem (Step 3 in paper)
    rebAction = solveRebFlow(
        env=env, res_path='metarl-amod', desiredAcc=desiredAcc, CPLEXPATH=cplexpath)
    # take action in environment
    o_t, rebreward, d_t, info, ext_rebreward, ext_done = env.reb_step(
        rebAction)

# Send current statistics to screen
queue = sum([len(env.queue[n]) for n in range(env.nregion)])
demand = sum([sum(list(v.values())[:max_steps])
             for _, v in env.demand.items()])
waiting = 0
for n in env.region:
    for t in env.passenger[n]:
        for pax in env.passenger[n][t]:
            waiting += pax.wait_time
waiting /= demand
print(
    f"Task {city} | Total demand: {demand} | Total vehicles: {totalAcc} | Served demand: {env.info['served_demand']} | Unserved demand: {env.info['unserved_demand']} | Average waiting time: {waiting:.2f} | Queue: {queue}| Reb cost: {env.info['rebalancing_cost']}")
