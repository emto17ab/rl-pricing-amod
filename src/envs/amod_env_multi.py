from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import subprocess
import os
import networkx as nx
from src.misc.utils import mat2str
from src.misc.helper_functions import demand_update
from src.envs.structures import generate_passenger
from copy import deepcopy
import json
import random

class AMoD:
    # initialization
    # updated to take scenario and beta (cost for rebalancing) as input
    def __init__(self, scenario, mode, beta, jitter, max_wait, choice_price_mult, seed, loss_aversion, fix_agent):
        # Setting the scenario
        self.scenario = deepcopy(scenario)

        # Setting the mode of the simulation
        self.mode = mode  # Mode of rebalancing (0:manul, 1:pricing, 2:both. default 1)
        self.jitter = jitter # Jitter for zero demand

        # Setting the maximum passenger waiting time
        self.max_wait = max_wait # Maximum passenger waiting time
        
        # Setting which agent to fix (0=fix agent 0, 1=fix agent 1, 2=no fixing)
        self.fix_agent = fix_agent
        
        # Add loss aversion parameter for unprofitable trip penalty
        self.loss_aversion = loss_aversion  # Multiplier for loss penalty (λ)
        
        # Track unprofitable trips for logging
        self.agent_unprofitable_trips = {agent_id: 0 for agent_id in [0, 1]}

        # Setting up the road graph
        self.G = scenario.G  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'

        # Set trip times and rebalancing times
        self.demandTime = self.scenario.demandTime # Nested dictionary, Travel time for demand, key: (i,j) - (origin, destination), key: t - time, with value of travel time
        self.rebTime = self.scenario.rebTime # Nested dictionary, Travel time for rebalancing, key: (i,j) - (origin, destination), key: t - time, with value of travel time

        # Set the simulation time, the time step and the final time
        self.time = 0  # current time
        self.tf = scenario.tf  # final time
        self.tstep = scenario.tstep

        # Set available agents
        self.agents = [0, 1] # List of available agents

        # Set the regions in the simulation
        self.region = list(self.G)  # set of regions

        # Demand of Nodes
        self.demand = defaultdict(dict)  # Nested dictionary, Demand at node, key: (i,j) - (origin, destination), key: t - time, with value of demand

        # Multi-agent passenger tracking: passenger[agent_id][region][time] = [passenger_objects] 
        self.agent_passenger = {agent_id: dict() for agent_id in self.agents} 
        
        # Multi-agent passenger queue: queue[agent_id][region] = [passenger_objects]
        self.agent_queue = {agent_id: defaultdict(list) for agent_id in self.agents}

        # Initialize passenger tracking and demand for each agent and region
        for agent_id in self.agents:
            for i in self.region:
                self.agent_passenger[agent_id][i] = defaultdict(list)

        # Multi-agent pricing: price[agent_id][(i,j)][t] = price
        self.agent_price = {agent_id: defaultdict(dict) for agent_id in self.agents} # Set the price for each agent, origin-destination pair and time

        # Total arrivals for each agent: arrivals[agent_id] = total number of added passengers
        self.agent_arrivals = {agent_id: 0 for agent_id in self.agents}

        # Initialize demand and pricing from scenario data
        # For each trip attribute (origin i, destination j, time t, demand d, base price p):
        # - Store O-D specific demand for matching
        # - Set initial prices for both agents (they can adjust independently later)
        # - Accumulate departure demand (total passengers leaving region i at time t)
        # - Accumulate arrival demand (total passengers arriving at region i at time t+travel_time)
        for i, j, t, d, p in scenario.tripAttr:
            self.demand[i, j][t] = d
            # Initialize price for each agent
            for agent_id in self.agents:
                self.agent_price[agent_id][i, j][t] = p

        # Multi-agent vehicle tracking
        # acc[agent_id][region][time] = number of available vehicles for agent in region at time
        self.agent_acc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        # dacc[agent_id][region][time] = number of vehicles arriving at region at time for agent
        self.agent_dacc = {agent_id: defaultdict(dict) for agent_id in self.agents}

        # Multi-agent rebalancing flows: rebFlow[agent_id][(i,j)][t] = number of rebalancing vehicles
        self.agent_rebFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_rebFlow_ori = {agent_id: defaultdict(dict) for agent_id in self.agents}

        # Multi-agent passenger flows: paxFlow[agent_id][(i,j)][t] = number of vehicles with passengers
        self.agent_paxFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}

        # Multi-agent passenger wait lists: paxWait[agent_id][(i,j)] = [waiting passengers for this agent]
        self.agent_paxWait = {agent_id: defaultdict(list) for agent_id in self.agents}

        # Initialize graph structure and flow tracking
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions

        # Build complete edge list including self-loops (staying in same region)
        for i in self.G:
            self.edges.append((i, i))  # self-loop for staying in region
            for e in self.G.out_edges(i):
                self.edges.append(e)  # edges to adjacent regions
        self.edges = list(set(self.edges))

        # Count outgoing edges per region (for action space dimensionality)
        self.nedge = [len(self.G.out_edges(n))+1 for n in self.region]

        # Initialize rebalancing flows for each agent and edge
        # For each edge in the graph:
        # - Set travel time from rebTime lookup
        # - Initialize empty vehicle flow tracking (rebFlow) for both agents
        # - Initialize backup flow tracking (rebFlow_ori) for both agents
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            for agent_id in self.agents:
                self.agent_rebFlow[agent_id][i, j] = defaultdict(float)
                self.agent_rebFlow_ori[agent_id][i, j] = defaultdict(float)

        # Initialize passenger flows and waiting lists for each agent and O-D pair
        # For each origin-destination pair with demand:
        # - Initialize occupied vehicle flow tracking (paxFlow) per agent
        # - Initialize passenger waiting list (paxWait) per agent
        for i, j in self.demand:
            for agent_id in self.agents:
                self.agent_paxFlow[agent_id][i, j] = defaultdict(float)
                self.agent_paxWait[agent_id][i, j] = []

        # Initialize vehicle counts for each agent and region
        # Use agent-specific vehicle distributions from scenario (already split and distributed)
        # Store initial distribution for fixed agent rebalancing
        self.agent_initial_acc = {agent_id: {} for agent_id in self.agents}
        for agent_id in self.agents:
            for n in self.region:
                # Use agent-specific accInit values from scenario
                acc_key = f'accInit_agent{agent_id}'
                initial_count = self.G.nodes[n][acc_key]
                self.agent_acc[agent_id][n][0] = initial_count
                self.agent_initial_acc[agent_id][n] = initial_count  # Store for fixed agent
                self.agent_dacc[agent_id][n] = defaultdict(float)


        # scenario.tstep: number of steps as one timestep
        self.beta = beta * scenario.tstep # Cost for rebalancing per time unit in simulation time

        # Multi-agent demand tracking: demand[agent_id][(i,j)][t] = total demand for this agent
        self.agent_demand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        
        # Initialize agent demand for each O-D pair
        for agent_id in self.agents:
            for i, j in self.demand:
                self.agent_demand[agent_id][i, j] = defaultdict(float)

        # Multi-agent demand tracking: servedDemand[agent_id][(i,j)][t] = passengers served
        self.agent_servedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        # Multi-agent unserved tracking: unservedDemand[agent_id][(i,j)][t] = passengers rejected
        self.agent_unservedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}


        # Initialize served and unserved demand tracking for each agent and O-D pair
        for agent_id in self.agents:
            for i, j in self.demand:
                self.agent_servedDemand[agent_id][i, j] = defaultdict(float)
                self.agent_unservedDemand[agent_id][i, j] = defaultdict(float)

        self.N = len(self.region)  # total number of cells

        # Multi-agent info tracking: info[agent_id] = {metrics}
        # Tracks performance metrics for each agent independently:
        # - revenue: total revenue from served passengers
        # - served_demand: number of passengers successfully served
        # - unserved_demand: number of passengers rejected or timed out
        # - rebalancing_cost: cost of moving empty vehicles
        # - operating_cost: total operational costs
        # - served_waiting: cumulative waiting time of served passengers
        self.agent_info = {agent_id: dict.fromkeys(['revenue', 'served_demand', 'unserved_demand',
                                    'rebalancing_cost', 'operating_cost', 'served_waiting', 
                                    'true_profit', 'adjusted_profit'], 0) 
                    for agent_id in self.agents}
        
        # System-level info tracking (not agent-specific)
        # Tracks metrics at the system level:
        # - rejected_demand: number of passengers who rejected both agents via choice model
        # - total_demand: total demand generated in the system
        # - rejection_rate: ratio of rejected demand to total demand
        self.system_info = dict.fromkeys(['rejected_demand', 'total_demand', 'rejection_rate'], 0)

        # Multi-agent external rewards (operating costs): ext_reward[agent_id] = np.array of external rewards per region
        self.ext_reward_agents = {a: np.zeros(self.nregion) for a in [0, 1]}

        # Multi-agent observations: obs[agent_id] = (acc, time, dacc, demand)
        # Each agent observes:
        # - acc: their own vehicle distribution across regions
        # - time: current simulation time (shared)
        # - dacc: their own vehicles arriving at regions
        # - demand: passenger demand (shared, agents compete for it)
        self.agent_obs = {agent_id: (self.agent_acc[agent_id], self.time, 
                            self.agent_dacc[agent_id], self.demand) 
            for agent_id in self.agents}
    
        self.choice_price_mult = choice_price_mult

        self.seed = seed
        
        # Trip assignment tracking: stores detailed data for each trip
        self.trip_assignments = []
    
    def match_step_simple(self, price = None):
        t = self.time
        paxreward = {0: 0, 1: 0}
        
        # Reset violation tracking for this timestep
        for agent_id in self.agents:
            self.agent_unprofitable_trips[agent_id] = 0
        
        # Reset agent_info for this timestep
        for agent_id in self.agents:
            for key in self.agent_info[agent_id]:
                self.agent_info[agent_id][key] = 0
        
        # Reset system_info for this timestep
        for key in self.system_info:
            self.system_info[key] = 0

        total_original_demand = 0
        total_rejected_demand = 0

        for n in self.region:
            # Update current queue
            for j in self.G[n]:
                d = self.demand[n, j][t]
                
                # Apply node-level price scaling if provided by the agent
                # price[agent_id] is a list of scalars, one per region/node
                # price[agent_id][n] is the price scalar for node n (from Beta distribution)
                # Skip price update if price is None or all zeros (first step of episode). Then it just applies baseline prices.
                # Mode 0 (rebalancing only) should NOT modify prices at all
                if self.mode != 0 and price is not None and np.sum([np.sum(price[a]) for a in self.agents]) != 0:
                    for agent_id in self.agents:
                        # Get baseline price for this O-D pair
                        baseline_price = self.agent_price[agent_id][n, j][t]
                        
                        # For fixed agent, always use price scalar of 0.5 (keeps base price)
                        # Otherwise use the learned price scalar
                        if self.fix_agent == agent_id:
                            price_scalar = 0.5
                        else:
                            # Apply node-level price scalar (from action_rl)
                            # price[agent_id][n] is a scalar between 0 and 1 from Beta distribution
                            # In mode 1: price[agent_id][n] is directly the scalar
                            # In mode 2: price[agent_id][n] is [price_scalar, reb_scalar], so we take [0]
                            price_scalar = price[agent_id][n]
                            if isinstance(price_scalar, (list, np.ndarray)):
                                price_scalar = price_scalar[0]
                        
                        # Calculate proposed price (multiply by 2 to allow range [0, 2×baseline])
                        p = 2 * baseline_price * price_scalar
                        
                        # Ensure absolute minimum price (avoid zero prices)
                        if p <= 1e-6:
                            p = self.jitter
                        
                        self.agent_price[agent_id][n, j][t] = p

                ####################### Choice Model Implementation #################
                d_original = d  # before applying choice model

                #--Choice Model--
                
                pr0 = self.agent_price[0][n, j][t]
                pr1 = self.agent_price[1][n, j][t]
                
                travel_time = self.demandTime[n, j][t]
                
                travel_time_in_hours = travel_time / 60
                U_reject = 0 
                
                exp_utilities = []
                labels = []

                wage = 25

                income_effect = 25 / wage

                # Compute utilities for all agents
                U_0 = 12.1 - 0.71 * wage * travel_time_in_hours - income_effect * self.choice_price_mult * pr0
                U_1 = 12.1 - 0.71 * wage * travel_time_in_hours - income_effect * self.choice_price_mult * pr1
                
                # Always include both agents in the choice set
                # (Fixed agent will use base price due to scalar 0.5)
                exp_utilities.append(np.exp(U_0))
                labels.append("agent0")
                exp_utilities.append(np.exp(U_1))
                labels.append("agent1")
                
                # Always include reject option
                exp_utilities.append(np.exp(U_reject))
                labels.append("reject")

                Probabilities = np.array(exp_utilities) / np.sum(exp_utilities)
                labels_array = np.array(labels)

                d0 = d1 = dr = 0

                # Use choice model with appropriate choice set
                if d_original > 0:
                    for _ in range(d_original):
                        choice = np.random.choice(labels_array, p=Probabilities)
                        if choice == "agent0":
                            d0 += 1
                        elif choice == "agent1":
                            d1 += 1
                        elif choice == "reject":
                            dr += 1
                    
                    # Log trip assignment details
                    prob_dict = {}
                    utility_dict = {}
                    for idx, label in enumerate(labels):
                        prob_dict[label] = Probabilities[idx]
                        if label == "agent0":
                            utility_dict[label] = U_0
                        elif label == "agent1":
                            utility_dict[label] = U_1
                        elif label == "reject":
                            utility_dict[label] = U_reject
                    
                    self.trip_assignments.append({
                        'time': t,
                        'origin': n,
                        'destination': j,
                        'travel_time': travel_time,
                        'price_agent0': pr0,
                        'price_agent1': pr1,
                        'utility_agent0': U_0,
                        'utility_agent1': U_1,
                        'utility_reject': U_reject,
                        'prob_agent0': prob_dict.get("agent0", 0.0),
                        'prob_agent1': prob_dict.get("agent1", 0.0),
                        'prob_reject': prob_dict.get("reject", 0.0),
                        'demand_agent0': d0,
                        'demand_agent1': d1,
                        'demand_rejected': dr,
                        'total_demand': d_original
                    })

                self.agent_demand[0][(n, j)][t] += d0
                self.agent_demand[1][(n, j)][t] += d1

                pax0, self.agent_arrivals[0] = generate_passenger((n, j, t, d0, pr0), self.max_wait, self.agent_arrivals[0])
                pax1, self.agent_arrivals[1] = generate_passenger((n, j, t, d1, pr1), self.max_wait, self.agent_arrivals[1])

                self.agent_passenger[0][n][t].extend(pax0)
                self.agent_passenger[1][n][t].extend(pax1)

                random.Random(self.seed).shuffle(self.agent_passenger[0][n][t])
                random.Random(self.seed).shuffle(self.agent_passenger[1][n][t])

                total_original_demand += d_original
                total_rejected_demand += dr

                self.demand[n, j][t] = d0 + d1
            
            for agent_id in [0, 1]:
                accCurrent = self.agent_acc[agent_id][n][t]

                # Add new entering passengers to this agent's queue
                new_enterq = [pax for pax in self.agent_passenger[agent_id][n][t] if pax.enter()]
                queueCurrent = self.agent_queue[agent_id][n] + new_enterq
                self.agent_queue[agent_id][n] = queueCurrent

                matched_leave_index = []

                for i, pax in enumerate(queueCurrent):
                    if accCurrent != 0:
                        accept = pax.match(t)
                        if accept:
                            # Store matched passenger index for removal from queue
                            matched_leave_index.append(i)
                            
                            # Remove an available vehicle
                            accCurrent -= 1
                            
                            # Add passenger to agent passenger flow
                            arr_t = t + self.demandTime[pax.origin, pax.destination][t]
                            self.agent_paxFlow[agent_id][pax.origin, pax.destination][arr_t] += 1
                            
                            # Set agents wait time
                            wait_t = pax.wait_time
                            self.agent_paxWait[agent_id][pax.origin, pax.destination].append(wait_t)

                            # Add passenger to agent's arriving passengers at destination
                            self.agent_dacc[agent_id][pax.destination][arr_t] += 1

                            # Update served demand tracking
                            self.agent_servedDemand[agent_id][pax.origin, pax.destination][t] += 1

                            trip_cost = self.demandTime[pax.origin, pax.destination][t] * self.beta
                            trip_revenue = pax.price
                            
                            # Calculate profitability-aware reward
                            base_reward = trip_revenue - trip_cost
                            
                            # Penalty for unprofitable trips (loss aversion)
                            if base_reward < 0:
                                self.agent_unprofitable_trips[agent_id] += 1
                                # Apply quadratic penalty: λ × (loss)²
                                loss_penalty = self.loss_aversion * (base_reward ** 2)
                                adjusted_reward = base_reward - loss_penalty
                            else:
                                adjusted_reward = base_reward
                            
                            paxreward[agent_id] += adjusted_reward

                            # Update the operating costs
                            self.ext_reward_agents[agent_id][n] += max(0, trip_cost)

                            # Track true metrics separately for monitoring
                            self.agent_info[agent_id]['revenue'] += trip_revenue
                            self.agent_info[agent_id]['served_demand'] += 1
                            self.agent_info[agent_id]['operating_cost'] += trip_cost
                            self.agent_info[agent_id]['served_waiting'] += wait_t
                            self.agent_info[agent_id]['true_profit'] += base_reward
                            self.agent_info[agent_id]['adjusted_profit'] += adjusted_reward
                        else:
                            if pax.unmatched_update():
                                matched_leave_index.append(i)
                                self.agent_unservedDemand[agent_id][pax.origin, pax.destination][t] += 1
                                self.agent_info[agent_id]['unserved_demand'] += 1
                    else:
                        if pax.unmatched_update():
                            matched_leave_index.append(i)
                            self.agent_unservedDemand[agent_id][pax.origin, pax.destination][t] += 1
                            self.agent_info[agent_id]['unserved_demand'] += 1

                # Removes matched or leaving passengers from the queue
                self.agent_queue[agent_id][n] = [
                    queueCurrent[i] for i in range(len(queueCurrent)) if i not in matched_leave_index
                ]
                # Update available vehicles after matching
                self.agent_acc[agent_id][n][t+1] = accCurrent
            
        done = (self.tf == t+1)
        ext_done = [done]*self.nregion

        self.obs = {
            0: (self.agent_acc[0], self.time, self.agent_dacc[0], self.agent_demand[0]),
            1: (self.agent_acc[1], self.time, self.agent_dacc[1], self.agent_demand[1])
        }

        # Update system-level info
        self.system_info['rejected_demand'] = total_rejected_demand
        self.system_info['total_demand'] = total_original_demand
        self.system_info['rejection_rate'] = (
            total_rejected_demand / total_original_demand if total_original_demand > 0 else 0
        )

        # Add unprofitable trips count to agent info
        for agent_id in [0, 1]:
            self.agent_info[agent_id]['unprofitable_trips'] = self.agent_unprofitable_trips[agent_id]

        return self.obs, paxreward, done, self.agent_info, self.system_info, self.ext_reward_agents, ext_done

    def matching_update(self):
        """Update properties if there is no rebalancing after matching"""
        t = self.time
        # Update acc. Assuming arriving vehicle will only be availbe for the next timestamp.
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            for agent_id in [0, 1]:
                if (i, j) in self.agent_paxFlow[agent_id] and t in self.agent_paxFlow[agent_id][i, j]:
                    self.agent_acc[agent_id][j][t+1] += self.agent_paxFlow[agent_id][i, j][t]
        
        # For fixed agents, reset vehicle distribution to initial state
        if self.fix_agent in [0, 1]:
            fixed_agent_id = self.fix_agent
            for n in self.region:
                self.agent_acc[fixed_agent_id][n][t+1] = self.agent_initial_acc[fixed_agent_id][n]
        
        self.time += 1

    def reb_step(self, rebAction_agents):
        # Set the time
        t = self.time
        
        # Set the counter for rewards
        rebreward = {0: 0, 1: 0}

        # Set the counter for operating costs
        self.ext_reward_agents = {a: np.zeros(self.nregion) for a in [0, 1]}
    
        # Initialize the info_agents dictionary
        for agent_id in [0, 1]:
            self.agent_info[agent_id]['rebalancing_cost'] = 0

        # Loop through agents
        for agent_id in [0, 1]:
            
            rebAction = rebAction_agents[agent_id]
    
            # Loop through the edges for rebalancing
            for k in range(len(self.edges)):
                i, j = self.edges[k]

                # Update rebalancing actions and flows
                # Ensure rebalancing does not exceed available vehicles
                rebAction[k] = min(self.agent_acc[agent_id][i][t+1], rebAction[k])

                # Calculate rebalancing time
                reb_time = self.rebTime[i, j][t]

                # Set the inflow of vechiles for rebalancing
                self.agent_rebFlow[agent_id][i, j][t + reb_time] = rebAction[k]
                self.agent_rebFlow_ori[agent_id][i, j][t] = rebAction[k]
    
                # Update the vehicle counts based on rebalancing actions
                self.agent_acc[agent_id][i][t+1] -= rebAction[k]
                self.agent_dacc[agent_id][j][t + reb_time] += rebAction[k]

                # Calculate rebalancing costs for the agent
                rebalancing_cost = self.rebTime[i, j][t] * self.beta * rebAction[k]
                rebreward[agent_id] -= rebalancing_cost
                self.ext_reward_agents[agent_id][i] -= rebalancing_cost

                # Track rebalancing costs in agent_info
                self.agent_info[agent_id]['rebalancing_cost'] += rebalancing_cost
    
        # Vehicle arrivals from past rebalancing and passenger trips
        for agent_id in [0, 1]:
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (i, j) in self.agent_rebFlow[agent_id] and t in self.agent_rebFlow[agent_id][i, j]:
                    self.agent_acc[agent_id][j][t+1] += self.agent_rebFlow[agent_id][i, j][t]
                if (i, j) in self.agent_paxFlow[agent_id] and t in self.agent_paxFlow[agent_id][i, j]:
                    self.agent_acc[agent_id][j][t+1] += self.agent_paxFlow[agent_id][i, j][t]
        
        # For fixed agents, reset vehicle distribution to initial state
        if self.fix_agent in [0, 1]:
            fixed_agent_id = self.fix_agent
            for n in self.region:
                self.agent_acc[fixed_agent_id][n][t+1] = self.agent_initial_acc[fixed_agent_id][n]
    
        # Increment time step
        self.time += 1
    
        self.obs = {
            0: (self.agent_acc[0], self.time, self.agent_dacc[0], self.agent_demand[0]),
            1: (self.agent_acc[1], self.time, self.agent_dacc[1], self.agent_demand[1])
        }
    
        # Update rebalancing time on edges
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
    
        # Check if the episode is done
        done = (self.tf == t + 1)
        ext_done = [done] * self.nregion
    
        return self.obs, rebreward, done, self.agent_info, self.system_info, self.ext_reward_agents, ext_done

    def get_total_vehicles(self, agent_id=None):
        """
        Calculate total number of vehicles in the system at current time for each agent.
        Includes: available vehicles + vehicles with passengers + rebalancing vehicles
        
        Args:
            agent_id: If provided, return total for specific agent. If None, return dict with totals for all agents.
        
        Returns:
            If agent_id is None: dict with {agent_id: total_vehicles}
            If agent_id is provided: int with total vehicles for that agent
        """
        t = self.time
        
        if agent_id is not None:
            # Calculate total vehicles for the agent
            total = 0
            
            # Count available vehicles at all regions for CURRENT time
            for region in self.region:
                # Try current time first, then fallback to t+1
                if t in self.agent_acc[agent_id][region]:
                    total += self.agent_acc[agent_id][region][t]
                elif t+1 in self.agent_acc[agent_id][region]:
                    total += self.agent_acc[agent_id][region][t+1]
            
            # Count vehicles with passengers (all current and future arrivals)
            for (i, j), time_dict in self.agent_paxFlow[agent_id].items():
                for time_step, flow in time_dict.items():
                    if time_step >= t:  # Current and future arrivals (vehicles in transit)
                        total += flow
            
            # Count rebalancing vehicles (all current and future arrivals)
            for (i, j), time_dict in self.agent_rebFlow[agent_id].items():
                for time_step, flow in time_dict.items():
                    if time_step >= t:  # Current and future arrivals (vehicles in transit)
                        total += flow
            
            return total
        else:
            # Calculate totals for all agents
            totals = {}
            for agent_id in self.agents:
                # For fixed agents, just return the total from initial distribution
                if self.fix_agent == agent_id:
                    totals[agent_id] = sum(self.agent_initial_acc[agent_id].values())
                    continue
                
                # Calculate total for active (non-fixed) agent
                total = 0
                
                # Count available vehicles at all regions for CURRENT time
                for region in self.region:
                    # Try current time first, then fallback to t+1
                    if t in self.agent_acc[agent_id][region]:
                        total += self.agent_acc[agent_id][region][t]
                    elif t+1 in self.agent_acc[agent_id][region]:
                        total += self.agent_acc[agent_id][region][t+1]
                
                # Count vehicles with passengers (all current and future arrivals)
                for (i, j), time_dict in self.agent_paxFlow[agent_id].items():
                    for time_step, flow in time_dict.items():
                        if time_step >= t:  # Current and future arrivals (vehicles in transit)
                            total += flow
                
                # Count rebalancing vehicles (all current and future arrivals)
                for (i, j), time_dict in self.agent_rebFlow[agent_id].items():
                    for time_step, flow in time_dict.items():
                        if time_step >= t:  # Current and future arrivals (vehicles in transit)
                            total += flow
                
                totals[agent_id] = total
            
            return totals

    def get_initial_vehicles(self):
        """Get the initial number of vehicles in the system (total across both agents)"""
        return sum(
            self.G.nodes[n]['accInit_agent0'] + self.G.nodes[n]['accInit_agent1']
            for n in self.G.nodes
        )

    def get_trip_assignments(self):
        """Get and clear the trip assignments log"""
        trips = self.trip_assignments.copy()
        self.trip_assignments = []
        return trips
    
    def reset(self):
        """Reset the episode for multi-agent environment"""
        
        # Reset trip assignments
        self.trip_assignments = []
        
        # Reset multi-agent vehicle tracking
        self.agent_acc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_dacc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        
        # Reset multi-agent flow tracking
        self.agent_rebFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_rebFlow_ori = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_paxFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_paxWait = {agent_id: defaultdict(list) for agent_id in self.agents}
        
        # Reset multi-agent passenger tracking
        self.agent_passenger = {agent_id: dict() for agent_id in self.agents}
        self.agent_queue = {agent_id: defaultdict(list) for agent_id in self.agents}
        
        # Initialize passenger tracking for each agent and region
        for agent_id in self.agents:
            for i in self.region:
                self.agent_passenger[agent_id][i] = defaultdict(list)
        
        # Reset edge list
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        
        # Reset demand and pricing
        self.demand = defaultdict(dict)
        self.agent_demand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_price = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_arrivals = {agent_id: 0 for agent_id in self.agents}
        
        # Get new random demand
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        
        # Trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            # Set initial price for both agents
            for agent_id in self.agents:
                self.agent_price[agent_id][i, j][t] = p
                # Initialize agent demand
                if (i, j) not in self.agent_demand[agent_id]:
                    self.agent_demand[agent_id][i, j] = defaultdict(float)
            
            # Track region-level demand
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            self.regionDemand[i][t] += d
        
        # Reset time
        self.time = 0
        
        # Initialize flows for each agent and edge
        for i, j in self.G.edges:
            for agent_id in self.agents:
                self.agent_rebFlow[agent_id][i, j] = defaultdict(float)
                self.agent_rebFlow_ori[agent_id][i, j] = defaultdict(float)
                self.agent_paxFlow[agent_id][i, j] = defaultdict(float)
                self.agent_paxWait[agent_id][i, j] = []
        
        # Initialize vehicle counts for each agent and region
        # Use agent-specific vehicle distributions from scenario (same as __init__)
        for agent_id in self.agents:
            for n in self.G:
                # Use agent-specific accInit values from scenario
                acc_key = f'accInit_agent{agent_id}'
                self.agent_acc[agent_id][n][0] = self.G.nodes[n][acc_key]
                self.agent_dacc[agent_id][n] = defaultdict(float)
        
        # Initialize served and unserved demand tracking
        self.agent_servedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_unservedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        
        for agent_id in self.agents:
            for i, j in self.demand:
                self.agent_servedDemand[agent_id][i, j] = defaultdict(float)
                self.agent_unservedDemand[agent_id][i, j] = defaultdict(float)
        
        # Reset multi-agent info tracking
        self.agent_info = {agent_id: dict.fromkeys(['revenue', 'served_demand', 'unserved_demand',
                                    'rebalancing_cost', 'operating_cost', 'served_waiting', 
                                    'true_profit', 'adjusted_profit'], 0) 
                    for agent_id in self.agents}
        
        # Reset system-level info tracking
        self.system_info = dict.fromkeys(['rejected_demand', 'total_demand', 'rejection_rate'], 0)
        
        
        # Create observations for each agent
        self.agent_obs = {agent_id: (self.agent_acc[agent_id], self.time, 
                            self.agent_dacc[agent_id], self.demand) 
            for agent_id in self.agents}
        
        return self.agent_obs

    

class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio=None, supply_ratio=1,
                 trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.0, json_file=None, json_hr=19, json_tstep=3, varying_time=False, json_regions=None, impute=False):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distributionjson_tstep
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        if json_file == None:
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1*N2)
            self.G = self.G.to_directed()
            # Add self-loops to the graph for within-region trips
            self.G.add_edges_from([(i, i) for i in self.G.nodes])
            self.demandTime = defaultdict(dict)  # traveling time between nodes
            self.rebTime = defaultdict(dict)
            # Self-loops are now part of G.edges, no need to add them separately
            self.edges = list(self.G.edges)
            self.tstep = json_tstep
            for i, j in self.edges:
                for t in range(tf*2):
                    self.demandTime[i, j][t] = (
                        (abs(i//N1-j//N1) + abs(i % N1-j % N1))*grid_travel_time)
                    self.rebTime[i, j][t] = (
                        (abs(i//N1-j//N1) + abs(i % N1-j % N1))*grid_travel_time)

            # Total fleet = ninit vehicles per node
            total_fleet = ninit * len(self.G.nodes)
            
            # Split fleet between two agents (round down)
            fleet_per_agent = int(total_fleet // 2)
            
            # Distribute each agent's fleet evenly across nodes
            num_nodes = len(self.G.nodes)
            base_vehicles_per_node = fleet_per_agent // num_nodes
            remainder = fleet_per_agent % num_nodes
            
            # Create list of nodes and shuffle for random remainder assignment
            nodes_list = list(self.G.nodes)
            random.seed(sd)
            random.shuffle(nodes_list)
            
            # Assign vehicles to each node for both agents
            for idx, n in enumerate(nodes_list):
                vehicles_for_agent = base_vehicles_per_node + (1 if idx < remainder else 0)
                self.G.nodes[n]['accInit_agent0'] = vehicles_for_agent
                self.G.nodes[n]['accInit_agent1'] = vehicles_for_agent
            
            self.tf = tf
            self.demand_ratio = defaultdict(list)

            # demand mutiplier over time
            if demand_ratio == None or type(demand_ratio) == list or type(demand_ratio) == dict:
                for i, j in self.edges:
                    if type(demand_ratio) == list:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio)-1)), demand_ratio))+[demand_ratio[-1]]*tf
                    elif type(demand_ratio) == dict:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(0, tf+1, tf/(len(demand_ratio[i]) - 1)), demand_ratio[i]))+[demand_ratio[i][-1]]*tf
                    else:
                        self.demand_ratio[i, j] = [1]*(tf+tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio[i, j])-1)), demand_ratio[i, j]))+[1]*tf
                    else:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio['default'])-1)), demand_ratio['default']))+[1]*tf
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand()*2+1) * \
                        (self.demandTime[i, j][0]+1)
            if tripAttr != None:  # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand()  # randomly generated demand
        else:
            # Set the varying time (default False)
            self.varying_time = varying_time
            
            # Since we are in this branch, we are reading a json file
            self.is_json = True

            # Read json file
            with open(json_file, "r") as file:
                data = json.load(file)

            # Stores the time-step size (presumably in minutes) into self.tstep. This value is used when binning timestamps into discrete time indices.
            self.tstep = json_tstep

            # Number of latitude and longitude divisions in the grid
            self.N1 = data["nlat"]
            self.N2 = data["nlon"]

            #  Will hold aggregated demand per OD per time bin. 
            #  It's a defaultdict(dict): keys are OD tuples (o,d) and values will be dicts mapping time indices → demand volumes.
            self.demand_input = defaultdict(dict)

            # See if the data has regions specified
            self.json_regions = json_regions

            # Create a directed graph representing the regions and their connections.
            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif 'region' in data:
                self.G = nx.complete_graph(data['region'])
            else:
                self.G = nx.complete_graph(self.N1*self.N2)
            
            self.G = self.G.to_directed()
            # Add self-loops to the graph for within-region trips
            self.G.add_edges_from([(i, i) for i in self.G.nodes])

            # Will hold aggregated/averaged prices per OD per time bin (p[(o,d)][t])
            self.p = defaultdict(dict)

            # No randomness is added to demand input. Hence demand is fixed. If alpha = 0.2 demand_input will fluctuate within [0.8, 1.2] * demand_input 
            self.alpha = alpha

            # Creates stucture for travel time per OD per time bin (demandTime[(o,d)][t])
            self.demandTime = defaultdict(dict)

            # Creates structure for rebalancing time per OD per time bin (rebTime[(o,d)][t])
            self.rebTime = defaultdict(dict)

            # Multiply hour by minutes to get the starting time in minutes after midnight
            self.json_start = json_hr * 60

            # Sets the number of steps per episode (default 20). Hence for each time step we generate a demand, travel time, rebalancing time, and price profile.
            self.tf = tf

            # Self-loops are now part of G.edges, no need to add them separately
            self.edges = list(self.G.edges)

            # Sets the number of regions based on the graph's nodes (# of regions = # of nodes)
            self.nregion = len(self.G)

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(float)
                self.rebTime[i, j] = 1
           
            # Creates nregion × nregion numpy array of zeros (default) for time index t. The code will accumulate demand volumes into array cells [o,d]
            matrix_demand = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))

            # Similarly stores aggregated (volume-weighted) price sums for each time index.
            matrix_price_ori = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))
            # Loops over the data demand file
            for item in data["demand"]:
                # Sets the variables
                # t= time stamp (in minutes after midnight)
                # o= origin region
                # d= destination region
                # v= demand
                # tt= travel time (in minutes)
                # p= price (in dollars)
                t, o, d, v, tt, p = item["time_stamp"], item["origin"], item[
                    "destination"], item["demand"], item["travel_time"], item["price"]
                
                # If json_regions was provided and this OD pair is not in that list, the record is skipped.
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                
                # If this OD (o,d) has not been seen before, initialize three dicts for it:
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = defaultdict(
                        float), defaultdict(float), defaultdict(float)

                # Set ALL demand, price, and traveling time for OD. 

                # First a time index is calculated by subtracting the starting time (self.json_start) from the timestamp t
                # and dividing by the time step size (json_tstep). This bins the timestamp into discrete time indices.

                # The demand is incremented for specific OD and time index by the demand volume v, scaled by a demand_ratio factor.
                self.demand_input[o, d][(
                    t-self.json_start)//json_tstep] += v*demand_ratio

                # The price p is accumulated in a volume-weighted manner (p*v) for the same OD and time index. 
                # This is not just summing prices: it's building a demand-weighted sum. Later we divide by total demand to get average price.
                self.p[o, d][(t-self.json_start) //
                             json_tstep] += p*v*demand_ratio

                # Same as price. Accumulates travel times weighted by demand volume. 
                # Also divide by json_tstep (normalizing since demand is aggregated over that bin length) to get average travel time per unit time.
                self.demandTime[o, d][(t-self.json_start) //
                                      json_tstep] += tt*v*demand_ratio/json_tstep


                # At time bin k, matrix_demand[k][o,d] = total number of demand from o to d
                matrix_demand[(t-self.json_start) //
                                      json_tstep][o,d] += v*demand_ratio

                # At time bin k, matrix_price_ori[k][o,d] = total price (weighted by demand) from o to d
                # Later divided by matrix_demand to get average observed price per OD/time bin             
                matrix_price_ori[(t-self.json_start) //
                                      json_tstep][o,d] += p*v*demand_ratio

            
            # Price and traveling time will be averaged by demand after
            # Loop over all Edges in the graph
            for o, d in self.edges:
                # Loop over all time indices from 0 to tf*2
                for t in range(0, tf*2):

                    # See if there was any demand recorded for this OD at this time index
                    if t in self.demand_input[o, d]:
                        # Divide the accumulated p * v sum by total v to get volume-weighted average price at that bin
                        self.p[o, d][t] /= self.demand_input[o, d][t]

                        # Similarly compute average travel time and ensure it is an integer of at least 1 minute
                        self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] = max(
                            int(round(self.demandTime[o, d][t])), 1)

                        # Compute the matrix-based average price for that time slice
                        matrix_price_ori[t][o,d] /= matrix_demand[t][o,d]
                    
                    # If not set it to zero
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demandTime[o, d][t] = 0

            # Creates a matrix matrix_reb (nregion × nregion) initialized to zeros to store baseline rebalancing times per OD.
            matrix_reb = np.zeros((self.nregion,self.nregion))

            # Loops over the rebalancing time data in the JSON file
            for item in data["rebTime"]:

                # Extracts the relevant fields
                # hr= the hour associated with the rebalancing time
                # o= origin region
                # d= destination region
                # rt= rebalancing time (in minutes)
                hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]

                # Skips the record if it doesn't belong to json_regions (if that filter exists)
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue

                # If varying time is true (default False
                # Each JSON rebTime record with hour hr is mapped to the time bins that cover that hour (a sliding window). 
                # Effect: rebalancing time is written only into the bins that correspond to the actual hour hr in the JSON 
                # (so rebTime varies across the timeline according to the timestamps in the file). 
                if varying_time:
                    t0 = int((hr*60 - self.json_start)//json_tstep)
                    t1 = int((hr*60 + 60 - self.json_start)//json_tstep)
                    for t in range(t0, t1):
                        self.rebTime[o, d][t] = max(
                            int(round(rt/json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, tf+1):
                            self.rebTime[o, d][t] = max(
                                int(round(rt/json_tstep)), 1)
                            matrix_reb[o,d] = rt/json_tstep
            
            # KNN regression for each time step
            if impute:
                # Create dictionary to store the regresors
                knn = defaultdict(lambda: KNeighborsRegressor(n_neighbors=3))
                # Loop over time steps
                for t in matrix_price_ori.keys():
                    reb = matrix_reb
                    price = matrix_price_ori[t]
                    X = []
                    y = []
                    # Loop over all region pairs to construct training set
                    for i in range(self.nregion):
                        for j in range(self.nregion):
                            # if the price is not zero use it as training data for the regressor
                            if price[i,j] != 0:
                                X.append(reb[i,j])
                                y.append(price[i,j])
                    X_train = np.array(X).reshape(-1, 1)
                    y_train = np.array(y)

                    # Fit the regressor for that time point
                    knn[t].fit(X_train, y_train)

                # Test point
                for o, d in self.edges:
                    for t in range(0, tf*2):
                        # If there is no price for specific time point, and a regressor exists for that time point, use it 
                        # to impute the price at that time point.s
                        if self.p[o,d][t]==0 and t in knn.keys():
                            
                            knn_regressor = knn[t]

                            X_test = np.array([[matrix_reb[o,d]]])

                            # Predict the value for the test point
                            y_pred = knn_regressor.predict(X_test)[0]
                            self.p[o,d][t] = float(y_pred)

            # Initial vehicle distribution
            # Data contains hour and total number of vehicles in network
            for item in data["totalAcc"]:
                hr, acc = item["hour"], item["acc"]
                if hr == json_hr+int(round(json_tstep/2*tf/60)):
                    # Total fleet with supply ratio applied
                    total_fleet = supply_ratio * acc
                    
                    # Split fleet between two agents (round down to ensure integers)
                    fleet_per_agent = int(total_fleet // 2)
                    
                    # Distribute each agent's fleet evenly across nodes
                    num_nodes = len(self.G)
                    base_vehicles_per_node = fleet_per_agent // num_nodes
                    remainder = fleet_per_agent % num_nodes
                    
                    # Create list of nodes and shuffle for random remainder assignment
                    nodes_list = list(self.G.nodes)
                    random.seed(sd)  # Use scenario seed for reproducibility
                    random.shuffle(nodes_list)
                    
                    # Assign vehicles to each node for both agents
                    for idx, n in enumerate(nodes_list):
                        # Each agent gets base amount, plus 1 extra if within remainder
                        vehicles_for_agent = base_vehicles_per_node + (1 if idx < remainder else 0)
                        self.G.nodes[n]['accInit_agent0'] = vehicles_for_agent
                        self.G.nodes[n]['accInit_agent1'] = vehicles_for_agent


            self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        # assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(
                            self.demand_input[i, j][t])
                        price[i, j][t] = self.p[i, j][t]
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:
            self.static_demand = dict()
            region_rand = (np.random.rand(len(self.G))*self.alpha *
                            2+1-self.alpha)  # multiplyer of demand
            if type(self.demand_input) in [float, int, list, np.array]:

                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:  # demand in the format of each region
                    self.region_demand = region_rand * \
                        np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    prob = np.array(
                        [np.exp(-self.rebTime[i, j][0]*self.trip_length_preference) for j in J])
                    prob = prob/sum(prob)
                    for idx in range(len(J)):
                        # allocation of demand to OD pairs
                        self.static_demand[i, J[idx]
                                            ] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = self.demand_input[i, j] if (
                        i, j) in self.demand_input else self.demand_input['default']

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception(
                    "demand_input should be number, array-like, or dictionary-like values")

            # generating demand and prices
            if self.fix_price:
                p = self.p
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(
                        self.static_demand[i, j]*self.demand_ratio[i, j][t])
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = min(3, np.random.exponential(
                            2)+1)*self.demandTime[i, j][t]
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr