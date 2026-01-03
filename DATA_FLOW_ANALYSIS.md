# Comprehensive Data Flow Analysis
**Multi-Agent RL for Pricing and Rebalancing in AMoD Systems**

Date: January 2, 2026
Analysis Type: Training Flow with use_od_prices=True

---

## Executive Summary

✅ **Overall Status**: Data flow is **CORRECT** for all three modes (0, 1, 2)
✅ **Shape Consistency**: All tensor shapes align properly through the pipeline
✅ **Model Architecture**: GNN actor/critic work as intended
⚠️ **Note**: Initial price loading creates identical prices for both agents (potential design choice to verify)

---

## 1. JSON Data Loading

### 1.1 File Structure
```json
{
  "tripAttr": [[origin, dest, time, demand, price], ...],
  "topology_graph": [{"i": src, "j": dst}, ...],
  ...
}
```

### 1.2 Initialization (`Scenario.__init__`)
- **Trip Attributes**: List of `[i, j, t, d, p]` tuples
- **Regions**: Set of unique origin/destination nodes (nregion=10 for San Francisco)
- **Demand Matrix**: `demand[(i,j)][t]` → scalar demand value
- **Base Prices**: `agent_price[agent_id][(i,j)][t]` → scalar price

**⚠️ Finding**: Both agents initialized with **identical** base prices from JSON. Is this intended?

---

## 2. Environment Reset & Observation Structure

### 2.1 Reset Returns (`env.reset()`)
Returns: `agent_obs = {agent_id: (acc, time, dacc, demand)}`

### 2.2 Observation Components
```python
obs = (acc, time, dacc, demand)
```

#### `acc` - Vehicle Accumulator
- Structure: `acc[region][time]` → scalar (number of vehicles)
- Example: `acc[5][0] = 20` (20 vehicles in region 5 at time 0)
- Initialized from: `G.nodes[n][f'accInit_agent{agent_id}']`

#### `time` - Current Timestep
- Structure: scalar integer
- Range: `[0, tf]` where `tf=60` (1 hour in minutes)

#### `dacc` - Delta Accumulator
- Structure: `dacc[region][time]` → scalar (future vehicle changes)
- Updated by: `pax_step()` and `reb_step()`
- Tracks vehicles in transit

#### `demand` - Trip Demand
- Structure: `demand[(i,j)][t]` → scalar demand
- Shared across agents (system-level demand)

### 2.3 Important Flow Detail
**CRITICAL**: After `reset()`, `match_step_simple()` MUST be called before parsing observations.
- `reset()` only initializes `acc[region][0]`
- `match_step_simple()` updates `acc[region][t+1]` which parser needs
- Parser accesses `acc[n][time+1]` (not `time`)

---

## 3. GNNParser - Observation → Model Input

### 3.1 Parser Initialization
```python
parser = GNNParser(
    env=env,
    T=6,  # Look-ahead horizon
    scale_factor=0.01,  # Feature scaling
    agent_id=0,
    json_file="data/scenario_san_francisco.json",
    use_od_prices=True  # KEY PARAMETER
)
```

### 3.2 Feature Construction (with use_od_prices=True)

#### Step 1: Temporal Vehicle Features
```python
# Current availability at t+1
current_avb = [acc[n][time+1] * scale_factor for n in regions]
# Shape: [1, 1, nregion]

# Future availability t+2 to t+T
future_avb = [[(acc[n][time+1] + dacc[n][t]) * scale_factor 
               for n in regions] 
              for t in range(time+1, time+T+1)]
# Shape: [1, T, nregion] = [1, 6, nregion]
```

#### Step 2: Queue Length
```python
queue_length = [len(env.agent_queue[agent_id][n]) * scale_factor 
                for n in regions]
# Shape: [1, 1, nregion]
```

#### Step 3: Current Demand (Outgoing)
```python
current_demand = [sum([demand[i,j][time] * scale_factor 
                       for j in regions]) 
                  for i in regions]
# Shape: [1, 1, nregion]
# Note: Aggregates outgoing demand from each region
```

#### Step 4: OD Price Matrices (KEY FEATURE)
```python
# Own agent's OD prices
own_price_od = [[agent_price[agent_id][(i,j)].get(time, 0) * scale_factor
                 for j in regions]
                for i in regions]
# Shape: [1, nregion, nregion]

# Competitor's OD prices
competitor_price_od = [[agent_price[opponent_id][(i,j)].get(time, 0) * scale_factor
                        for j in regions]
                       for i in regions]
# Shape: [1, nregion, nregion]
```

**Key Insight**: OD price matrices preserve spatial structure - each origin gets a full vector of destination prices.

#### Step 5: Concatenation & Reshaping
```python
# Concatenate along dim=1 (time dimension)
x = torch.cat([
    current_avb,        # [1, 1, nregion]
    future_avb,         # [1, 6, nregion]
    queue_length,       # [1, 1, nregion]
    current_demand,     # [1, 1, nregion]
    own_price_od,       # [1, nregion, nregion]
    competitor_price_od # [1, nregion, nregion]
], dim=1)
# Shape: [1, 1+6+1+1+nregion+nregion, nregion]
#      = [1, 9+2*nregion, nregion]
#      = [1, 29, 10] for San Francisco

# Reshape and transpose
x = x.squeeze(0).view(9+2*nregion, nregion).T
# Final shape: [nregion, 9+2*nregion] = [10, 29]
```

### 3.3 Final GNN Data Object
```python
data = Data(x, edge_index)
# data.x shape: [nregion, num_features] = [10, 29]
# data.edge_index shape: [2, num_edges] = [2, 38]
```

**Feature Breakdown** (San Francisco, use_od_prices=True):
- Position 0: Current availability
- Positions 1-6: Future availability (T=6 steps)
- Position 7: Queue length
- Position 8: Current outgoing demand
- Positions 9-18: Own OD prices (10 destinations per origin)
- Positions 19-28: Competitor OD prices (10 destinations per origin)
- **TOTAL**: 29 features per node

### 3.4 Alternative: use_od_prices=False
```python
# Aggregated prices (sum over destinations)
own_price = [sum([agent_price[agent_id][(i,j)][time] * scale_factor 
                  for j in regions]) 
             for i in regions]
# Shape: [1, 1, nregion]

competitor_price = [sum([agent_price[opponent_id][(i,j)][time] * scale_factor 
                        for j in regions]) 
                   for i in regions]
# Shape: [1, 1, nregion]

# Final shape: [nregion, 11] = [10, 11]
# Features: 1 + 6 + 1 + 1 + 1 + 1 = 11
```

---

## 4. A2C Model Architecture

### 4.1 Initialization
```python
model = A2C(
    env=env,
    input_size=29,  # CRITICAL: Must match parser output!
    device=torch.device('cpu'),
    hidden_size=256,
    T=6,
    mode=mode,  # 0, 1, or 2
    gamma=0.99,
    p_lr=0.0005,  # Actor learning rate
    q_lr=0.001,   # Critic learning rate
    actor_clip=10000,
    critic_clip=10000,
    scale_factor=0.01,
    agent_id=0,
    json_file="data/scenario_san_francisco.json",
    use_od_prices=True,
    reward_scale=1000.0,
    entropy_coef_max=0.0,
    entropy_coef_min=0.0,
    entropy_decay_rate=0.0
)
```

### 4.2 Input Size Calculation (CRITICAL)
```python
# In main_a2c_multi_agent.py
if args.use_od_prices:
    input_size = args.look_ahead + 3 + 2 * env.nregion
    # = 6 + 3 + 2*10 = 29 ✓
else:
    input_size = args.look_ahead + 5
    # = 6 + 5 = 11 ✓
```

**✅ Verified**: Input size calculation matches parser output for both cases.

---

## 5. Actor Network - Action Generation

### 5.1 Architecture (GNNActor)
```python
# GNN Convolution
out = F.relu(self.conv1(data.x, data.edge_index))  # [nregion, hidden_size]
x = out + data.x  # Residual connection

# Reshape for per-node processing
x = x.reshape(-1, act_dim, in_channels)  # [1, nregion, hidden_size]

# MLP layers
x = F.leaky_relu(self.lin1(x))  # [1, nregion, hidden_size]
x = F.leaky_relu(self.lin2(x))  # [1, nregion, hidden_size]

# Concentration parameters
x = F.softplus(self.lin3(x)) + min_conc  # [1, nregion, output_dim]
```

### 5.2 Mode-Specific Processing

#### Mode 0: Rebalancing Only (Dirichlet)
```python
# Concentration shape: [1, nregion]
concentration = x.squeeze(-1)  # No multiplier

# Sampling
m = Dirichlet(concentration)
action = m.rsample()  # [1, nregion]
log_prob = m.log_prob(action)  # [1] - JOINT distribution
action = action.squeeze(0)  # [nregion]

# Properties:
# - action.sum() ≈ 1.0 (probability distribution)
# - Single log_prob for entire action vector
```

**Output Shapes**:
- `action`: `[nregion]` - Rebalancing distribution
- `concentration`: `[1, nregion]` - Dirichlet parameters
- `log_prob`: `[1]` - Scalar (joint probability)

#### Mode 1: Pricing Only (Beta)
```python
# Concentration shape: [1, nregion, 2]
concentration = x * 30  # 30x multiplier for higher values

# Sampling
m_o = Beta(concentration[:,:,0], concentration[:,:,1])
action_o = m_o.rsample()  # [1, nregion]
action_o = torch.clamp(action_o, min=0.2, max=2.0)  # Price scalar range

# Log prob: SUM over independent distributions
log_prob = m_o.log_prob(action_o).sum(dim=-1)  # [1]
action = action_o.squeeze(0)  # [nregion]

# Properties:
# - action in [0.2, 2.0] (price scalars)
# - nregion independent Beta distributions
```

**Output Shapes**:
- `action`: `[nregion]` - Price scalars per region
- `concentration`: `[1, nregion, 2]` - (alpha, beta) pairs
- `log_prob`: `[1]` - Sum of independent log probs

#### Mode 2: Both Pricing & Rebalancing
```python
# Concentration shape: [1, nregion, 3]
beta_alpha = x[:, :, 0:1] * 30  # Pricing alpha
beta_beta = x[:, :, 1:2] * 30   # Pricing beta
dirichlet = x[:, :, 2:3]        # Rebalancing (no multiplier)
concentration = torch.cat([beta_alpha, beta_beta, dirichlet], dim=-1)

# Pricing (Beta)
m_o = Beta(concentration[:,:,0], concentration[:,:,1])
action_o = m_o.rsample()  # [1, nregion]
action_o = torch.clamp(action_o, min=0.2, max=2.0)

# Rebalancing (Dirichlet)
m_reb = Dirichlet(concentration[:,:,-1])
action_reb = m_reb.rsample()  # [1, nregion]

# Combined action and log prob
action = torch.stack([action_o.squeeze(0), action_reb.squeeze(0)], dim=-1)
log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_reb.log_prob(action_reb)

# Properties:
# - action[:, 0] in [0.2, 2.0] (price scalars)
# - action[:, 1].sum() ≈ 1.0 (rebalancing distribution)
```

**Output Shapes**:
- `action`: `[nregion, 2]` - [price, rebalancing] per region
- `concentration`: `[1, nregion, 3]` - [alpha, beta, dirichlet]
- `log_prob`: `[1]` - Sum of all log probs

### 5.3 Concentration Multipliers (Important!)
- **Beta (pricing)**: 30x multiplier → higher concentration → less exploration
- **Dirichlet (rebalancing)**: 1x (no multiplier) → more exploration
- **Reasoning**: Prices need more stability, rebalancing needs more flexibility

---

## 6. Critic Network - Value Estimation

### 6.1 Architecture (GNNCritic)
```python
# Same GNN processing as actor
out = F.relu(self.conv1(data.x, data.edge_index))
x = out + data.x

# Global mean pooling across all nodes
x = x.reshape(-1, act_dim, in_channels)
x = torch.mean(x, dim=1)  # [1, in_channels]

# MLP to scalar value
x = F.leaky_relu(self.lin1(x))
x = F.leaky_relu(self.lin2(x))
x = self.lin3(x)  # [1, 1]
```

**Output Shape**: `[1]` - Scalar state value estimate

---

## 7. Training Step - Gradient Flow

### 7.1 Action Selection During Training
```python
# In A2C.select_action()
state = self.parse_obs(obs).to(device)

# Forward passes
action, log_prob, concentration, entropy = self.actor(state, deterministic=False)
value = self.critic(state)

# Save for training
self.saved_actions.append(SavedAction(log_prob, value))
self.entropies.append(entropy)
```

### 7.2 Returns Calculation
```python
# Backward through episode
R = 0
returns = []
for r in rewards[::-1]:
    R = r + gamma * R
    returns.insert(0, R)

returns = torch.tensor(returns) / reward_scale
```

### 7.3 Advantage Calculation
```python
values = torch.stack([sa.value for sa in saved_actions])
advantages = returns - values.detach()
```

### 7.4 Loss Computation

#### Actor Loss (Policy Gradient)
```python
policy_loss = []
for saved_action, advantage in zip(saved_actions, advantages):
    policy_loss.append(-saved_action.log_prob * advantage)

actor_loss = torch.stack(policy_loss).mean()

# Entropy regularization (if enabled)
if entropy_coef > 0:
    entropy_loss = -entropy_coef * torch.stack(entropies).mean()
    actor_loss = actor_loss + entropy_loss
```

#### Critic Loss (TD Error)
```python
value_loss = []
for saved_action, R in zip(saved_actions, returns):
    value_loss.append(F.smooth_l1_loss(saved_action.value, R))

critic_loss = torch.stack(value_loss).mean()
```

### 7.5 Backward Pass
```python
# Critic update
optimizer_critic.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_clip)
optimizer_critic.step()

# Actor update (if not in warmup)
if episode >= critic_warmup_episodes:
    optimizer_actor.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_clip)
    optimizer_actor.step()
```

---

## 8. Shape Verification Summary

### Mode 0: Rebalancing Only
| Component | Shape | Notes |
|-----------|-------|-------|
| Parser output | `[10, 29]` | 29 features per region |
| Actor action | `[10]` | Dirichlet distribution |
| Actor concentration | `[1, 10]` | Single param per region |
| Actor log_prob | `[1]` | Joint distribution |
| Critic value | `[1]` | Scalar estimate |
| select_action output | `[10]` | Flattened for env |

### Mode 1: Pricing Only
| Component | Shape | Notes |
|-----------|-------|-------|
| Parser output | `[10, 29]` | 29 features per region |
| Actor action | `[10]` | Price scalars |
| Actor concentration | `[1, 10, 2]` | (alpha, beta) pairs |
| Actor log_prob | `[1]` | Sum of independent |
| Critic value | `[1]` | Scalar estimate |
| select_action output | `[10]` | Flattened for env |

### Mode 2: Both Pricing & Rebalancing
| Component | Shape | Notes |
|-----------|-------|-------|
| Parser output | `[10, 29]` | 29 features per region |
| Actor action | `[10, 2]` | [price, reb] per region |
| Actor concentration | `[1, 10, 3]` | [alpha, beta, dirichlet] |
| Actor log_prob | `[1]` | Sum of all distributions |
| Critic value | `[1]` | Scalar estimate |
| select_action output | `[10, 2]` | Kept as 2D for env |

✅ **All shapes verified and consistent!**

---

## 9. Key Findings & Observations

### ✅ Correct Implementations
1. **Parser correctly handles OD prices** via clever reshaping (.squeeze().view().T)
2. **Input size calculation matches parser output** (29 with OD prices, 11 without)
3. **Mode-specific action distributions** work as intended
4. **Concentration multipliers** differentiate price/reb exploration
5. **Log probability handling** correctly sums independent distributions

### ⚠️ Design Considerations
1. **Initial Prices**: Both agents start with identical prices from JSON
   - Is competitive differentiation only learned through RL?
   - Consider: Different initial price policies for agents?

2. **OD Price Representation**: 
   - Current: Full nregion×nregion matrix per agent (20 features)
   - Alternative: Could use demand-weighted aggregation to reduce dimensionality
   - Trade-off: Spatial detail vs. model complexity

3. **Demand Features**:
   - Only uses outgoing demand per region (aggregated over destinations)
   - Could include: incoming demand, OD-specific demand
   - Current design: Simpler, demand already influences prices via choice model

4. **Queue Length Scaling**:
   - Multiplied by 0.01 like other features
   - Could be normalized differently (e.g., per available vehicle)

### 📊 Hyperparameter Insights
1. **Concentration Multipliers**:
   - Beta (30x): Produces tight distributions around learned prices
   - Dirichlet (1x): Allows more rebalancing exploration
   - Well-justified: Prices need stability, rebalancing needs flexibility

2. **Reward Scaling** (1000.0):
   - Critical for gradient magnitudes
   - Normalized returns prevent value explosion

3. **Gradient Clipping** (10000):
   - Very high threshold
   - May allow large updates - monitor for instability

---

## 10. Recommendations

### For Current System
1. ✅ **No immediate bugs found** - system works as designed
2. Consider logging concentration values to monitor exploration
3. Track price divergence between agents over training
4. Monitor gradient norms to verify clipping threshold

### For Future Enhancements
1. **Experiment with OD price representation**:
   - Try demand-weighted aggregation
   - Compare full matrix vs. reduced features

2. **Agent Initialization**:
   - Test different initial price policies per agent
   - Consider asymmetric initialization for faster differentiation

3. **Feature Engineering**:
   - Add time-of-day features (currently implicit in demand)
   - Include historical demand/price trends
   - Add spatial features (distance-weighted neighborhoods)

4. **Architecture**:
   - Experiment with attention mechanisms for OD prices
   - Try deeper GNN layers for spatial reasoning
   - Test different aggregation methods in critic

---

## Conclusion

**The data flow is CORRECT and CONSISTENT across all three modes.** The implementation shows solid engineering:
- Proper shape handling through the pipeline
- Mode-specific distributions well-designed
- Gradient flow correctly implemented
- No blocking bugs found

The system is **ready for training** with current architecture. Future improvements should focus on feature engineering and initialization strategies rather than fixing bugs.
