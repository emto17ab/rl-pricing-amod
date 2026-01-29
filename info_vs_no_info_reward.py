import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup file paths
files = {
    'mode0_info': 'data/mode0_info.csv',
    'mode0_no_info': 'data/mode0_no_info.csv',
    'mode1_info': 'data/mode1_info.csv',
    'mode1_no_info': 'data/mode1_no_info.csv',
    'mode2_info': 'data/mode2_info.csv',
    'mode2_no_info': 'data/mode2_no_info.csv'
}

# 2. Load the datasets
dfs = {k: pd.read_csv(v) for k, v in files.items()}

# 3. Parameters for the plot
threshold = 5000  # Exclude first 5,000 steps
scale = 1000.0    # Scale axes to thousands
window = 300      # Smoothing window size
c_info = '#0072B2'    # Blue (Colorblind-friendly)
c_no_info = '#D55E00' # Vermillion

# Define labels and titles
modes = [
    {"title": "(a) Rebalancing Policy", "k_in": "mode0_info", "k_no": "mode0_no_info"},
    {"title": "(b) Pricing Policy",     "k_in": "mode1_info", "k_no": "mode1_no_info"},
    {"title": "(c) Joint Policy Control", "k_in": "mode2_info", "k_no": "mode2_no_info"}
]

# 4. Create the Plot
# figsize=(4.5, 9) is optimized for a single column in a two-column LaTeX paper
fig, axes = plt.subplots(3, 1, figsize=(4.5, 9), sharex=True)

for i, ax in enumerate(axes):
    m = modes[i]
    
    # Process both experimental groups (Info vs No Info)
    for key, color, label, linestyle in [
        (m['k_in'], c_info, 'Shared Info', '-'), 
        (m['k_no'], c_no_info, 'No Shared Info', '-')
    ]:
        df = dfs[key]
        
        # Identify the reward column (ignoring __MIN/__MAX columns)
        reward_col = [c for c in df.columns if 'total_reward' in c and '__' not in c][0]
        
        # NOTE: Apply smoothing FIRST on the raw data, then filter.
        # This prevents "jumps" or artifacts at the start of the plot.
        df_processed = df.copy()
        df_processed['reward_smooth'] = df_processed[reward_col].rolling(window=window, min_periods=1).mean()
        
        # Filter for the steps we want to show
        plot_df = df_processed[df_processed['Step'] >= threshold]
        
        # Plot with scaling applied to both axes
        ax.plot(
            plot_df['Step'] / scale, 
            plot_df['reward_smooth'] / scale, 
            color=color, 
            label=label, 
            linestyle=linestyle, 
            linewidth=1.5
        )

    # Styling the subplot
    ax.set_title(m['title'], loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel("Total Reward ($10^3$)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Tick formatting: Outward facing ticks on both axes
    ax.tick_params(direction='out', which='both', top=False, right=False)
    
    # Legend only on the top subplot to reduce clutter
    if i == 0:
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

# Common X-axis label
axes[-1].set_xlabel("Training Episodes ($10^3$)", fontsize=10)

# Overall Figure Title
fig.suptitle("Convergence Dynamics: With vs. Without Competitor Price Visibility", 
             fontsize=13, y=0.97)

# Adjust layout to fit titles and labels
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the plot
plt.savefig('reward_curves_final.png', dpi=300, bbox_inches='tight')
plt.show()