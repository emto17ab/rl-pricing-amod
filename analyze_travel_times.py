import json
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_dataset(file_path, city_name):
    print(f"Loading {city_name} from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Analyze Demand
    demands = data.get('demand', [])
    print(f"Total demand records: {len(demands)}")
    
    # Filter for 7-8 PM (1140 to 1200 minutes)
    # Assuming time_stamp is in minutes for demand
    demand_window = [d for d in demands if 1140 <= d.get('time_stamp', -1) < 1200]
    
    travel_times = []
    prices = []
    weights = []
    weighted_travel_times = []
    total_demand_in_window = 0
    
    for d in demand_window:
        dem = d.get('demand', 0)
        tt = d.get('travel_time', 0)
        p = d.get('price', 0)
        
        travel_times.append(tt)
        prices.append(p)
        weights.append(dem)
        
        weighted_travel_times.append(tt * dem)
        total_demand_in_window += dem
        
    print(f"Total Demand Volume (7-8 PM): {total_demand_in_window}")
    
    avg_travel_time = np.average(travel_times, weights=weights) if travel_times else 0
    weighted_avg_travel_time = sum(weighted_travel_times) / total_demand_in_window if total_demand_in_window > 0 else 0
    avg_price = np.average(prices, weights=weights) if prices else 0
    
    print(f"Average Travel Time (7-8 PM): {avg_travel_time:.2f}")
    print(f"Weighted Average Travel Time (7-8 PM): {weighted_avg_travel_time:.2f}")
    print(f"Weighted Average Price (7-8 PM): {avg_price:.2f}")
    
    # Analyze Rebalancing Time
    # rebTime is usually a list in the JSON based on previous grep
    # We need to check if it's a list or dict in the JSON directly.
    # The grep showed 'for item in data["rebTime"]', so it's a list.
    reb_times_data = data.get('rebTime', [])
    print(f"Total rebTime records: {len(reb_times_data)}")
    
    # Check structure of first item
    if reb_times_data:
        print(f"Sample rebTime item: {reb_times_data[0]}")
        
    # Filter for 7-8 PM
    # If time_stamp is 0-23 (hours), we want 19.
    # If time_stamp is minutes, we want 1140-1200.
    # Based on '23' seen in file, it's likely hours.
    
    reb_window = []
    for r in reb_times_data:
        ts = r.get('time_stamp')
        # Check if ts is likely hours or minutes
        if ts is not None:
            if ts == 19: # Hours
                reb_window.append(r.get('reb_time'))
            elif 1140 <= ts < 1200: # Minutes
                reb_window.append(r.get('reb_time'))
                
    avg_reb_time = np.mean(reb_window) if reb_window else 0
    print(f"Average Rebalancing Time (7-8 PM): {avg_reb_time:.2f}")
    print(f"Number of rebalancing entries found for 7-8 PM: {len(reb_window)}")

    return {
        'travel_times': travel_times,
        'prices': prices,
        'weights': weights,
        'reb_times': reb_window,
        'weighted_avg_tt': weighted_avg_travel_time,
        'avg_reb': avg_reb_time
    }

nyc_path = '/work3/s233791/rl-pricing-amod/data/scenario_nyc_manhattan.json'
sf_path = '/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json'

print("--- NYC Manhattan ---")
nyc_stats = analyze_dataset(nyc_path, "NYC Manhattan")
print("\n--- San Francisco ---")
sf_stats = analyze_dataset(sf_path, "San Francisco")

# Plotting
plt.figure(figsize=(18, 6))

# Travel Time Histogram
plt.subplot(1, 3, 1)
plt.hist(nyc_stats['travel_times'], weights=nyc_stats['weights'], bins=30, alpha=0.5, label='NYC', density=True)
plt.hist(sf_stats['travel_times'], weights=sf_stats['weights'], bins=30, alpha=0.5, label='SF', density=True)
plt.title('Travel Time Distribution (7-8 PM)')
plt.xlabel('Travel Time (min)')
plt.ylabel('Density')
plt.legend()

# Rebalancing Time Histogram
plt.subplot(1, 3, 2)
if nyc_stats['reb_times']:
    plt.hist(nyc_stats['reb_times'], bins=30, alpha=0.5, label='NYC', density=True)
if sf_stats['reb_times']:
    plt.hist(sf_stats['reb_times'], bins=30, alpha=0.5, label='SF', density=True)
plt.title('Rebalancing Time Distribution (7-8 PM)')
plt.xlabel('Rebalancing Time (min)')
plt.ylabel('Density')
plt.legend()

# Price Histogram
plt.subplot(1, 3, 3)
plt.hist(nyc_stats['prices'], weights=nyc_stats['weights'], bins=30, alpha=0.5, label='NYC', density=True)
plt.hist(sf_stats['prices'], weights=sf_stats['weights'], bins=30, alpha=0.5, label='SF', density=True)
plt.title('Price Distribution (7-8 PM)')
plt.xlabel('Price')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig('travel_reb_time_price_comparison.png')
print("\nPlot saved to travel_reb_time_price_comparison.png")
