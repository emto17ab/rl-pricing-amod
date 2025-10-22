"""
Dataset Analysis: San Francisco vs NYC Brooklyn
Analyzes key characteristics of both datasets including:
- Total trips after demand ratio application
- Average trip times
- Average rebalancing times
- Demand patterns
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict

# Define calibrated simulation parameters from main_a2c_multi_agent.py
demand_ratio = {
    'san_francisco': 2, 
    'nyc_brooklyn': 9
}

json_hr = {
    'san_francisco': 19, 
    'nyc_brooklyn': 19
}

def analyze_dataset(city_name, json_file, demand_ratio_value, json_hr_value):
    """
    Analyze a single dataset and return statistics
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {city_name.upper()}")
    print(f"{'='*80}")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f, parse_constant=lambda x: float('nan') if x == 'NaN' else None)
    
    print(f"\nConfiguration:")
    print(f"  Demand Ratio: {demand_ratio_value}")
    print(f"  JSON Hour: {json_hr_value} (= {json_hr_value * 60} minutes)")
    print(f"  Number of Regions (nlat × nlon): {data['nlat']} × {data['nlon']} = {data['nlat'] * data['nlon']}")
    
    # ============================================================================
    # DEMAND ANALYSIS
    # ============================================================================
    print(f"\n{'-'*80}")
    print("DEMAND ANALYSIS")
    print(f"{'-'*80}")
    
    demand_data = data['demand']
    
    # Convert json_hr (hour) to minute range: hour 19 = minutes 1140-1199 (19:00-19:59)
    json_hr_start = json_hr_value * 60
    json_hr_end = json_hr_start + 59  # Full hour range
    
    # Filter demand for the ENTIRE hour (all minutes in hour 19)
    demand_filtered = [d for d in demand_data if json_hr_start <= d['time_stamp'] <= json_hr_end]
    
    # Original trip count (before demand ratio) - sum all demand in the hour
    original_trip_count = sum(d['demand'] for d in demand_filtered)
    
    # Trip count after demand ratio application
    scaled_trip_count = original_trip_count * demand_ratio_value
    
    print(f"\nTrip Counts:")
    print(f"  Time range: minutes {json_hr_start}-{json_hr_end} (hour {json_hr_value})")
    print(f"  Original trips (entire hour): {original_trip_count:.1f}")
    print(f"  Scaled trips (after demand_ratio={demand_ratio_value}): {scaled_trip_count:.1f}")
    print(f"  Scaling factor: {demand_ratio_value}x")
    
    # Travel time statistics (from demand data)
    # Weight travel times by demand count (demand can be float)
    travel_times = [d['travel_time'] for d in demand_filtered for _ in range(int(d['demand']))]
    
    if travel_times:
        print(f"\nTravel Time Statistics (minutes):")
        print(f"  Mean: {np.mean(travel_times):.2f}")
        print(f"  Median: {np.median(travel_times):.2f}")
        print(f"  Std Dev: {np.std(travel_times):.2f}")
        print(f"  Min: {np.min(travel_times):.2f}")
        print(f"  Max: {np.max(travel_times):.2f}")
        print(f"  25th percentile: {np.percentile(travel_times, 25):.2f}")
        print(f"  75th percentile: {np.percentile(travel_times, 75):.2f}")
    
    # Price statistics (from demand data)
    prices = [d['price'] for d in demand_filtered for _ in range(int(d['demand']))]
    
    if prices:
        print(f"\nBase Price Statistics ($):")
        print(f"  Mean: ${np.mean(prices):.2f}")
        print(f"  Median: ${np.median(prices):.2f}")
        print(f"  Std Dev: ${np.std(prices):.2f}")
        print(f"  Min: ${np.min(prices):.2f}")
        print(f"  Max: ${np.max(prices):.2f}")
    
    # OD pair analysis
    od_pairs = defaultdict(int)
    for d in demand_filtered:
        od_pairs[(d['origin'], d['destination'])] += d['demand']
    
    print(f"\nOrigin-Destination Patterns:")
    print(f"  Unique OD pairs: {len(od_pairs)}")
    print(f"  Total possible OD pairs: {data['nlat'] * data['nlon'] * data['nlat'] * data['nlon']}")
    print(f"  Coverage: {len(od_pairs) / (data['nlat'] * data['nlon'] * data['nlat'] * data['nlon']) * 100:.1f}%")
    
    # Top 5 OD pairs by demand
    top_od = sorted(od_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 OD pairs by demand:")
    for (o, d), count in top_od:
        scaled_count = count * demand_ratio_value
        print(f"    {o} → {d}: {count} trips (scaled: {scaled_count:.0f})")
    
    # ============================================================================
    # REBALANCING TIME ANALYSIS
    # ============================================================================
    print(f"\n{'-'*80}")
    print("REBALANCING TIME ANALYSIS")
    print(f"{'-'*80}")
    
    reb_time_data = data['rebTime']
    
    # Convert json_hr to minutes for rebalancing time filter
    # But note: rebTime uses hour format (not minutes), so keep json_hr_value
    # Filter rebalancing times for the specific hour
    reb_filtered = [r for r in reb_time_data if r['time_stamp'] == json_hr_value]
    
    # Separate NaN and valid rebalancing times
    import math
    reb_times_valid = []
    nan_count = 0
    
    for r in reb_filtered:
        if isinstance(r['reb_time'], float) and math.isnan(r['reb_time']):
            nan_count += 1
        else:
            reb_times_valid.append(r['reb_time'])
    
    print(f"\nRebalancing Time Data:")
    print(f"  Total OD pairs with reb time: {len(reb_filtered)}")
    print(f"  Valid reb times: {len(reb_times_valid)}")
    print(f"  NaN reb times: {nan_count}")
    
    if reb_times_valid:
        print(f"\nRebalancing Time Statistics (minutes):")
        print(f"  Mean: {np.mean(reb_times_valid):.2f}")
        print(f"  Median: {np.median(reb_times_valid):.2f}")
        print(f"  Std Dev: {np.std(reb_times_valid):.2f}")
        print(f"  Min: {np.min(reb_times_valid):.2f}")
        print(f"  Max: {np.max(reb_times_valid):.2f}")
        print(f"  25th percentile: {np.percentile(reb_times_valid, 25):.2f}")
        print(f"  75th percentile: {np.percentile(reb_times_valid, 75):.2f}")
    
    if nan_count > 0:
        print(f"\n  ⚠️  WARNING: {nan_count} OD pairs have NaN rebalancing times!")
        print(f"     This will cause crashes during scenario initialization.")
        print(f"     These should be filtered out or replaced with estimates.")
    
    # Compare travel time vs rebalancing time
    if travel_times and reb_times_valid:
        print(f"\nTravel vs Rebalancing Time Comparison:")
        print(f"  Avg Travel Time: {np.mean(travel_times):.2f} min")
        print(f"  Avg Rebalancing Time: {np.mean(reb_times_valid):.2f} min")
        print(f"  Ratio (Reb/Travel): {np.mean(reb_times_valid) / np.mean(travel_times):.2f}x")
    
    # ============================================================================
    # TIME DISTRIBUTION ANALYSIS
    # ============================================================================
    print(f"\n{'-'*80}")
    print("TEMPORAL DISTRIBUTION")
    print(f"{'-'*80}")
    
    # Get all unique timestamps in demand data
    all_timestamps = set(d['time_stamp'] for d in demand_data)
    print(f"\nTime Coverage:")
    print(f"  Available timestamps in data: {sorted(all_timestamps)}")
    print(f"  Hour range: {min(all_timestamps)} - {max(all_timestamps)}")
    print(f"  Using hour: {json_hr_value}")
    
    # Demand by timestamp
    demand_by_hour = defaultdict(int)
    for d in demand_data:
        demand_by_hour[d['time_stamp']] += d['demand']
    
    print(f"\n  Demand by hour (original, not scaled):")
    for hour in sorted(demand_by_hour.keys()):
        print(f"    Hour {hour}: {demand_by_hour[hour]} trips")
    
    return {
        'city': city_name,
        'nregion': data['nlat'] * data['nlon'],
        'demand_ratio': demand_ratio_value,
        'original_trips': original_trip_count,
        'scaled_trips': scaled_trip_count,
        'avg_travel_time': np.mean(travel_times) if travel_times else None,
        'avg_reb_time': np.mean(reb_times_valid) if reb_times_valid else None,
        'reb_nan_count': nan_count,
        'unique_od_pairs': len(od_pairs),
        'avg_price': np.mean(prices) if prices else None,
    }


def main():
    """
    Main analysis function
    """
    print("\n" + "="*80)
    print("DATASET COMPARISON ANALYSIS")
    print("San Francisco vs NYC Brooklyn")
    print("="*80)
    
    # Analyze both datasets
    sf_stats = analyze_dataset(
        city_name='San Francisco',
        json_file='data/scenario_san_francisco.json',
        demand_ratio_value=demand_ratio['san_francisco'],
        json_hr_value=json_hr['san_francisco']
    )
    
    brooklyn_stats = analyze_dataset(
        city_name='NYC Brooklyn',
        json_file='data/scenario_nyc_brooklyn.json',
        demand_ratio_value=demand_ratio['nyc_brooklyn'],
        json_hr_value=json_hr['nyc_brooklyn']
    )
    
    # ============================================================================
    # COMPARATIVE SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'San Francisco':>20} {'NYC Brooklyn':>20}")
    print(f"{'-'*72}")
    print(f"{'Number of Regions':<30} {sf_stats['nregion']:>20} {brooklyn_stats['nregion']:>20}")
    print(f"{'Demand Ratio':<30} {sf_stats['demand_ratio']:>20.1f} {brooklyn_stats['demand_ratio']:>20.1f}")
    print(f"{'Original Trips (hour)':<30} {sf_stats['original_trips']:>20} {brooklyn_stats['original_trips']:>20}")
    print(f"{'Scaled Trips (hour)':<30} {sf_stats['scaled_trips']:>20.0f} {brooklyn_stats['scaled_trips']:>20.0f}")
    
    if sf_stats['avg_travel_time'] and brooklyn_stats['avg_travel_time']:
        print(f"{'Avg Travel Time (min)':<30} {sf_stats['avg_travel_time']:>20.2f} {brooklyn_stats['avg_travel_time']:>20.2f}")
    
    if sf_stats['avg_reb_time'] and brooklyn_stats['avg_reb_time']:
        print(f"{'Avg Rebalancing Time (min)':<30} {sf_stats['avg_reb_time']:>20.2f} {brooklyn_stats['avg_reb_time']:>20.2f}")
    
    print(f"{'Unique OD Pairs':<30} {sf_stats['unique_od_pairs']:>20} {brooklyn_stats['unique_od_pairs']:>20}")
    
    if sf_stats['avg_price'] and brooklyn_stats['avg_price']:
        print(f"{'Avg Base Price ($)':<30} {sf_stats['avg_price']:>20.2f} {brooklyn_stats['avg_price']:>20.2f}")
    
    print(f"{'NaN Rebalancing Times':<30} {sf_stats['reb_nan_count']:>20} {brooklyn_stats['reb_nan_count']:>20}")
    
    # Key differences
    print(f"\n{'='*80}")
    print("KEY DIFFERENCES")
    print(f"{'='*80}")
    
    print(f"\n1. Trip Volume:")
    if sf_stats['scaled_trips'] > 0 and brooklyn_stats['scaled_trips'] > 0:
        trip_ratio = brooklyn_stats['scaled_trips'] / sf_stats['scaled_trips']
        print(f"   Brooklyn has {trip_ratio:.1f}x more trips than San Francisco (after scaling)")
    else:
        print(f"   SF: {sf_stats['scaled_trips']:.0f} trips, Brooklyn: {brooklyn_stats['scaled_trips']:.0f} trips")
    
    if sf_stats['avg_travel_time'] and brooklyn_stats['avg_travel_time']:
        time_diff = brooklyn_stats['avg_travel_time'] - sf_stats['avg_travel_time']
        print(f"\n2. Travel Time:")
        if time_diff > 0:
            print(f"   Brooklyn trips are {time_diff:.1f} minutes longer on average")
        else:
            print(f"   San Francisco trips are {abs(time_diff):.1f} minutes longer on average")
    
    if sf_stats['avg_reb_time'] and brooklyn_stats['avg_reb_time']:
        reb_diff = brooklyn_stats['avg_reb_time'] - sf_stats['avg_reb_time']
        print(f"\n3. Rebalancing Time:")
        if reb_diff > 0:
            print(f"   Brooklyn rebalancing is {reb_diff:.1f} minutes longer on average")
        else:
            print(f"   San Francisco rebalancing is {abs(reb_diff):.1f} minutes longer on average")
    
    print(f"\n4. Data Quality:")
    if brooklyn_stats['reb_nan_count'] > 0:
        print(f"   ⚠️  Brooklyn has {brooklyn_stats['reb_nan_count']} NaN rebalancing times that need fixing!")
    if sf_stats['reb_nan_count'] > 0:
        print(f"   ⚠️  San Francisco has {sf_stats['reb_nan_count']} NaN rebalancing times that need fixing!")
    if sf_stats['reb_nan_count'] == 0 and brooklyn_stats['reb_nan_count'] == 0:
        print(f"   ✓ Both datasets have clean rebalancing time data")
    
    # ============================================================================
    # BETA PARAMETER EXPLANATION
    # ============================================================================
    print(f"\n{'='*80}")
    print("BETA PARAMETER EXPLANATION")
    print(f"{'='*80}")
    
    print(f"\nThe BETA parameter represents the **cost of rebalancing** per unit distance/time.")
    print(f"\nFrom your code (main_a2c_multi_agent.py):")
    print(f"  - San Francisco: beta = 0.2")
    print(f"  - NYC Brooklyn: beta = 0.5")
    print(f"\nWhat does BETA do?")
    print(f"  1. It penalizes the agent for moving empty vehicles (rebalancing)")
    print(f"  2. Higher beta = more expensive rebalancing = agent tries to minimize it")
    print(f"  3. Lower beta = cheaper rebalancing = agent may rebalance more freely")
    print(f"\nIn the reward function:")
    print(f"  reward = passenger_revenue - beta * rebalancing_cost")
    print(f"\nImplications:")
    print(f"  - Brooklyn (beta=0.5) penalizes rebalancing 2.5x more than SF (beta=0.2)")
    print(f"  - This means the Brooklyn agent must be more strategic about repositioning")
    print(f"  - San Francisco agent has more flexibility to move empty vehicles")
    print(f"\nWhy different values?")
    print(f"  - Calibrated based on real-world operating costs in each city")
    print(f"  - May reflect differences in:")
    print(f"    • Fuel/electricity costs")
    print(f"    • Driver wages")
    print(f"    • Traffic congestion")
    print(f"    • Distance between demand hotspots")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
