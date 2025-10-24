"""
Calculate total demand for San Francisco and Manhattan datasets
at json_hr 19 with demand_ratio applied
"""

import json
import numpy as np

# Configuration from main_a2c_multi_agent.py
demand_ratio = {
    'san_francisco': 2,
    'nyc_manhattan': 0.10
}

json_hr = {
    'san_francisco': 19,
    'nyc_manhattan': 19
}

def calculate_demand(city_name, json_file, demand_ratio_value, json_hr_value):
    """
    Calculate total demand for a city at specific hour with demand ratio applied
    """
    print(f"\n{'='*80}")
    print(f"Calculating demand for: {city_name.upper()}")
    print(f"{'='*80}")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f, parse_constant=lambda x: float('nan') if x == 'NaN' else None)
    
    print(f"\nConfiguration:")
    print(f"  Demand Ratio: {demand_ratio_value}")
    print(f"  JSON Hour: {json_hr_value}")
    print(f"  Grid: {data['nlat']} × {data['nlon']} = {data['nlat'] * data['nlon']} regions")
    
    demand_data = data['demand']
    
    # Convert json_hr (hour) to minute range: hour 19 = minutes 1140-1199 (19:00-19:59)
    json_hr_start = json_hr_value * 60
    json_hr_end = json_hr_start + 59  # Full hour range
    
    print(f"  Time range: minutes {json_hr_start}-{json_hr_end}")
    
    # Filter demand for the ENTIRE hour (all minutes in hour 19)
    demand_filtered = [d for d in demand_data if json_hr_start <= d['time_stamp'] <= json_hr_end]
    
    # Original trip count (before demand ratio) - sum all demand in the hour
    original_trip_count = sum(d['demand'] for d in demand_filtered)
    
    # Trip count after demand ratio application
    scaled_trip_count = original_trip_count * demand_ratio_value
    
    print(f"\nDemand Calculation:")
    print(f"  Original trips (hour {json_hr_value}): {original_trip_count:.1f}")
    print(f"  Demand ratio: {demand_ratio_value}x")
    print(f"  TOTAL SCALED DEMAND: {scaled_trip_count:.1f} trips")
    
    return scaled_trip_count

def main():
    print("\n" + "="*80)
    print("TOTAL DEMAND CALCULATION")
    print("="*80)
    
    # Calculate for San Francisco
    sf_demand = calculate_demand(
        city_name='San Francisco',
        json_file='data/scenario_san_francisco.json',
        demand_ratio_value=demand_ratio['san_francisco'],
        json_hr_value=json_hr['san_francisco']
    )
    
    # Calculate for Manhattan
    manhattan_demand = calculate_demand(
        city_name='NYC Manhattan',
        json_file='data/scenario_nyc_manhattan.json',
        demand_ratio_value=demand_ratio['nyc_manhattan'],
        json_hr_value=json_hr['nyc_manhattan']
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal Scaled Demand at Hour 19:")
    print(f"  San Francisco: {sf_demand:.1f} trips")
    print(f"  NYC Manhattan: {manhattan_demand:.1f} trips")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
