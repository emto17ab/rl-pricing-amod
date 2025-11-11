"""
Calculate average revenue per minute PER TRIP for San Francisco and Manhattan datasets
at json_hr 19 with demand_ratio applied
Revenue per minute per trip = Price / Travel Time
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

def calculate_revenue_per_minute(city_name, json_file, demand_ratio_value, json_hr_value):
    """
    Calculate average revenue per minute PER TRIP for a city at specific hour
    This calculates how much revenue is earned per minute of travel time per trip
    """
    print(f"\n{'='*80}")
    print(f"Analyzing revenue for: {city_name.upper()}")
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
    json_hr_end = json_hr_start + 59  # Full hour range (60 minutes total)
    
    print(f"  Time range: minutes {json_hr_start}-{json_hr_end} (60 minutes)")
    
    # Filter demand for the ENTIRE hour (all minutes in hour 19)
    demand_filtered = [d for d in demand_data if json_hr_start <= d['time_stamp'] <= json_hr_end]
    
    # Calculate revenue per minute PER TRIP for each OD pair
    revenue_per_minute_per_trip = []
    for d in demand_filtered:
        if d['travel_time'] > 0:  # Avoid division by zero
            rpm = d['price'] / d['travel_time']
            revenue_per_minute_per_trip.append(rpm)
    
    # Calculate statistics
    total_trips = sum(d['demand'] for d in demand_filtered)
    total_revenue = sum(d['demand'] * d['price'] for d in demand_filtered)
    
    # Apply demand ratio to totals
    scaled_trips = total_trips * demand_ratio_value
    scaled_revenue = total_revenue * demand_ratio_value
    
    # Average revenue per minute per trip (weighted by demand)
    weighted_rpm = sum(d['demand'] * (d['price'] / d['travel_time']) for d in demand_filtered if d['travel_time'] > 0)
    total_demand_nonzero = sum(d['demand'] for d in demand_filtered if d['travel_time'] > 0)
    avg_revenue_per_minute_per_trip = weighted_rpm / total_demand_nonzero if total_demand_nonzero > 0 else 0
    
    # Calculate average price per trip
    avg_price_per_trip = total_revenue / total_trips if total_trips > 0 else 0
    
    # Additional statistics
    prices = [d['price'] for d in demand_filtered]
    travel_times = [d['travel_time'] for d in demand_filtered]
    
    print(f"\nTrip Statistics:")
    print(f"  Original trips (hour {json_hr_value}): {total_trips:.1f}")
    print(f"  Scaled trips (with ratio): {scaled_trips:.1f}")
    
    print(f"\nRevenue Statistics:")
    print(f"  Original revenue (hour {json_hr_value}): ${total_revenue:.2f}")
    print(f"  Scaled revenue (with ratio): ${scaled_revenue:.2f}")
    print(f"  Average price per trip: ${avg_price_per_trip:.2f}")
    
    print(f"\nPrice Distribution:")
    print(f"  Min price: ${min(prices):.2f}")
    print(f"  Max price: ${max(prices):.2f}")
    print(f"  Mean price: ${np.mean(prices):.2f}")
    print(f"  Median price: ${np.median(prices):.2f}")
    print(f"  Std dev: ${np.std(prices):.2f}")
    
    print(f"\nTravel Time Distribution:")
    print(f"  Min: {min(travel_times):.1f} min")
    print(f"  Max: {max(travel_times):.1f} min")
    print(f"  Mean: {np.mean(travel_times):.1f} min")
    print(f"  Median: {np.median(travel_times):.1f} min")
    
    print(f"\nRevenue per Minute per Trip Distribution:")
    print(f"  Min: ${min(revenue_per_minute_per_trip):.2f}/min")
    print(f"  Max: ${max(revenue_per_minute_per_trip):.2f}/min")
    print(f"  Mean (unweighted): ${np.mean(revenue_per_minute_per_trip):.2f}/min")
    print(f"  Median: ${np.median(revenue_per_minute_per_trip):.2f}/min")
    
    print(f"\n{'*'*80}")
    print(f"AVERAGE REVENUE PER MINUTE PER TRIP (weighted): ${avg_revenue_per_minute_per_trip:.2f}/min")
    print(f"{'*'*80}")
    
    return {
        'city': city_name,
        'scaled_trips': scaled_trips,
        'scaled_revenue': scaled_revenue,
        'revenue_per_minute_per_trip': avg_revenue_per_minute_per_trip,
        'avg_price_per_trip': avg_price_per_trip,
        'avg_travel_time': np.mean(travel_times),
        'median_price': np.median(prices),
        'median_travel_time': np.median(travel_times),
        'median_rpm_per_trip': np.median(revenue_per_minute_per_trip)
    }

def main():
    print("\n" + "="*80)
    print("REVENUE PER MINUTE ANALYSIS")
    print("Comparing San Francisco and Manhattan at Hour 19")
    print("="*80)
    
    # Calculate for San Francisco
    sf_stats = calculate_revenue_per_minute(
        city_name='San Francisco',
        json_file='data/scenario_san_francisco.json',
        demand_ratio_value=demand_ratio['san_francisco'],
        json_hr_value=json_hr['san_francisco']
    )
    
    # Calculate for Manhattan
    manhattan_stats = calculate_revenue_per_minute(
        city_name='NYC Manhattan',
        json_file='data/scenario_nyc_manhattan.json',
        demand_ratio_value=demand_ratio['nyc_manhattan'],
        json_hr_value=json_hr['nyc_manhattan']
    )
    
    # Comparative Summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<40} {'San Francisco':>20} {'NYC Manhattan':>20}")
    print(f"{'-'*82}")
    print(f"{'Scaled Demand (trips/hour)':<40} {sf_stats['scaled_trips']:>20.1f} {manhattan_stats['scaled_trips']:>20.1f}")
    print(f"{'Total Revenue ($/hour)':<40} ${sf_stats['scaled_revenue']:>19.2f} ${manhattan_stats['scaled_revenue']:>19.2f}")
    print(f"{'Avg Price per Trip ($)':<40} ${sf_stats['avg_price_per_trip']:>19.2f} ${manhattan_stats['avg_price_per_trip']:>19.2f}")
    print(f"{'Avg Travel Time (min)':<40} {sf_stats['avg_travel_time']:>20.1f} {manhattan_stats['avg_travel_time']:>20.1f}")
    print(f"{'-'*82}")
    print(f"{'REVENUE PER MINUTE PER TRIP ($/min)':<40} ${sf_stats['revenue_per_minute_per_trip']:>19.2f} ${manhattan_stats['revenue_per_minute_per_trip']:>19.2f}")
    print(f"{'Median Revenue/min/trip ($/min)':<40} ${sf_stats['median_rpm_per_trip']:>19.2f} ${manhattan_stats['median_rpm_per_trip']:>19.2f}")
    print(f"{'-'*82}")
    print(f"{'Median Price ($)':<40} ${sf_stats['median_price']:>19.2f} ${manhattan_stats['median_price']:>19.2f}")
    print(f"{'Median Travel Time (min)':<40} {sf_stats['median_travel_time']:>20.1f} {manhattan_stats['median_travel_time']:>20.1f}")
    
    # Calculate ratio
    ratio = manhattan_stats['revenue_per_minute_per_trip'] / sf_stats['revenue_per_minute_per_trip']
    print(f"\n{'='*82}")
    print(f"Manhattan earns {ratio:.2f}× more revenue per minute per trip than San Francisco")
    print(f"{'='*82}\n")

if __name__ == "__main__":
    main()
