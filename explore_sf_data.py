import json
import numpy as np

# Load the San Francisco scenario data
json_file = "data/scenario_san_francisco.json"

with open(json_file, "r") as file:
    data = json.load(file)

print("=" * 60)
print("SAN FRANCISCO DATASET ANALYSIS")
print("=" * 60)

# Display basic grid information
print("\n1. GRID STRUCTURE:")
print(f"   N1 (nlat): {data['nlat']}")
print(f"   N2 (nlon): {data['nlon']}")
print(f"   Grid size (N1 * N2): {data['nlat'] * data['nlon']}")

# Display actual number of regions
print("\n2. ACTUAL REGIONS:")
if 'region' in data:
    num_regions = data['region']
    print(f"   Number of regions: {num_regions}")
    print(f"   Region IDs: 0 to {num_regions - 1}")
else:
    print("   'region' field not found in JSON")
    print(f"   Defaulting to N1 * N2 = {data['nlat'] * data['nlon']}")

# Analyze demand data to see which regions actually have trips
print("\n3. REGIONS WITH DEMAND:")
origins = set()
destinations = set()
for item in data["demand"]:
    origins.add(item["origin"])
    destinations.add(item["destination"])

all_regions = origins.union(destinations)
print(f"   Unique origin regions: {len(origins)}")
print(f"   Unique destination regions: {len(destinations)}")
print(f"   Total unique regions: {len(all_regions)}")
print(f"   Region IDs in demand data: {sorted(all_regions)}")

# Analyze rebalancing time data
print("\n4. REGIONS IN REBALANCING DATA:")
reb_origins = set()
reb_destinations = set()
for item in data["rebTime"]:
    reb_origins.add(item["origin"])
    reb_destinations.add(item["destination"])

reb_regions = reb_origins.union(reb_destinations)
print(f"   Unique regions in rebTime: {len(reb_regions)}")
print(f"   Region IDs in rebTime data: {sorted(reb_regions)}")

# Check vehicle distribution
print("\n5. VEHICLE DISTRIBUTION:")
if "totalAcc" in data:
    for item in data["totalAcc"]:
        print(f"   Hour {item['hour']}: {item['acc']} vehicles")

# Sample some demand records
print("\n6. SAMPLE DEMAND RECORDS (first 5):")
for i, item in enumerate(data["demand"][:5]):
    print(f"   Record {i+1}: Time={item['time_stamp']}, O={item['origin']}, "
          f"D={item['destination']}, Demand={item['demand']}, "
          f"TT={item['travel_time']}, Price=${item['price']:.2f}")

# Check consistency
print("\n7. CONSISTENCY CHECK:")
if 'region' in data:
    expected_regions = set(range(data['region']))
    if all_regions == expected_regions:
        print("   ✓ Demand data covers all regions defined in 'region' field")
    else:
        missing = expected_regions - all_regions
        extra = all_regions - expected_regions
        if missing:
            print(f"   ⚠ Missing regions in demand: {sorted(missing)}")
        if extra:
            print(f"   ⚠ Extra regions in demand: {sorted(extra)}")
else:
    print("   Cannot check consistency - 'region' field not in JSON")

print("\n" + "=" * 60)