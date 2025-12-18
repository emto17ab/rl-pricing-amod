import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('manhattan_income_discrete_with_hourly_wage.csv')

# Clean up the ZCTA column to get just the 5-digit zip code
df['zip_code'] = df['zcta']. str.replace('ZCTA5 ', '')

# List of zip codes you're interested in
target_zip_codes = ['10002', '10003', '10005', '10006', '10007', '10009', 
                    '10010', '10011', '10012', '10013', '10014', '10038']

# Filter data for only your target zip codes
df_filtered = df[df['zip_code'].isin(target_zip_codes)]

# Create the income distribution dictionary
income_distribution = {}

for zip_code in target_zip_codes:
    zip_data = df_filtered[df_filtered['zip_code'] == zip_code]
    
    # Extract hourly wages and probabilities
    bins = zip_data['assumed_hourly_wage_usd'].tolist()
    probabilities = zip_data['probability_mass'].tolist()
    
    income_distribution[zip_code] = {
        'bins': bins,
        'probabilities': probabilities
    }

# Print the distribution dictionary (formatted for easy copy-paste)
print("income_distribution = {")
for zip_code, data in income_distribution.items():
    print(f"    '{zip_code}': {{")
    print(f"        'bins': {data['bins']},")
    print(f"        'probabilities': {data['probabilities']}")
    print("    },")
print("}")

# Calculate the overall average hourly wage across all zip codes
# Weighted by the number of households in each bin
total_weighted_wage = 0
total_households = 0

for zip_code in target_zip_codes:
    zip_data = df_filtered[df_filtered['zip_code'] == zip_code]
    
    for _, row in zip_data.iterrows():
        weighted_wage = row['assumed_hourly_wage_usd'] * row['household_count']
        total_weighted_wage += weighted_wage
        total_households += row['household_count']

average_hourly_wage = total_weighted_wage / total_households
print(f"\nAverage hourly wage across all zip codes: ${average_hourly_wage:.2f}")

# Calculate aggregated distribution across all zip codes
aggregated_bins = df_filtered. groupby('assumed_hourly_wage_usd').agg({
    'household_count': 'sum'
}).reset_index()

# Calculate probability mass for aggregated data
aggregated_bins['probability_mass'] = aggregated_bins['household_count'] / aggregated_bins['household_count'].sum()

print("\nAggregated distribution across all zip codes:")
print("aggregated_distribution = {")
print(f"    'bins': {aggregated_bins['assumed_hourly_wage_usd']. tolist()},")
print(f"    'probabilities': {aggregated_bins['probability_mass'].tolist()}")
print("}")

# Verify probabilities sum to 1 for each zip code
print("\nVerification - Probabilities sum to ~1.0 for each zip code:")
for zip_code, data in income_distribution.items():
    prob_sum = sum(data['probabilities'])
    print(f"  {zip_code}: {prob_sum:.4f}")

# Show summary statistics
print("\n=== Summary Statistics ===")
print(f"Total households: {total_households}")
print(f"Weighted average hourly wage: ${average_hourly_wage:.2f}")
print(f"Weighted average annual income: ${average_hourly_wage * 40 * 52:.2f}")

# Distribution of households by zip code
print("\nHouseholds by zip code:")
zip_household_counts = df_filtered.groupby('zip_code')['household_count'].sum().sort_values(ascending=False)
for zip_code, count in zip_household_counts.items():
    pct = (count / total_households) * 100
    print(f"  {zip_code}: {count: 4d} households ({pct: 5.2f}%)")