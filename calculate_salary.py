import pandas as pd

def calculate_regional_wages(s1901_file, s1902_file):
    # Helper to extract zip code from NAME column
    def extract_zip(name):
        return name.split('ZCTA5 ')[1] if (isinstance(name, str) and 'ZCTA5' in name) else None

    # Load S1901 for Household counts
    df1 = pd.read_csv(s1901_file, low_memory=False).iloc[1:]
    df1['zip_code'] = df1['NAME'].apply(extract_zip)
    # S1901_C01_001E: Total Households
    df_hh = df1[['zip_code', 'S1901_C01_001E']].rename(columns={'S1901_C01_001E': 'households'})

    # Load S1902 for Population and Per Capita Income
    df2 = pd.read_csv(s1902_file, low_memory=False).iloc[1:]
    df2['zip_code'] = df2['NAME'].apply(extract_zip)
    # S1902_C01_015E: Total Population
    # S1902_C02_015E: Per Capita Income (Mean income per person)
    df_inc = df2[['zip_code', 'S1902_C01_015E', 'S1902_C02_015E']].rename(columns={
        'S1902_C01_015E': 'population',
        'S1902_C02_015E': 'per_capita_income'
    })

    # Merge data and convert to numeric
    df = pd.merge(df_hh, df_inc, on='zip_code')
    for col in ['households', 'population', 'per_capita_income']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Define Regions
    regions = {
        'Southern Manhattan': ['10002', '10003', '10005', '10006', '10007', '10009', '10010', '10011', '10012', '10013', '10014', '10038'],
        'Brooklyn': [str(z) for z in range(11201, 11257)],
        'San Francisco': [str(z) for z in range(94102, 94189)]
    }

    results = []
    for region, zips in regions.items():
        rdf = df[df['zip_code'].isin(zips)]
        
        pop = rdf['population'].sum()
        hhs = rdf['households'].sum()
        # Aggregate total income for the region
        total_income = (rdf['per_capita_income'] * rdf['population']).sum()
        
        if pop > 0 and hhs > 0:
            avg_pc_income = total_income / pop
            avg_hh_size = pop / hhs
            # Hourly wage based on 2080 working hours/year
            hourly = avg_pc_income / 2080
            
            results.append({
                'Region': region,
                'HH Size': round(avg_hh_size, 2),
                'Hourly Wage': round(hourly, 2)
            })
            
    return pd.DataFrame(results)

# Execution
res = calculate_regional_wages("ACSST5Y2013.S1901-Data.csv", "ACSST5Y2013.S1902-Data.csv")
print(res)