import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
# Assuming the file is named: API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv
df = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv', skiprows=4)

# Clean the column names (remove spaces and special characters)
df.columns = df.columns.str.strip()

# Display basic info about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:", df.columns.tolist()[:10], "...")  # Show first 10 columns

# Find the column for 2024 (or the latest available year)
year_columns = [col for col in df.columns if col.isdigit()]
latest_year = max(year_columns) if year_columns else None

if latest_year:
    print(f"\nLatest year in dataset: {latest_year}")
    
    # Filter out regional/group data and keep only individual countries
    # We'll exclude rows with certain patterns that indicate regions/groups
    exclude_keywords = ['World', 'Region', 'Income', 'aggregate', 'IBRD', 'IDA', 
                       'Latin America', 'Sub-Saharan', 'Middle East', 'Euro area',
                       'European Union', 'East Asia', 'South Asia', 'North America',
                       'Small states', 'Fragile', 'Heavily indebted', 'Least developed']
    
    # Create a filter for individual countries
    country_mask = ~df['Country Name'].str.contains('|'.join(exclude_keywords), case=False, na=False)
    
    # Get data for individual countries in the latest year
    country_data = df[country_mask].copy()
    country_data['Population'] = pd.to_numeric(country_data[latest_year], errors='coerce')
    
    # Remove rows with missing population data
    country_data = country_data.dropna(subset=['Population'])
    
    # Sort by population and get top 15
    top_15 = country_data.sort_values('Population', ascending=False).head(15)
    
    # Create the bar chart
    plt.figure(figsize=(16, 9))
    bars = plt.bar(top_15['Country Name'], top_15['Population'] / 1e6, 
                   color=plt.cm.tab20c(np.arange(len(top_15))))
    
    plt.title(f'Top 15 Most Populous Countries in {latest_year}', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Population (Millions)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:,.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add a horizontal line for visual reference
    plt.axhline(y=0, color='black', linewidth=0.8)
    
    # Add footer with data source
    plt.figtext(0.5, 0.01, 'Data Source: World Bank World Development Indicators', 
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    # Display the top 15 countries in a table
    print(f"\nTop 15 Most Populous Countries in {latest_year}:")
    print("=" * 70)
    print(f"{'Rank':<5} {'Country':<30} {'Population':>20}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top_15.iterrows(), 1):
        pop_millions = row['Population'] / 1e6
        print(f"{i:<5} {row['Country Name'][:28]:<30} {pop_millions:>19.1f}M")
    
    # Optional: Create a histogram of population distribution
    plt.figure(figsize=(12, 7))
    
    # Filter for countries with population > 1 million for better visualization
    pop_values = country_data[country_data['Population'] > 1e6]['Population']
    
    # Create histogram with log scale
    plt.hist(np.log10(pop_values), bins=30, edgecolor='black', 
             alpha=0.7, color='steelblue')
    
    plt.title(f'Distribution of Country Populations in {latest_year} (Countries > 1M)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Log10(Population)', fontsize=12)
    plt.ylabel('Number of Countries', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Custom x-axis labels
    x_ticks = [6, 7, 8, 9]  # 1M, 10M, 100M, 1B
    x_labels = ['1M', '10M', '100M', '1B']
    plt.xticks(x_ticks, x_labels)
    
    plt.tight_layout()
    plt.show()
    
else:
    print("No year columns found in the dataset!")
    
# Show some statistics about the dataset
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"Total number of countries/regions: {len(df)}")
print(f"Time range: {min(year_columns)} - {max(year_columns)}")
print(f"Number of years covered: {len(year_columns)}")

# Count individual countries vs regions
individual_countries = sum(country_mask)
regions = len(df) - individual_countries
print(f"Individual countries: {individual_countries}")
print(f"Regions/groups: {regions}")