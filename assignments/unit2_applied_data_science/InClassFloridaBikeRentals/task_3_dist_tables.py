
# Task 3: Data Analysis with Pandas - Distribution Tables

import pandas as pd
import numpy as np

# 1. Temperature and Rented Bike Count distribution by Hour
print("=== TEMPERATURE AND RENTED BIKE COUNT DISTRIBUTION BY HOUR ===")

# Group by Hour and calculate statistics
hourly_distribution = bikes.groupby('Hour').agg({
    'Temperature(°C)': ['mean', 'min', 'max', 'std'],
    'Rented Bike Count': ['mean', 'min', 'max', 'std', 'sum']
}).round(2)

# Flatten column names for better readability
hourly_distribution.columns = [
    'Temp_Mean', 'Temp_Min', 'Temp_Max', 'Temp_Std',
    'Rentals_Mean', 'Rentals_Min', 'Rentals_Max', 'Rentals_Std', 'Rentals_Total'
]

print("Hourly Distribution Table:")
print(hourly_distribution)

# Create a more detailed hourly summary
hourly_summary = bikes.groupby('Hour').agg({
    'Temperature(°C)': 'mean',
    'Rented Bike Count': ['mean', 'count', 'sum']
}).round(1)

hourly_summary.columns = ['Avg_Temperature', 'Avg_Rentals', 'Data_Points', 'Total_Rentals']
print("\nSimplified Hourly Summary:")
print(hourly_summary)

print("\n" + "="*70 + "\n")

# 2. Seasons and Rented Bike Count Distribution
print("=== SEASONS AND RENTED BIKE COUNT DISTRIBUTION ===")

# Group by Seasons and calculate comprehensive statistics
seasonal_distribution = bikes.groupby('Seasons').agg({
    'Rented Bike Count': ['count', 'mean', 'median', 'min', 'max', 'std', 'sum'],
    'Temperature(°C)': ['mean', 'min', 'max'],
    'Humidity(%)': 'mean',
    'Wind speed (m/s)': 'mean'
}).round(2)

# Flatten column names
seasonal_distribution.columns = [
    'Data_Points', 'Rentals_Mean', 'Rentals_Median', 'Rentals_Min', 'Rentals_Max', 
    'Rentals_Std', 'Rentals_Total', 'Temp_Mean', 'Temp_Min', 'Temp_Max',
    'Humidity_Mean', 'WindSpeed_Mean'
]

print("Seasonal Distribution Table:")
print(seasonal_distribution)

# Create percentage distribution of total rentals by season
total_rentals = bikes['Rented Bike Count'].sum()
seasonal_percentage = bikes.groupby('Seasons')['Rented Bike Count'].sum() / total_rentals * 100

print(f"\nSeasonal Rental Distribution (% of total {total_rentals:,} rentals):")
for season, percentage in seasonal_percentage.items():
    print(f"{season}: {percentage:.1f}%")

print("\n" + "="*70 + "\n")

# 3. Additional Distribution Tables

# Temperature ranges distribution
print("=== TEMPERATURE RANGE DISTRIBUTION ===")
# Create temperature bins
bikes['Temp_Range'] = pd.cut(bikes['Temperature(°C)'], 
                            bins=[-20, -10, 0, 10, 20, 30], 
                            labels=['Very Cold (<-10°C)', 'Cold (-10-0°C)', 
                                   'Cool (0-10°C)', 'Mild (10-20°C)', 'Warm (20-30°C)'])

temp_range_dist = bikes.groupby('Temp_Range', observed=True).agg({
    'Rented Bike Count': ['count', 'mean', 'sum']
}).round(1)

temp_range_dist.columns = ['Data_Points', 'Avg_Rentals', 'Total_Rentals']
print("Temperature Range Distribution:")
print(temp_range_dist)

print("\n" + "="*70 + "\n")

# 4. Cross-tabulation: Seasons vs Holiday
print("=== CROSS-TABULATION: SEASONS VS HOLIDAY ===")
cross_tab = pd.crosstab(bikes['Seasons'], bikes['Holiday'], 
                        values=bikes['Rented Bike Count'], 
                        aggfunc='mean', margins=True).round(1)

print("Average Rentals by Season and Holiday Status:")
print(cross_tab)

print("\n" + "="*70 + "\n")

# 5. Hour vs Season Distribution (Heatmap data)
print("=== HOURLY RENTALS BY SEASON (PIVOT TABLE) ===")
hour_season_pivot = bikes.pivot_table(
    values='Rented Bike Count',
    index='Hour',
    columns='Seasons',
    aggfunc='mean'
).round(1)

print("Average Rentals by Hour and Season:")
print(hour_season_pivot)

# Export the distribution tables to CSV files
print("\n" + "="*50)
print("EXPORTING DISTRIBUTION TABLES...")

hourly_distribution.to_csv('hourly_distribution.csv')
seasonal_distribution.to_csv('seasonal_distribution.csv')
temp_range_dist.to_csv('temperature_range_distribution.csv')
hour_season_pivot.to_csv('hour_season_pivot.csv')

print("✓ hourly_distribution.csv")
print("✓ seasonal_distribution.csv") 
print("✓ temperature_range_distribution.csv")
print("✓ hour_season_pivot.csv")
print("Distribution tables exported successfully!")
