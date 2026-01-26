"""
Polynomial Features Example

This script demonstrates how polynomial features work using actual data
from the FloridaBikeRentals dataset. It shows how 3 features expand into
9 polynomial features and explains why each feature type matters.

Run from the ML_Incremental_Capstone directory:
    python examples/polynomial_example.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def load_data():
    """Load and clean the bike rentals dataset."""
    df = pd.read_csv('../data/FloridaBikeRentals.csv', encoding='latin-1')
    df.columns = (df.columns
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('°C', 'C', regex=False)
                  .str.replace('%', 'Pct', regex=False)
                  .str.replace(' ', '_', regex=False)
                  .str.replace('/', '_', regex=False))
    return df


def example_basic_transformation():
    """Show basic polynomial transformation on first 2 rows."""
    df = load_data()

    # Take just 2 rows and 3 features to keep it simple
    simple_data = df[['TemperatureC', 'HumidityPct', 'Hour']].head(2)

    print('=' * 60)
    print('ORIGINAL DATA (2 rows, 3 features)')
    print('=' * 60)
    print(simple_data.to_string())
    print()

    row0 = simple_data.iloc[0]
    print(f'Row 0: Temp={row0["TemperatureC"]}, Humidity={row0["HumidityPct"]}, Hour={row0["Hour"]}')

    # Apply polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_data = poly.fit_transform(simple_data)

    print()
    print('=' * 60)
    print('AFTER POLYNOMIAL TRANSFORMATION')
    print('=' * 60)
    print(f'Original shape: {simple_data.shape} (2 rows, 3 features)')
    print(f'Polynomial shape: {poly_data.shape} (2 rows, 9 features)')
    print()

    feature_names = ['Temp', 'Humid', 'Hour', 'Temp^2', 'Temp*Humid',
                     'Temp*Hour', 'Humid^2', 'Humid*Hour', 'Hour^2']
    print('The 9 polynomial features are:')
    for i, name in enumerate(feature_names):
        print(f'  {i + 1}. {name}')

    print()
    print('=' * 60)
    print('MANUAL CALCULATION FOR ROW 0')
    print('=' * 60)
    t, h, hr = row0['TemperatureC'], row0['HumidityPct'], row0['Hour']
    print(f'Original values: Temp={t}, Humidity={h}, Hour={hr}')
    print()
    print('Step-by-step polynomial calculation:')
    print()
    print('--- Original features (kept as-is) ---')
    print(f'  1. Temp        = {t}')
    print(f'  2. Humid       = {h}')
    print(f'  3. Hour        = {hr}')
    print()
    print('--- Squared terms (feature * itself) ---')
    print(f'  4. Temp^2      = {t} * {t} = {t ** 2}')
    print(f'  7. Humid^2     = {h} * {h} = {h ** 2}')
    print(f'  9. Hour^2      = {hr} * {hr} = {hr ** 2}')
    print()
    print('--- Interaction terms (feature * different feature) ---')
    print(f'  5. Temp*Humid  = {t} * {h} = {t * h}')
    print(f'  6. Temp*Hour   = {t} * {hr} = {t * hr}')
    print(f'  8. Humid*Hour  = {h} * {hr} = {h * hr}')

    print()
    print('=' * 60)
    print('FINAL RESULT: Row 0 transformed')
    print('=' * 60)
    print()
    print('Before (3 values):')
    print(f'  [{t}, {h}, {hr}]')
    print()
    print('After (9 values):')
    print(f'  [{t}, {h}, {hr}, {t ** 2}, {t * h}, {t * hr}, {h ** 2}, {h * hr}, {hr ** 2}]')
    print()
    print('Verification - sklearn output:')
    print(f'  {list(poly_data[0])}')


def example_noon_hour():
    """Show polynomial transformation for a noon hour example."""
    df = load_data()

    # Find a row with Hour=12 (noon) for a more interesting example
    noon_row = df[df['Hour'] == 12].iloc[50]

    print()
    print('=' * 60)
    print('EXAMPLE WITH NON-ZERO HOUR (Noon)')
    print('=' * 60)
    t = noon_row['TemperatureC']
    h = noon_row['HumidityPct']
    hr = noon_row['Hour']
    rentals = noon_row['Rented_Bike_Count']

    print(f'Temperature: {t}C')
    print(f'Humidity: {h}%')
    print(f'Hour: {hr} (noon)')
    print(f'Actual bike rentals: {rentals}')
    print()

    print('=' * 60)
    print('POLYNOMIAL FEATURES CALCULATION')
    print('=' * 60)
    print()
    print('Original 3 features:')
    print(f'  [Temp={t}, Humid={h}, Hour={hr}]')
    print()
    print('Expanded to 9 features:')
    print()
    print(f'  1. Temp         = {t}')
    print(f'  2. Humid        = {h}')
    print(f'  3. Hour         = {hr}')
    print(f'  4. Temp^2       = {t}^2 = {t ** 2}')
    print(f'  5. Temp*Humid   = {t} * {h} = {t * h}')
    print(f'  6. Temp*Hour    = {t} * {hr} = {t * hr}')
    print(f'  7. Humid^2      = {h}^2 = {h ** 2}')
    print(f'  8. Humid*Hour   = {h} * {hr} = {h * hr}')
    print(f'  9. Hour^2       = {hr}^2 = {hr ** 2}')


def explain_why_features_matter():
    """Explain the purpose of each polynomial feature type."""
    print()
    print('=' * 60)
    print('WHY THESE FEATURES MATTER FOR PREDICTION')
    print('=' * 60)
    print('''
The model can now learn patterns like:

1. Temp^2 (squared term):
   "Rentals increase with temperature, but decrease
   at extreme temperatures (too hot or too cold)"
   This creates a curve, not a straight line.

2. Temp*Hour (interaction term):
   "The effect of temperature depends on the hour.
   Hot weather at noon = lots of rentals.
   Hot weather at midnight = fewer rentals."
   Temperature matters MORE during daytime hours.

3. Humid*Hour (interaction term):
   "High humidity at rush hour has a bigger negative
   impact than high humidity at 3am."

Without polynomial features, the model treats each
feature independently. With them, it can capture
these combined effects.
''')


def show_feature_count_formula():
    """Show how to calculate the number of polynomial features."""
    print()
    print('=' * 60)
    print('POLYNOMIAL FEATURE COUNT FORMULA')
    print('=' * 60)
    print('''
For degree=2 polynomial features with n original features:

  Total features = n + n + C(n,2)
                 = n (original) + n (squared) + n*(n-1)/2 (interactions)

Example with 3 features:
  = 3 + 3 + 3*2/2
  = 3 + 3 + 3
  = 9 features

Example with 5 features (as in the notebook):
  = 5 + 5 + 5*4/2
  = 5 + 5 + 10
  = 20 features
''')

    # Verify with sklearn
    print('Verification with sklearn:')
    for n in [3, 5, 10]:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        dummy_data = np.zeros((1, n))
        transformed = poly.fit_transform(dummy_data)
        print(f'  {n} features -> {transformed.shape[1]} polynomial features')


if __name__ == '__main__':
    example_basic_transformation()
    example_noon_hour()
    explain_why_features_matter()
    show_feature_count_formula()
