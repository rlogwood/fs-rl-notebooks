# from typing import List, Tuple
# import pandas as pd
#
# DEFAULT_MISSING_VALUES = ("", "NA", "N/A", "-", "null", "None", "?")
# DEFAULT_NEW_VALUE = "unknown"
#
# def replace_missing_values(df: pd.Series,


#
# # Create both datasets at once
# incomes_unknown = incomes.copy()
# incomes_imputed = incomes.copy()
#
# columns_to_replace = ['native.country', 'occupation', 'workclass']
#
# for col in columns_to_replace:
#     # Dataset 1: Replace with "Unknown"
#     incomes_unknown[col] = incomes_unknown[col].replace('?', 'Unknown')
#
#     # Dataset 2: Replace with most common value
#     mode_value = incomes_imputed[incomes_imputed[col] != '?'][col].mode()[0]
#     incomes_imputed[col] = incomes_imputed[col].replace('?', mode_value)
#
# # Verify the changes
# print("Checking '?' values in unknown dataset:")
# for col in columns_to_replace:
#     print(f"{col}: {(incomes_unknown[col] == '?').sum()}")
#
# print("\nChecking '?' values in imputed dataset:")
# for col in columns_to_replace:
#     print(f"{col}: {(incomes_imputed[col] == '?').sum()}")