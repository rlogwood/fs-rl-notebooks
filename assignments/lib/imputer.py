from typing import List, Tuple
import pandas as pd

DEFAULT_MISSING_VALUES = ("", "NA", "N/A", "-", "null", "None", "?")
DEFAULT_NEW_VALUE = "unknown"

def replace_missing_values(df: pd.Series,
    missing_values: List[str] = DEFAULT_MISSING_VALUES,
    new_value: str = DEFAULT_NEW_VALUE) -> pd.Series:
    # missing str value lambda
    missing_str_val = lambda s: s.str.strip().isin(missing_values)

    # missing numeric value lambda
    missing_num_val = lambda v: v.isin(missing_values)
    df_unknown = df.copy()

    # Get only object columns
    object_columns = df.select_dtypes(include=['object']).columns

    # Apply the lambda function only to object columns
    if len(object_columns) > 0:
        mask = df[object_columns].apply(missing_str_val)
        df_unknown[mask] = new_value

    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        mask = df[numeric_columns].apply(missing_num_val)
        df_unknown[mask] = pd.NA

    return df_unknown

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