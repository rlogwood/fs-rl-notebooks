from pandas import describe_option

try:
    # When imported as part of a package
    from . import text_util as tu, imputer as im
except ImportError:
    # When run as a standalone script
    import text_util as tu
    import imputer as im

#
# Short answer: it’s necessary but not always sufficient.
#
# - What it guarantees: no pandas-native missing markers (NaN/NaT/None) are present in any column.
# - What it doesn’t catch:
#   - “Missing” encoded as strings like "", " ", "NA", "N/A", "-", "null", "None", "?".
#   - Columns read as object because of mixed types where missing are hidden as text.
#   - Out-of-range placeholders (e.g., -1, 9999) that semantically mean missing.
#   - Datetime parsing failures if the column is still string.
#
# Recommendations:
# - Inspect object columns for suspicious values:
#   - sales.select_dtypes(include="object").agg(lambda s: s.str.strip().isin(["", "NA","N/A","-","null","None","?"]).sum())
# - Normalize such tokens to NaN, then re-run isna().sum().
# - For numerics that came in as strings, coerce:
#   - sales[col] = pd.to_numeric(sales[col], errors="coerce")
# - For dates:
#   - sales[col] = pd.to_datetime(sales[col], errors="coerce")
# - Check whitespace-only cells:
#   - sales.select_dtypes("object").apply(lambda s: s.str.strip().eq("").sum())
#
# If all of the above show zero after cleaning, you can confidently state there’s no missing data.


# Check column names and sample values first
def print_col_info(df):
    tu.styled_display("Column info:")
    print(f"{tu.bold_text("Columns:")} {df.columns.tolist()}")
    print(f"{tu.bold_text("Shape:")} {df.shape}")
    print(f"\n{tu.bold_text("Dtypes:")}\n{"="*7}\n{df.dtypes}")

    print(f"\n{tu.bold_text("Sample values for object columns:")}")
    print("="*70)
    for col in df.select_dtypes("object").columns:
        print(f"{col}: {df[col].head(3).tolist()}")


def print_missing_values_counts(df):
    # Standard NA Check
    tu.styled_display("Standard NA Check:")
    print(df.isna().sum())

    # Check object columns for bad values
    #obj_cols = df.select_dtypes(include="object").agg(
    #    lambda s: s.str.strip().isin(["", "NA", "N/A", "-", "null", "None", "?"]).sum())

    obj_cols = df.select_dtypes(include="object").agg(
        lambda s: s.str.strip().isin(im.DEFAULT_MISSING_VALUES).sum())
    tu.styled_display("Bad object Values Check:")
    print(obj_cols)


def show_group_by_analysis(df, group_by_col, agg_col):
    tu.styled_display(f"Group by analysis for {group_by_col}:")
    analysis = df.groupby(group_by_col)[agg_col].agg(['mean', 'count', 'sum'])
    analysis.loc['Total'] = analysis.sum()
    print("")
    description = f"{agg_col} Analysis"
    print(tu.print_sub_heading(description))
    print(analysis)


def show_group_by_analysis_multi(df, group_by_col, agg_cols, description, sort_by = None):
    analysis = df.groupby(group_by_col).agg(agg_cols).round(2)

    # Debugging Stuff
    # try:
    #     print(f"sort_by: {sort_by}")
    #     print(f"sort_by[0]: {sort_by[0]}")
    #     print(f"sort_by[1]: {sort_by[1]}")
    #
    #     print(f"agg_cols: {agg_cols}")
    #     print(f"agg_cols[0]: {agg_cols[0]}, agg_cols[0][0]: {agg_cols[0][0]}")
    # except Exception as e:
    #     print(f"Exception: {e}")

    if sort_by:
        try:
            analysis = analysis.sort_values((sort_by[0], sort_by[1]), ascending=False)
        except Exception as e:
            print(f"Sorting Exception: {e}")

    analysis.loc['Total'] = analysis.sum()
    analysis = analysis.map(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
    print("")
    print(tu.print_sub_heading(description))
    print(analysis)


def add_commas(data):
    return data.map(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)


def show_unique_values_for_object_columns(df, max_unique=50):
    """
    Display unique values for all object (string) columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze
    max_unique (int): Maximum number of unique values to display per column

    Returns:
    dict: Dictionary with column names as keys and unique values as values
    """
    object_cols = df.select_dtypes(include=['object']).columns

    if len(object_cols) == 0:
        print("No object columns found in the DataFrame.")
        return {}

    unique_values_dict = {}

    tu.print_heading(f"Unique values for object columns (up to {max_unique}):")

    for col in object_cols:
        unique_vals = df[col].unique()
        unique_count = len(unique_vals)

        print(f"\n📊 Column: {col}")
        print(f"   Total unique values: {unique_count}")

        if unique_count <= max_unique:
            print(f"   Values: {list(unique_vals)}")
        else:
            print(f"   First {max_unique} values: {list(unique_vals[:max_unique])}")
            print(f"   ... and {unique_count - max_unique} more")

        # Also show value counts
        value_counts = df[col].value_counts()

        print(f"   Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} times)")

        unique_values_dict[col] = unique_vals

    return unique_values_dict


# Usage example:
# unique_dict = show_unique_values_for_object_columns(sales)


def clean_object_columns(df, columns=None, inplace=False):
    """
    clean string values in object columns by:
    - Trimming whitespace
    - Converting to title case (or specify other cases)
    - Replacing common missing value representations with NaN

    Parameters:
    df (pd.DataFrame): The DataFrame to clean
    columns (list): Specific columns to clean. If None, cleans all object columns
    inplace (bool): Whether to modify the original DataFrame or return a copy

    Returns:
    pd.DataFrame: cleaned DataFrame
    """
    import pandas as pd
    import numpy as np

    # Common representations of missing values
    missing_values = ['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NULL', 'null',
                      'None', 'none', 'NaN', 'nan', '-', '?', 'unknown', 'UNKNOWN']

    # Work on copy if not inplace
    result_df = df if inplace else df.copy()

    # Get object columns to process
    if columns is None:
        columns = result_df.select_dtypes(include=['object']).columns
    else:
        # Filter to only object columns
        columns = [col for col in columns if result_df[col].dtype == 'object']

    print(f"Cleaning columns: {list(columns)}")

    for col in columns:
        print(f"\n🔧 Processing column: {col}")

        # Store original unique values for comparison
        original_unique = len(result_df[col].unique())

        # Step 1: Convert to string and strip whitespace
        result_df[col] = result_df[col].astype(str).str.strip()

        # Step 2: Replace missing value representations with NaN
        result_df[col] = result_df[col].replace(missing_values, np.nan)

        # Step 3: standardize capitalization (Title Case)
        # Only apply to non-null values
        mask = result_df[col].notna()
        result_df.loc[mask, col] = result_df.loc[mask, col].str.title()

        # Optional: Additional normalizations
        # Remove extra spaces between words
        result_df.loc[mask, col] = result_df.loc[mask, col].str.replace(r'\s+', ' ', regex=True)

        # Report changes
        new_unique = len(result_df[col].unique())
        null_count = result_df[col].isnull().sum()

        print(f"   Original unique values: {original_unique}")
        print(f"   New unique values: {new_unique}")
        print(f"   Null values created: {null_count}")
        print(f"   Sample cleaned values: {result_df[col].dropna().unique()[:5].tolist()}")

    return result_df


# Alternative version with more customization options
def clean_object_columns_advanced(df, columns=None, columns_case_styles=None, default_case_style='title',
                                      custom_missing_values=None, inplace=False):
    """
    Advanced clean with more options.

    Parameters:
    case_style (str): 'title', 'upper', 'lower', or 'original'
    custom_missing_values (list): Custom list of values to treat as missing
    """

    import pandas as pd
    import numpy as np

    # Default missing values
    missing_values = ['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NULL', 'null',
                      'None', 'none', 'NaN', 'nan', '-', '?', 'unknown', 'UNKNOWN']

    # Add custom missing values if provided
    if custom_missing_values:
        missing_values.extend(custom_missing_values)

    result_df = df if inplace else df.copy()

    if columns is None:
        columns = result_df.select_dtypes(include=['object']).columns

    for col in columns:
        # Convert to string and strip
        result_df[col] = result_df[col].astype(str).str.strip()

        # Replace missing values
        result_df[col] = result_df[col].replace(missing_values, np.nan)

        # Apply case transformation
        mask = result_df[col].notna()

        case_style = columns_case_styles.get(col) or default_case_style

        if case_style == 'title':
            result_df.loc[mask, col] = result_df.loc[mask, col].str.title()
        elif case_style == 'upper':
            result_df.loc[mask, col] = result_df.loc[mask, col].str.upper()
        elif case_style == 'lower':
            result_df.loc[mask, col] = result_df.loc[mask, col].str.lower()
        # 'original' case - do nothing

        # Clean up extra spaces
        result_df.loc[mask, col] = result_df.loc[mask, col].str.replace(r'\s+', ' ', regex=True)

    return result_df


# Usage examples:
# show_unique_values_for_object_columns(sales)
# sales_cleaned = clean_object_columns(sales)
# sales_cleaned = clean_object_columns(sales, columns=['Group', 'State'], case_style='upper')




if __name__ == "__main__":
    print(tu.bold_and_colored_text("hello","red"))
