# lib/data_cleaner.py
"""
Data cleaning utilities for handling missing values and standardizing columns.

Functions:
    clean_object_columns: Handle string/category columns
    clean_numeric_columns: Handle numeric columns with NaN
    clean_dataframe: Convenience function that calls both

Enums:
    CaseStyle: Valid case transformation options
    FillStrategy: Valid strategies for handling NaN in numeric columns
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple


class CaseStyle(Enum):
    """Case transformation options for string columns."""
    ORIGINAL = 'original'  # No transformation
    TITLE = 'title'        # Title Case
    UPPER = 'upper'        # UPPER CASE
    LOWER = 'lower'        # lower case


class FillStrategy(Enum):
    """Strategies for handling NaN values in numeric columns."""
    NONE = 'none'      # Leave NaN as-is
    DROP = 'drop'      # Drop rows with NaN
    MEAN = 'mean'      # Fill with column mean
    MEDIAN = 'median'  # Fill with column median
    MODE = 'mode'      # Fill with column mode
    ZERO = 'zero'      # Fill with 0
    VALUE = 'value'    # Fill with custom fill_value


# Common string representations of missing values
DEFAULT_MISSING_VALUES = ("", " ", "NA", "N/A", "na", "n/a", "NULL", "null",
                          "None", "none", "NaN", "nan", "-", "?",
                          "unknown", "UNKNOWN")


def clean_object_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    missing_values: Tuple[str, ...] = DEFAULT_MISSING_VALUES,
    replacement: Optional[str] = None,
    case_style: CaseStyle = CaseStyle.ORIGINAL,
    strip_whitespace: bool = True
) -> pd.DataFrame:
    """
    Clean string/object columns by replacing missing value placeholders and standardizing format.

    Parameters:
        df: The DataFrame to clean
        columns: Specific columns to clean. If None, cleans all object columns
        missing_values: String values to treat as missing
        replacement: Value to replace missing with. None means np.nan
        case_style: CaseStyle enum value (ORIGINAL, TITLE, UPPER, LOWER)
        strip_whitespace: Whether to strip leading/trailing whitespace

    Returns:
        Cleaned DataFrame

    Example:
        df = clean_object_columns(df, case_style=CaseStyle.TITLE)
    """
    result = df.copy()

    # Determine which columns to process
    if columns is None:
        columns = result.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        # Filter to only object/category columns that exist
        valid_cols = result.select_dtypes(include=['object', 'category']).columns
        columns = [col for col in columns if col in valid_cols]

    if not columns:
        return result

    for col in columns:
        # Step 1: Strip whitespace
        if strip_whitespace:
            result[col] = result[col].astype(str).str.strip()

        # Step 2: Replace missing value placeholders
        mask = result[col].isin(missing_values)
        if replacement is None:
            # Note: this creates a NaN for missing values, pd.isna(), df.dropna() and df.fillna() will work as expected
            # on these converted columns
            result.loc[mask, col] = np.nan
        else:
            result.loc[mask, col] = replacement

        # Step 3: Apply case transformation (only to non-null values)
        if case_style != CaseStyle.ORIGINAL:
            non_null_mask = result[col].notna()
            if case_style == CaseStyle.TITLE:
                result.loc[non_null_mask, col] = result.loc[non_null_mask, col].str.title()
            elif case_style == CaseStyle.UPPER:
                result.loc[non_null_mask, col] = result.loc[non_null_mask, col].str.upper()
            elif case_style == CaseStyle.LOWER:
                result.loc[non_null_mask, col] = result.loc[non_null_mask, col].str.lower()

    return result


def clean_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: FillStrategy = FillStrategy.NONE,
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle NaN values in numeric columns.

    Parameters:
        df: The DataFrame to clean
        columns: Specific columns to clean. If None, cleans all numeric columns
        strategy: FillStrategy enum value for handling NaN:
            - NONE: Leave NaN as-is
            - DROP: Drop rows with NaN (use with caution)
            - MEAN: Fill with column mean
            - MEDIAN: Fill with column median
            - MODE: Fill with column mode
            - ZERO: Fill with 0
            - VALUE: Fill with fill_value parameter
        fill_value: Value to use when strategy=FillStrategy.VALUE

    Returns:
        Cleaned DataFrame

    Example:
        df = clean_numeric_columns(df, strategy=FillStrategy.MEDIAN)
    """
    result = df.copy()

    # Determine which columns to process
    if columns is None:
        columns = result.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter to only numeric columns that exist
        valid_cols = result.select_dtypes(include=['number']).columns
        columns = [col for col in columns if col in valid_cols]

    if not columns or strategy == FillStrategy.NONE:
        return result

    if strategy == FillStrategy.DROP:
        result = result.dropna(subset=columns)
    elif strategy == FillStrategy.MEAN:
        for col in columns:
            result[col] = result[col].fillna(result[col].mean())
    elif strategy == FillStrategy.MEDIAN:
        for col in columns:
            result[col] = result[col].fillna(result[col].median())
    elif strategy == FillStrategy.MODE:
        for col in columns:
            mode_val = result[col].mode()
            if len(mode_val) > 0:
                result[col] = result[col].fillna(mode_val[0])
    elif strategy == FillStrategy.ZERO:
        result[columns] = result[columns].fillna(0)
    elif strategy == FillStrategy.VALUE:
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy=FillStrategy.VALUE")
        result[columns] = result[columns].fillna(fill_value)

    return result


def clean_dataframe(
    df: pd.DataFrame,
    object_config: Optional[Dict] = None,
    numeric_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Convenience function to clean both object and numeric columns.

    Parameters:
        df: The DataFrame to clean
        object_config: Keyword arguments for clean_object_columns
            Example: {'case_style': CaseStyle.LOWER, 'replacement': 'unknown'}
        numeric_config: Keyword arguments for clean_numeric_columns
            Example: {'strategy': FillStrategy.MEAN}

    Returns:
        Cleaned DataFrame

    Example:
        df = clean_dataframe(df,
            object_config={'case_style': CaseStyle.TITLE},
            numeric_config={'strategy': FillStrategy.MEDIAN}
        )
    """
    result = df.copy()

    if object_config is not None:
        result = clean_object_columns(result, **object_config)

    if numeric_config is not None:
        result = clean_numeric_columns(result, **numeric_config)

    return result
