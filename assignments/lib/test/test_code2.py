from typing import List, Tuple
import pandas as pd

# Example dataframe (replace df with your actual dataframe)
data = {
    'Column1': [' ', 'SomeValue', 'NA'],
    'Column2': ['One', 'N/A', '-'],
    'Column3': ['null', 'None', '?']
}
df = pd.DataFrame(data)

#DEFAULT_MISSING_VALUES: Tuple[str] = Tuple(["", "NA", "N/A", "-", "null", "None", "?"])
DEFAULT_MISSING_VALUES = ("", "NA", "N/A", "-", "null", "None", "?")
DEFAULT_NEW_VALUE = "unknown"

def replace_missing_values(df: pd.Series,
    missing_values: List[str] = DEFAULT_MISSING_VALUES,
    new_value: str = DEFAULT_NEW_VALUE) -> pd.Series:
    l = lambda s: s.str.strip().isin(missing_values)
    df_unknown = df.copy()
    df_unknown[df.apply(l)] = new_value
    return df_unknown

#df_unknown = df.copy()
#df_unknown[df.apply(match_condition)] = new_value

udf = replace_missing_values(df)
print("** udf\n**")
print(udf)


#df[df.apply(match_condition)] = "unknown"
#print("** df\n**")
#print(df)

