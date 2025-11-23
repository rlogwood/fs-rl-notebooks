import pandas as pd

# Example dataframe (replace df with your actual dataframe)
data = {
    'Column1': [' ', 'SomeValue', 'NA'],
    'Column2': ['One', 'N/A', '-'],
    'Column3': ['null', 'None', '?']
}
df = pd.DataFrame(data)

# Lambda function detection
def match_condition(s):
    return s.str.strip().isin(["", "NA", "N/A", "-", "null", "None", "?"])

# Update all matching values to 'unknown'
df[df.apply(match_condition)] = "unknown"

print(df)
