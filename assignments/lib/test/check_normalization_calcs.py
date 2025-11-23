min_sales = sales['Sales'].min()
max_sales = sales['Sales'].max()
min_unit = sales['Unit'].min()
max_unit = sales['Unit'].max()

sales_normal_denominator = max_sales - min_sales
unit_normal_denominator = max_unit - min_unit

for index, row in sales_normalized.iloc[:10].iterrows():
    # Your code here
    print(f"Index: {index}, Sales: {row['Sales']} ({row['Sales_normalized']}), Unit: {row['Unit']} ({row['Unit_normalized']})")
    normalized_sales = (row['Sales'] - min_sales) / sales_normal_denominator
    normalized_unit = (row['Unit'] - min_unit) / unit_normal_denominator

    print(f"calcd: {index}: Sales: {row['Sales']} ({normalized_sales})  Unit: {row['Unit']} ({normalized_unit})")
