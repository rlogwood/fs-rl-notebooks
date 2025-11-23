tu.styled_display("Data Shape:")
print(sales.shape)

# show data types
tu.styled_display("Data Types:")
print(sales.dtypes)

# Standard NA Check
tu.styled_display("Standard NA Check:")
print(sales.isna().sum())

# Check object columns for bad values
obj_cols = sales.select_dtypes(include="object").agg(lambda s: s.str.strip().isin(["", "NA","N/A","-","null","None","?"]).sum())
tu.styled_display("Bad object Values Check:")
print(obj_cols)

#styled_display("Blank object columns Check:", font_size=30, color='orange', bold=True)
#print(sales.select_dtypes("object").apply(lambda s: s.str.strip().eq("").sum()))


