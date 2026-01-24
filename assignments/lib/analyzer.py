try:
    # When imported as part of a package
    from . import text_util as tu, utility as utl, imputer as im
except ImportError:
    # When run as a standalone script
    import text_util as tu
    import utility as utl
    import imputer as im



# def statistical_group_test(df, group_by_column, column, group_name):
#     # 3. Statistical significance test (ANOVA)
#     from scipy import stats
#     group_totals = df.groupby(group_by_column)[column].sum()
#     group_name_total = df[df[group_by_column] == group_name][column].sum()
#
#     print(f"group_by total: {group_totals[group_name]}")
#     print(f"by == total:    {group_name_total}")
#
#     if group_totals[group_name] == group_name_total:
#         print("they are equal")
#     else:
#         print("they are not equal")


def statistical_significance_for_groups(df, group_by_series): #, column, group_name):
    from scipy import stats

    group_names = group_by_series.groups.keys()

    # Extract each group's data and pass to f_oneway
    group_data = [group_by_series.get_group(name) for name in group_names]
    f_stat, p_value = stats.f_oneway(*group_data)

    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    if p_value < 0.05:
        print("✅ Significant difference exists between groups (p < 0.05)")
    else:
        print("❌ No significant difference between groups (p >= 0.05)")

def transaction_range_analysis(df, group_by_series):
    from scipy import stats
    tu.print_sub_heading("📏 Transaction Range Analysis:")
    group_names = group_by_series.groups.keys()

    for group in group_names:
        group_data = group_by_series.get_group(group)
        range_val = group_data.max() - group_data.min()
        print(f"{group:>8}: Range = ${range_val:,.2f} (${group_data.min():,.2f} to ${group_data.max():,.2f})")


# def wip_statistical_significance_test(df, group_by_column, column):
#     # 3. Statistical significance test (ANOVA)
#     from scipy import stats
#
#     column_by_column = 'Group'
#     column = 'Sales'
#
#     group_totals = df.groupby(group_by_column)[column]
#
#     mens_sales = df[df['Group'] == 'Men']['Sales']
#
#     women_sales = df[df['Group'] == 'Women']['df']
#     kids_sales = df[df['Group'] == 'Kids']['df']
#     seniors_sales = df[df['Group'] == 'Seniors']['Sales']
#
#     f_stat, p_value = stats.f_oneway(mens_sales, women_sales, kids_sales, seniors_sales)
#
#     print(f"\n🔬 STATISTICAL SIGNIFICANCE TEST (ANOVA):")
#     print("=" * 60)
#     print(f"F-statistic: {f_stat:.4f}")
#     print(f"P-value: {p_value:.6f}")
#     if p_value < 0.05:
#         print("✅ Significant difference exists between groups (p < 0.05)")
#     else:
#         print("❌ No significant difference between groups (p >= 0.05)")
#
#
def coefficient_of_variation(df, group_stats): ##group_by_column, column):
    # 2. Calculate coefficient of variation (relative variability)
    group_stats['cv'] = (group_stats['std'] / group_stats['mean']) * 100
    tu.print_sub_heading("📈 Coefficient of Variation - Higher = More Variable:")
    for group in group_stats.index:
        cv = group_stats.loc[group, 'cv']
        print(f"{group:>8}: {cv:6.2f}% (std: ${group_stats.loc[group, 'std']:,.2f})")

def group_by_stats_for_column(df, group_by_column, column):
    ### Variance Analysis Between Demographic Groups
    # Detailed variance analysis for demographic groups
    tu.print_heading(f"VARIANCE ANALYSIS for '{column}' when grouped by '{group_by_column}'")

    # 1. Calculate basic statistics by group

    grouped_by_series = df.groupby(group_by_column)[column]

    group_stats = grouped_by_series.agg([
        'count',  # number of transactions
        'sum',  # total sales
        'mean',  # average transaction
        'std',  # standard deviation
        'var',  # variance
        'min',  # minimum transaction
        'max',  # maximum transaction
        'median'  # median transaction
    ]).round(2)

    tu.print_sub_heading(f"📊 Statistics for '{column}' when grouped by '{group_by_column}':")
    print(group_stats)
    return [grouped_by_series, group_stats]



def percentile_analysis(df, group_by_series):
    import numpy as np

    tu.print_sub_heading("📊 Percentile Distribution:")
    percentiles = [25, 50, 75, 90, 95]

    group_names = group_by_series.groups.keys()
    for group in group_names:
        group_data = group_by_series.get_group(group)
        print(f"{group} Group:")
        for p in percentiles:
            val = np.percentile(group_data, p)
            print(f"  {p:2d}th percentile: ${val:,.2f}")
        print("")


def variance_visualization(df, group_by_series, group_by_column, column, x_label=None, y_label=None):
    # 6. Visualization of variance
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Box plot showing distribution and variance
    df.boxplot(column=column, by=group_by_column, ax=axes[0, 0])
    axes[0, 0].set_title(f'{column} Distribution by {group_by_column} (Box Plot)')
    x_label = x_label or group_by_column
    y_label = y_label or column
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel(y_label)

    # Violin plot for distribution shape
    sns.violinplot(data=df, x=x_label, y=y_label, ax=axes[0, 1])
    axes[0, 1].set_title(f'{column} Distribution Shape by {group_by_column}')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Histogram comparison
    group_names = group_by_series.groups.keys()
    for group_name in group_names:
        group_data = group_by_series.get_group(group_name)
        axes[1, 0].hist(group_data, alpha=0.7, label=group_name, bins=30)

    axes[1, 0].set_title(f'{column} Distribution Histograms')
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Standard deviation comparison
    group_std = group_by_series.std()

    std_title = f'Standard Deviation by {group_by_column} for {column}'
    tu.print_sub_heading(std_title)
    print(f"group_std: {group_std}")

    group_std.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title(std_title)
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel(f'Standard Deviation {y_label}')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def variance_decomposition(df, group_by_series, column):
    # 7. Variance decomposition
    tu.print_sub_heading("🔍 Variance Decomposition:")
    total_variance = df[column].var()
    within_group_variance = 0
    group_sizes = []

    group_names = group_by_series.groups.keys()
    for group in group_names:
        group_data = group_by_series.get_group(group)
        group_var = group_data.var()
        group_size = len(group_data)
        group_sizes.append(group_size)
        within_group_variance += (group_size - 1) * group_var
        print(f"{group:>8} variance: ${group_var:,.2f}")

    within_group_variance = within_group_variance / (sum(group_sizes) - 3)
    between_group_variance = total_variance - within_group_variance

    print(f"\nTotal variance: ${total_variance:,.2f}")
    print(f"Within-group variance: ${within_group_variance:,.2f}")
    print(f"Between-group variance: ${between_group_variance:,.2f}")
    print(f"Variance ratio (between/within): {between_group_variance / within_group_variance:.4f}")




# ANOVA - "Analysis of Variance"
# The Coefficient of Variation (CV) = (Standard Deviation / Mean) × 100
# confusing name as it's actually testing means!
# ANOVA stands for **"Analysis of Variance"** - but this name can be confusing because it seems like it should be testing variances, when it's actually testing **means**!
#
# **ANOVA tests means by analyzing variances.** Here's the logic:
#
# ### The Key Insight:
# To determine if group means are different, ANOVA compares two types of variance:
#
# 1. Between-group variance: How much the group means differ from each other
# 2. Within-group variance: How much individual data points vary within each group
#
# The Logic:
# - If group means are truly the same: Between-group variance should be small (just random fluctuation)
# - If group means are truly different: Between-group variance should be large compared to within-group variance
#
# The F-statistic Formula:
# F = Between-group variance / Within-group variance
#
#
# - **Large F**: Group means are significantly different
# - **Small F**: Group means are basically the same (like your 0.2826)
#
# ## Why Not Call It "Analysis of Means"?
#
# Because the **method** uses variance analysis to **test** the means. It's named after the statistical technique, not the thing being tested.
#
# ## Analogy:
# Think of it like using a **thermometer** to check if you have a **fever**:
# - The **tool** is measuring temperature (variance analysis)
# - The **goal** is determining health status (are the means different?)
# - You'd call it "temperature analysis" even though you're really testing for illness
#
# So ANOVA = "**Analysis of Variance** to test if means are equal"
#
# The name reflects the **how** (analyzing variances) rather than the **what** (comparing means)!


def anova_for_column(df, group_by_column, column):
    (grouped_by_series, group_stats) = group_by_stats_for_column(df, group_by_column, column)

    # debug info
    # utl.inspect_variable(grouped_by_series, "grouped_by_series")

    coefficient_of_variation(df, group_stats)
    statistical_significance_for_groups(df, grouped_by_series)
    transaction_range_analysis(df, grouped_by_series)
    percentile_analysis(df, grouped_by_series)
    variance_visualization(df, grouped_by_series, group_by_column, column)
    variance_decomposition(df, grouped_by_series, column)


# def imbalance_analysis(df, group_by_column, column):
#     (grouped_by_series, group_stats) = group_by_stats_for_column(df, group_by_column, column)
#     statistical_significance_for_groups(df, grouped_by_series)
#     transaction_range_analysis(df, grouped_by_series)
#     percentile_analysis(df, grouped_by_series)
#     imbalance_ratio_analysis(df, grouped_by_series)
#     p_value_analysis(df, grouped_by_series)


import pandas as pd
import numpy as np


def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR (Interquartile Range) method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return {
        'column': column,
        'total_outliers': len(outliers),
        'percentage': f"{(len(outliers) / len(df)) * 100:.1f}%",
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_indices': outliers.index.tolist()
    }


import numpy as np


def check_imbalance_v1(class_counts):
    """
    Check if dataset is imbalanced

    Parameters:
    -----------
    class_counts : dict or array-like
        Counts of each class {class_label: count}

    Returns:
    --------
    dict with ratio and severity assessment
    """
    if isinstance(class_counts, dict):
        counts = np.array(list(class_counts.values()))
    else:
        counts = np.array(class_counts)

    majority_count = counts.max()
    minority_count = counts.min()
    ratio = majority_count / minority_count

    # Determine severity
    if ratio < 1.5:
        severity = "Balanced ✅"
        action = "No special handling needed"
    elif ratio < 3:
        severity = "Slight Imbalance ⚠️"
        action = "Monitor metrics closely"
    elif ratio < 9:
        severity = "Moderate Imbalance 🟡"
        action = "Use class weights or resampling"
    elif ratio < 99:
        severity = "Severe Imbalance 🟠"
        action = "Special techniques required (SMOTE, heavy class weights)"
    else:
        severity = "Extreme Imbalance 🔴"
        action = "Consider anomaly detection approaches"

    return {
        'ratio': f"{ratio:.2f}:1",
        'majority_class': majority_count,
        'minority_class': minority_count,
        'minority_percentage': f"{(minority_count / counts.sum()) * 100:.2f}%",
        'severity': severity,
        'recommended_action': action
    }


import numpy as np
import pandas as pd


def check_imbalance(class_counts, verbose=True):
    """
    Check if dataset is imbalanced for binary or multi-class problems

    Parameters:
    -----------
    class_counts : dict, pd.Series, or array-like
        Counts of each class {class_label: count} or array of counts
    verbose : bool
        Whether to print detailed information

    Returns:
    --------
    dict with comprehensive imbalance analysis
    """
    # Convert to dict format
    if isinstance(class_counts, pd.Series):
        class_dict = class_counts.to_dict()
        counts = class_counts.values
    elif isinstance(class_counts, dict):
        class_dict = class_counts
        counts = np.array(list(class_counts.values()))
    else:
        counts = np.array(class_counts)
        class_dict = {i: count for i, count in enumerate(counts)}

    # Basic statistics
    total = counts.sum()
    n_classes = len(counts)
    majority_count = counts.max()
    minority_count = counts.min()
    ratio = majority_count / minority_count

    # Calculate imbalance ratio for each class vs majority
    majority_idx = counts.argmax()
    minority_idx = counts.argmin()

    # Determine severity based on worst-case ratio
    if ratio < 1.5:
        severity = "Balanced ✅"
        action = "No special handling needed"
    elif ratio < 3:
        severity = "Slight Imbalance ⚠️"
        action = "Monitor metrics closely"
    elif ratio < 9:
        severity = "Moderate Imbalance 🟡"
        action = "Use class weights or resampling"
    elif ratio < 99:
        severity = "Severe Imbalance 🟠"
        action = "Special techniques required (SMOTE, heavy class weights)"
    else:
        severity = "Extreme Imbalance 🔴"
        action = "Consider anomaly detection approaches"

    # Per-class analysis
    class_analysis = []
    class_labels = list(class_dict.keys())

    for i, (label, count) in enumerate(class_dict.items()):
        percentage = (count / total) * 100
        ratio_to_majority = majority_count / count if count > 0 else np.inf
        ratio_to_minority = count / minority_count if minority_count > 0 else np.inf

        class_analysis.append({
            'class': label,
            'count': count,
            'percentage': percentage,
            'ratio_to_majority': ratio_to_majority,
            'ratio_to_minority': ratio_to_minority
        })

    result = {
        'n_classes': n_classes,
        'total_samples': int(total),
        'majority_class': class_labels[majority_idx],
        'majority_count': int(majority_count),
        'minority_class': class_labels[minority_idx],
        'minority_count': int(minority_count),
        'minority_percentage': f"{(minority_count / total) * 100:.2f}%",
        'imbalance_ratio': f"{ratio:.2f}:1",
        'severity': severity,
        'recommended_action': action,
        'class_analysis': class_analysis
    }

    if verbose:
        print("=" * 70)
        print("CLASS IMBALANCE ANALYSIS")
        print("=" * 70)
        print(f"\nNumber of classes: {n_classes}")
        print(f"Total samples: {total:,}")
        print(f"\nClass Distribution:")
        print("-" * 70)

        # Sort by count for better visualization
        sorted_analysis = sorted(class_analysis, key=lambda x: x['count'], reverse=True)

        for item in sorted_analysis:
            bar_length = int(item['percentage'] / 2)  # Scale for display
            bar = "█" * bar_length
            print(f"  Class {item['class']:>10}: {item['count']:>8,} ({item['percentage']:>6.2f}%) {bar}")

        print("\n" + "-" * 70)
        print(f"Majority class: {result['majority_class']} ({majority_count:,} samples)")
        print(f"Minority class: {result['minority_class']} ({minority_count:,} samples)")
        print(f"Imbalance ratio: {result['imbalance_ratio']}")
        print(f"\nSeverity: {severity}")
        print(f"Recommended action: {action}")
        print("=" * 70)

    return result
