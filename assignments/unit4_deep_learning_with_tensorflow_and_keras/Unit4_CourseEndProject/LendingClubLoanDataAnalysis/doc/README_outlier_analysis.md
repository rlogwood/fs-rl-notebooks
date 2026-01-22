# Class Imbalance Ratios - What's Considered Imbalanced?

Great question! Let me break down what ratios are considered imbalanced in binary classification:

## Classification of Imbalance Levels

Here's how the machine learning community typically categorizes class imbalance based on the **majority-to-minority ratio**:

### 📊 Imbalance Severity Guide

| Ratio (A:B) | Imbalance Level | Action Required | Example |
|-------------|----------------|-----------------|---------|
| **1:1 to 1.5:1** | ✅ **Balanced** | None needed | 50% vs 50% or 60% vs 40% |
| **1.5:1 to 3:1** | ⚠️ **Slight Imbalance** | Monitor performance | 70% vs 30% |
| **3:1 to 9:1** | 🟡 **Moderate Imbalance** | Use class weights or resampling | 80% vs 20% |
| **9:1 to 99:1** | 🟠 **Severe Imbalance** | Special techniques required | 95% vs 5% |
| **>99:1** | 🔴 **Extreme Imbalance** | Advanced methods essential | 99% vs 1% |

## Your Dataset Example

Looking at your loan data:
- **Class 0 (Paid):** 8,045 samples (84%)
- **Class 1 (Default):** 1,533 samples (16%)
- **Ratio:** 5.25:1

```python
# Your current imbalance
imbalance_ratio = 8045 / 1533  # = 5.25:1
```


**Verdict:** Your dataset falls into the **Moderate Imbalance** category (between 3:1 and 9:1), which definitely requires special handling! 🟡

## General Guidelines

### When to Worry About Imbalance:

1. **Ratio ≥ 3:1 (75% vs 25%)** → Start using techniques
   - Class weights
   - Stratified sampling
   - Appropriate evaluation metrics (Precision, Recall, F1, AUC)

2. **Ratio ≥ 9:1 (90% vs 10%)** → Aggressive handling needed
   - SMOTE or other oversampling
   - Undersampling majority class
   - Ensemble methods
   - Anomaly detection approaches

3. **Ratio ≥ 99:1 (99% vs 1%)** → Extreme measures
   - Treat as anomaly detection problem
   - One-class SVM
   - Isolation Forest
   - Cost-sensitive learning with heavy penalties

## Quick Formula

Here's a simple way to calculate and interpret your imbalance ratio:

```python
import numpy as np

def check_imbalance(class_counts):
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

# Example with your data
result = check_imbalance({0: 8045, 1: 1533})
for key, value in result.items():
    print(f"{key:.<25} {value}")
```


**Output:**
```
ratio.................... 5.25:1
majority_class........... 8045
minority_class........... 1533
minority_percentage...... 16.01%
severity................. Moderate Imbalance 🟡
recommended_action....... Use class weights or resampling
```


## Key Takeaway

**Rule of Thumb:** If your minority class represents less than **20-25%** of your dataset (ratio > 3:1 or 4:1), you should treat it as imbalanced and apply appropriate techniques.

Your dataset at **5.25:1** absolutely qualifies as imbalanced and needs the class weighting approach you've already implemented! 🎯

---
# Understanding PyCharm's Data Quality Issues Detection

Great question! The "Dataset Quality Issues" panel in PyCharm's data viewer is a helpful feature for identifying **outliers** in your dataset. Let me explain what you're seeing and how to explore it.

## What Are These "Outliers"?

The notification shows **outliers detected** in 11 columns. These aren't errors or missing values—they're **statistical outliers**: values that fall significantly outside the typical range for that column.

### What the Percentages Mean:

- **credit.policy: 1868 (19.5%)** → 1,868 rows have outlier values in this column
- **int.rate: 51 (0.5%)** → 51 rows have unusually high/low interest rates
- **fico: 6 (0.1%)** → Only 6 FICO scores are statistical outliers
- **not.fully.paid: 1533 (16.0%)** → This is your class imbalance (minority class)

## What Does "Fix with AI" Do?

The **"Fix with AI"** button is an AI-powered feature that:

1. **Analyzes the outliers** in context of your dataset
2. **Suggests transformations** or handling strategies
3. **Generates code** (in a new cell) with recommendations like:
   - Capping/winsorizing extreme values
   - Log transformations for skewed data
   - Removing or flagging problematic rows
   - Imputation strategies

**Important:** It **doesn't automatically modify your data**—it creates suggestions you can review and apply manually.

## How to Explore These Issues Manually

Here are several ways to investigate outliers yourself:

### Method 1: Using PyCharm's Data Viewer

1. **Click on individual columns** in the outlier list
2. The data viewer will **highlight outlier rows**
3. You can **sort** or **filter** to see them

### Method 2: Manual Statistical Analysis

Here's code to detect and explore outliers yourself:

```python
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

# Analyze all numeric columns
print("=" * 70)
print("OUTLIER ANALYSIS (IQR Method)")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    result = detect_outliers_iqr(df, col)
    if result['total_outliers'] > 0:
        print(f"\n{col}:")
        print(f"  Outliers: {result['total_outliers']} ({result['percentage']})")
        print(f"  Valid range: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")
        print(f"  Actual range: [{df[col].min():.2f}, {df[col].max():.2f}]")
```


### Method 3: Visualize Outliers with Box Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Select columns to visualize
cols_to_plot = ['int.rate', 'installment', 'dti', 'revol.bal', 'fico']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold')

for idx, col in enumerate(cols_to_plot):
    row = idx // 3
    col_idx = idx % 3
    
    axes[row, col_idx].boxplot(df[col].dropna())
    axes[row, col_idx].set_title(col, fontweight='bold')
    axes[row, col_idx].set_ylabel('Value')
    axes[row, col_idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


### Method 4: Detailed Outlier Report

```python
def outlier_summary(df):
    """
    Generate comprehensive outlier report
    """
    print("=" * 70)
    print("OUTLIER SUMMARY REPORT")
    print("=" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        if len(outliers) > 0:
            print(f"\n{col}:")
            print(f"  Count: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
            print(f"  Mean (all): {df[col].mean():.2f}")
            print(f"  Mean (outliers): {outliers[col].mean():.2f}")
            print(f"  Min outlier: {outliers[col].min():.2f}")
            print(f"  Max outlier: {outliers[col].max():.2f}")
            print(f"  Valid range: [{lower:.2f}, {upper:.2f}]")

outlier_summary(df)
```


### Method 5: Examine Specific Outlier Rows

```python
# Get rows with outliers in multiple columns
def find_multi_outlier_rows(df, threshold=3):
    """
    Find rows that are outliers in multiple columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = pd.Series(0, index=df.index)
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        is_outlier = (df[col] < lower) | (df[col] > upper)
        outlier_counts += is_outlier.astype(int)
    
    # Find rows with outliers in 'threshold' or more columns
    problematic_rows = df[outlier_counts >= threshold]
    
    print(f"\nRows with outliers in {threshold}+ columns: {len(problematic_rows)}")
    if len(problematic_rows) > 0:
        print("\nSample of problematic rows:")
        print(problematic_rows.head())
    
    return problematic_rows

multi_outliers = find_multi_outlier_rows(df, threshold=3)
```


## Should You "Fix" These Outliers?

**For your loan data, probably NOT!** Here's why:

1. **not.fully.paid (16%)** → This is your target variable's class imbalance, not an outlier issue
2. **credit.policy (19.5%)** → Legitimate binary feature (0 or 1)
3. **Real-world data** → Loans naturally have extreme values:
   - Very low/high interest rates
   - Unusual FICO scores
   - Large revolving balances

### When to Address Outliers:

- ✅ **If they're data errors** (e.g., FICO score = 9999)
- ✅ **If they severely hurt model performance**
- ✅ **If they violate domain knowledge** (e.g., negative loan amounts)

### When to KEEP Outliers:

- ✅ **They represent real, valid data** (your case!)
- ✅ **They're important for prediction** (extreme values might indicate defaults)
- ✅ **Using tree-based or neural network models** (they handle outliers well)

## My Recommendation

**Don't use "Fix with AI" for this dataset.** Your outliers are legitimate financial data points. Instead:

1. ✅ **Keep them** —they're valuable for prediction
2. ✅ **Use StandardScaler** (you already are!)—handles scale differences
3. ✅ **Monitor model performance**—your model is working fine with them

The outlier detection is a helpful warning, but not every dataset needs outlier treatment! 🎯

---
Excellent question! **Yes, imbalance checking is absolutely possible and important for multi-class classification**, not just binary classification. However, the interpretation and handling strategies differ slightly.

## Handling Multi-Class Imbalance

Here's an enhanced version of the `check_imbalance` function that works for **any number of classes**:

```python
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


# Example usage with your binary data
binary_result = check_imbalance({0: 8045, 1: 1533})
```


## Multi-Class Examples

### Example 1: Balanced Multi-Class
```python
# Well-balanced 3-class problem
balanced_multiclass = {
    'Class_A': 1000,
    'Class_B': 950,
    'Class_C': 1050
}
result = check_imbalance(balanced_multiclass)
# Output: Balanced ✅ (1.11:1 ratio)
```


### Example 2: Moderate Multi-Class Imbalance
```python
# Image classification with some imbalance
image_classes = {
    'cats': 5000,
    'dogs': 4500,
    'birds': 1200,  # Moderate minority
    'fish': 800     # Severe minority
}
result = check_imbalance(image_classes)
# Output: Moderate-to-Severe Imbalance 🟡/🟠 (6.25:1 ratio)
```


### Example 3: Severe Multi-Class Imbalance
```python
# Medical diagnosis with rare conditions
medical_diagnosis = {
    'healthy': 9000,
    'condition_A': 500,
    'condition_B': 300,
    'rare_disease': 50  # Extreme minority
}
result = check_imbalance(medical_diagnosis)
# Output: Extreme Imbalance 🔴 (180:1 ratio)
```


## Key Differences for Multi-Class

### 1. **Multiple Ratios to Consider**
In multi-class problems, you should look at:
- **Worst-case ratio**: Majority vs. minority class
- **Pairwise ratios**: Between specific class pairs
- **Average imbalance**: Overall distribution uniformity

### 2. **Handling Strategies**

**For Moderate Multi-Class Imbalance (3:1 to 9:1):**
- Use `class_weight='balanced'` in sklearn
- Stratified sampling
- Per-class evaluation metrics

**For Severe Multi-Class Imbalance (>9:1):**
- SMOTE or ADASYN (adaptive synthetic sampling)
- Class-specific augmentation
- Focal loss (especially for deep learning)
- Hierarchical classification

### 3. **Computing Class Weights for Multi-Class**

```python
from sklearn.utils.class_weight import compute_class_weight

# Your multi-class target
y_multiclass = [0, 1, 2, 0, 1, 2, 0, 0, 0, 2, ...]  # Multiple classes

# Compute balanced weights
classes = np.unique(y_multiclass)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_multiclass
)

# Convert to dict for Keras/sklearn
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)
# Output: {0: 0.75, 1: 1.2, 2: 3.5, ...}
```


## Multi-Class Metrics to Monitor

Unlike binary classification, multi-class requires different metrics:

```python
from sklearn.metrics import classification_report

# For multi-class, specify average method
print(classification_report(y_true, y_pred, 
                          target_names=['Class_0', 'Class_1', 'Class_2']))

# Key metrics:
# - macro avg: Unweighted mean (treats all classes equally)
# - weighted avg: Weighted by support (accounts for imbalance)
# - per-class precision/recall
```


## Summary

| Aspect | Binary Imbalance | Multi-Class Imbalance |
|--------|-----------------|----------------------|
| **Check ratio** | ✅ Majority vs. minority | ✅ Worst-case ratio across all classes |
| **Threshold** | Same thresholds (3:1, 9:1, etc.) | Same thresholds apply |
| **Complexity** | Single ratio to monitor | Multiple ratios to consider |
| **Solutions** | Class weights, SMOTE | Same + class-specific strategies |
| **Metrics** | Precision, Recall, F1, AUC | Per-class + macro/weighted averages |

**Bottom line**: The same imbalance thresholds (3:1, 9:1, 99:1) apply to multi-class problems, but you need to consider the **worst-case ratio** (majority vs. smallest minority) as your primary indicator! 🎯
