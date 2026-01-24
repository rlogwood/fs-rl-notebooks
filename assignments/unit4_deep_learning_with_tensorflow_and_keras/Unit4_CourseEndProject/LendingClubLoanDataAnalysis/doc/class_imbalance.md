# Determining class counts for a column
To get class counts for a DataFrame category column in pandas, you can use the `value_counts()` method. Here are the most common approaches:

## Basic Usage

```python
# Get absolute counts
class_counts = df['column_name'].value_counts()
print(class_counts)
```


## With Percentages

```python
# Get proportions (as decimals)
class_proportions = df['column_name'].value_counts(normalize=True)

# Get percentages
class_percentages = df['column_name'].value_counts(normalize=True) * 100
```


## Include Missing Values

```python
# Include NaN/None counts
class_counts = df['column_name'].value_counts(dropna=False)
```


## Sorted by Index (Category Names)

```python
# Sort by category name instead of count
class_counts = df['column_name'].value_counts().sort_index()
```


## Example from Your Notebook

Looking at your code, you're already using this correctly:

```python
# For the target variable
class_counts = df['not.fully.paid'].value_counts()
class_percentages = df['not.fully.paid'].value_counts(normalize=True) * 100

print(f"Paid (0):     {class_counts[0]:,} loans ({class_percentages[0]:.2f}%)")
print(f"Default (1):  {class_counts[1]:,} loans ({class_percentages[1]:.2f}%)")
```


## Using Your Helper Function

You've also created a nice helper function in your `analyzer.py`:

```python
import lib.analyzer as da

# Check imbalance with detailed analysis
result = da.check_imbalance(df['not.fully.paid'].value_counts())
```


This gives you a comprehensive breakdown including severity assessment and recommendations!
---
# Class Imbalance Ratios - What's Considered Imbalanced?

What ratios are considered imbalanced in binary classification:

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


---
Imbalance checking for multi-class classification**, as opposed to binary classification. 
The interpretation and handling strategies differ slightly.

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
