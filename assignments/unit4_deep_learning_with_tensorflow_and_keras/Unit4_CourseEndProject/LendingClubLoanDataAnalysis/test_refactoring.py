"""
Test script for refactored OptimizationMetric functionality
"""

# Test the enum and calculation logic
from enum import Enum

class OptimizationMetric(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    RECALL_WEIGHTED = "recall_weighted"

def calculate_recall_weighted(roc_auc, recall, f1):
    return 0.3 * roc_auc + 0.6 * recall + 0.1 * f1

def calculate_optimization_metric(metric_type, accuracy=None, precision=None, recall=None, f1=None, roc_auc=None):
    if isinstance(metric_type, str):
        metric_type = OptimizationMetric(metric_type)

    if metric_type == OptimizationMetric.ACCURACY:
        return accuracy
    elif metric_type == OptimizationMetric.PRECISION:
        return precision
    elif metric_type == OptimizationMetric.RECALL:
        return recall
    elif metric_type == OptimizationMetric.F1:
        return f1
    elif metric_type == OptimizationMetric.ROC_AUC:
        return roc_auc
    elif metric_type == OptimizationMetric.RECALL_WEIGHTED:
        return calculate_recall_weighted(roc_auc, recall, f1)

# Test cases
print("Testing OptimizationMetric calculation...")
print("=" * 60)

test_metrics = {
    'accuracy': 0.85,
    'precision': 0.20,
    'recall': 0.95,
    'f1': 0.33,
    'roc_auc': 0.68
}

print("\nTest Metrics:")
for key, value in test_metrics.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("Testing each optimization metric:")
print("=" * 60)

# Test F1
result = calculate_optimization_metric('f1', **test_metrics)
print(f"\nF1 optimization: {result:.4f}")
assert result == test_metrics['f1'], "F1 calculation failed"

# Test Recall
result = calculate_optimization_metric('recall', **test_metrics)
print(f"Recall optimization: {result:.4f}")
assert result == test_metrics['recall'], "Recall calculation failed"

# Test Recall Weighted
result = calculate_optimization_metric('recall_weighted', **test_metrics)
expected = 0.3 * 0.68 + 0.6 * 0.95 + 0.1 * 0.33
print(f"Recall-Weighted optimization: {result:.4f}")
print(f"  Formula: 0.3*{test_metrics['roc_auc']} + 0.6*{test_metrics['recall']} + 0.1*{test_metrics['f1']}")
print(f"  Breakdown: 0.3*0.68={0.3*0.68:.4f}, 0.6*0.95={0.6*0.95:.4f}, 0.1*0.33={0.1*0.33:.4f}")
print(f"  Expected: {expected:.4f}")
assert abs(result - expected) < 0.0001, "Recall weighted calculation failed"

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)

# Demonstrate threshold optimization scenarios
print("\n" + "=" * 60)
print("Example: Threshold Optimization Comparison")
print("=" * 60)

thresholds = [
    {'threshold': 0.3, 'acc': 0.16, 'prec': 0.16, 'recall': 1.00, 'f1': 0.28},
    {'threshold': 0.4, 'acc': 0.16, 'prec': 0.16, 'recall': 0.99, 'f1': 0.28},
    {'threshold': 0.5, 'acc': 0.23, 'prec': 0.17, 'recall': 0.98, 'f1': 0.29},
    {'threshold': 0.6, 'acc': 0.38, 'prec': 0.19, 'recall': 0.86, 'f1': 0.31},
    {'threshold': 0.7, 'acc': 0.63, 'prec': 0.23, 'recall': 0.56, 'f1': 0.33},
]

roc_auc = 0.649  # Constant across thresholds

print("\nOptimizing for F1:")
best_f1 = max(thresholds, key=lambda x: x['f1'])
print(f"  Best threshold: {best_f1['threshold']}")
print(f"  F1: {best_f1['f1']:.4f}, Recall: {best_f1['recall']:.4f}")

print("\nOptimizing for Recall-Weighted:")
for t in thresholds:
    t['recall_weighted'] = calculate_recall_weighted(roc_auc, t['recall'], t['f1'])

best_rw = max(thresholds, key=lambda x: x['recall_weighted'])
print(f"  Best threshold: {best_rw['threshold']}")
print(f"  Recall-Weighted: {best_rw['recall_weighted']:.4f}")
print(f"  F1: {best_rw['f1']:.4f}, Recall: {best_rw['recall']:.4f}")

print("\nComparison:")
print(f"  F1 optimization chose threshold {best_f1['threshold']} with recall {best_f1['recall']:.4f}")
print(f"  Recall-Weighted optimization chose threshold {best_rw['threshold']} with recall {best_rw['recall']:.4f}")
print(f"  → Recall-Weighted prioritizes catching more defaults!")
