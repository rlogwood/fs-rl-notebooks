## Brief Summary

Once you've picked a threshold, you convert probabilities into hard 0/1 predictions, then measure three things: **Precision** (of everyone you flagged as leaving, how many actually left?), **Recall** (of everyone who actually left, how many did you catch?), and **F1** (a single number balancing the two). These tell you how your model performs at that *specific* threshold.

The **Precision-Recall curve** is like the ROC curve but focused on your minority class — it shows the precision vs. recall trade-off across all thresholds. **Average Precision** summarizes that curve into one number, just like AUC summarizes the ROC curve. For imbalanced datasets like yours, these are more informative than ROC/AUC because they don't get inflated by correctly predicting the majority class.

---

## Step-by-Step Explanation

### 1. Precision, Recall, and F1 at a Chosen Threshold

Think of it from two perspectives using your retention project:

**Precision** answers: "When I tell HR someone is leaving, how often am I right?"

```
Precision = TP / (TP + FP)
```

Low precision means HR wastes time intervening with employees who were never going to leave.

**Recall** answers: "Of all the employees who actually left, how many did I warn HR about?"

```
Recall = TP / (TP + FN)
```

Low recall means employees are leaving without HR ever being alerted.

**F1 Score** is the harmonic mean of the two — it punishes you harshly if either one is low:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Here's how you compute them at your chosen threshold:

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Apply YOUR threshold instead of the default 0.5
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.35).astype(int)

# Individual metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Or get everything at once
print(classification_report(y_test, predictions))
```

### 2. The Threshold Trade-Off

This is the critical insight — precision and recall pull in opposite directions:

- **Lower the threshold (e.g., 0.20):** You flag more people as flight risks. Recall goes up (you catch more leavers), but precision drops (more false alarms).
- **Raise the threshold (e.g., 0.80):** You only flag the most obvious cases. Precision goes up (when you flag someone, you're usually right), but recall drops (you miss many leavers).

You can visualize this directly:

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

plt.figure()
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs. Threshold')
plt.legend()
plt.show()
```

This plot shows you exactly where the two lines cross and helps you pick a threshold that matches your business priority. For retention, you probably want higher recall (don't miss leavers), accepting some precision loss.

### 3. Precision-Recall Curve

Instead of plotting against the threshold, the PR curve plots precision (y-axis) against recall (x-axis) directly:

```python
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(y_test, probabilities)
plt.title('Precision-Recall Curve')
plt.show()
```

**How to read it:** A perfect model hugs the top-right corner (precision=1, recall=1). A random model on imbalanced data would be a flat line at `y = n_positives / n_total` — so if only 15% of employees leave, a random classifier gives a flat line at 0.15.

**Why it's better than ROC for imbalanced data:** ROC uses FPR, which divides by total negatives. When you have 12,750 stayers and 2,250 leavers, even a large number of false positives looks like a small FPR. The PR curve uses precision, which directly penalizes false positives regardless of how many negatives exist.

### 4. Average Precision (AP)

Average Precision summarizes the PR curve into a single number, analogous to AUC for ROC:

```python
from sklearn.metrics import average_precision_score

ap = average_precision_score(y_test, probabilities)
print(f"Average Precision: {ap:.3f}")
```

**Interpreting it:** AP ranges from 0 to 1. A perfect model scores 1.0. The baseline for a random model equals the proportion of positives in your data (e.g., ~0.15 if 15% leave). So an AP of 0.60 is much more impressive than it might initially sound on an imbalanced dataset — it's well above the 0.15 baseline.

**Use AP for model comparison on imbalanced data** the same way you'd use AUC:

```python
ap_lr = average_precision_score(y_test, probs_logistic)
ap_xgb = average_precision_score(y_test, probs_xgboost)

print(f"Logistic Regression AP: {ap_lr:.3f}")
print(f"XGBoost AP:             {ap_xgb:.3f}")
```

### 5. Putting It All Together — A Practical Workflow

```python
# 1. Compare models using summary metrics
print(f"AUC:              {roc_auc_score(y_test, probabilities):.3f}")
print(f"Average Precision: {average_precision_score(y_test, probabilities):.3f}")

# 2. Visualize the PR curve to understand the trade-off space
precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

# 3. Pick a threshold based on business needs
#    e.g., "I want at least 80% recall"
target_recall = 0.80
idx = np.argmin(np.abs(recalls[:-1] - target_recall))
chosen_threshold = thresholds[idx]
print(f"Threshold for ~{target_recall:.0%} recall: {chosen_threshold:.3f}")
print(f"Precision at that threshold: {precisions[idx]:.3f}")

# 4. Evaluate at your chosen threshold
predictions = (probabilities >= chosen_threshold).astype(int)
print(classification_report(y_test, predictions))
```

Step 3 is the key moment — that's where your business judgment ("missing leavers is expensive, I want 80% recall") gets translated into a concrete threshold, and then you can see exactly what precision you're giving up to get there.