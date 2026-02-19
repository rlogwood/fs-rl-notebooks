# Reciver Operating Characteristic (ROC) Curve

A **ROC curve** (Receiver Operating Characteristic curve) is a graphical plot that visualizes the performance of a binary classifier across all possible classification thresholds. [coursera](https://www.coursera.org/articles/what-is-roc-curve)

### What It Shows
The ROC curve plots the **True Positive Rate (TPR)**—also called sensitivity or recall—on the Y-axis against the **False Positive Rate (FPR)**—also called 1 - specificity—on the X-axis. [en.wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

- **True Positive Rate:** TP / (TP + FN) — the proportion of actual positives correctly identified
- **False Positive Rate:** FP / (FP + TN) — the proportion of actual negatives incorrectly classified as positive [en.wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
### How It Works
Each point on the curve represents a specific **decision threshold**: [evidentlyai](https://www.evidentlyai.com/classification-metrics/explain-roc-curve)

- A **strict threshold** (high confidence needed to predict positive) gives points toward the bottom-left (lower TPR, lower FPR)
- A **loose threshold** (low confidence needed) gives points toward the top-right (higher TPR, higher FPR) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8831439/)

By varying the threshold from 0 to 1, you trace out the full curve showing the trade-off between catching true positives and avoiding false positives.
### Interpreting the Curve
- A **perfect classifier** hugs the top-left corner (100% TPR, 0% FPR)
- The **diagonal line** represents random guessing (AUC = 0.5)
- The **closer the curve to the top-left**, the better the model discriminates between classes [c3](https://c3.ai/glossary/data-science/receiver-operating-characteristic-roc-curve/)
### AUC: Area Under the Curve
The **AUC** (Area Under the ROC Curve) summarizes overall performance in a single number from 0 to 1: [coursera](https://www.coursera.org/articles/what-is-roc-curve)

- **AUC = 1:** Perfect classification
- **AUC = 0.5:** No better than random chance
- **AUC < 0.5:** Worse than random (model is inverted)

AUC can be interpreted as the **probability that the model ranks a random positive instance higher than a random negative instance**. [c3](https://c3.ai/glossary/data-science/receiver-operating-characteristic-roc-curve/)
### History and Use Cases
The ROC curve originated in **World War II** for analyzing radar signals to detect enemy objects. Today it's widely used in: [en.wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

- Medical diagnostics (disease testing)
- Machine learning model evaluation
- Binary classification problems [coursera](https://www.coursera.org/articles/what-is-roc-curve)

Unlike accuracy, ROC curves are **robust to imbalanced classes** because they evaluate performance across all thresholds rather than at a single cutoff point.

---

                                                                                  
### Precision and AUC measure fundamentally different things:                                                                                                
                                                                                                                                                           
- Precision
                                                                                                                                                           
  - Question it answers: "Of all the times the model predicted positive, how many were actually positive?"
  - Formula: True Positives / (True Positives + False Positives)                                                                                           
  - Depends on a threshold — by default 0.5. A prediction above the threshold is "positive," below is "negative."
  - Focuses on one class — it only evaluates the positive predictions.
  - Use case: When the cost of false positives is high (e.g., you don't want to wrongly deny a good loan applicant).

- AUC (Area Under the ROC Curve)

  - Question it answers: "How well does the model rank positive examples higher than negative examples, across all possible thresholds?"
  - Threshold-independent — it sweeps across every possible threshold from 0 to 1 and measures the trade-off between True Positive Rate and False Positive
  Rate at each.
  - Value of 0.5 = random guessing, 1.0 = perfect separation.
  - Use case: Overall model quality — "does the model generally assign higher probabilities to actual positives than to actual negatives?"
```text
  Key Distinction
  ┌──────────────────────────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐
  │                                      │              Precision              │                 AUC                 │
  ├──────────────────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
  │ Threshold-dependent?                 │ Yes (fixed at 0.5 by default)       │ No (evaluates all thresholds)       │
  ├──────────────────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
  │ Scope                                │ Only positive predictions           │ Both classes, holistically          │
  ├──────────────────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
  │ Can be high while missing positives? │ Yes (if model is very conservative) │ No (penalizes missed positives too) │
  └──────────────────────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘
  In practice: A model can have high precision but low AUC if it only predicts positive when it's very confident (few false positives, but many missed true
   positives). AUC gives you the bigger picture of how well the model separates the two classes overall.

```

- - -
# ROC Curve
A Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates the performance of a binary classifier model at various threshold settings. It is used to visualize the trade-off between the true positive rate (TPR) and the false positive rate (FPR) across different thresholds.

Key Concepts

- True Positive Rate (TPR): Also known as sensitivity or recall, it measures the proportion of actual positives correctly identified by the model.

- False Positive Rate (FPR): It measures the proportion of actual negatives incorrectly classified as positives.

- Area Under the Curve (AUC): This is a single scalar value that summarizes the performance of the classifier. A higher AUC indicates better model performance.

## How to Create a ROC Curve

To create a ROC curve, you plot pairs of TPR vs. FPR for every possible decision threshold of a classifier. Here is an example using Python:

```
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# True binary labels
y_true = [1, 1, 0, 0, 1, 0]

# Predicted probabilities
y_pred = [0.95, 0.90, 0.85, 0.81, 0.78, 0.70]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# we don't need thresholds here

roc_auc = auc(fpr, tpr)
```

```text
#aligned
actual    = [   1,    1,    0,    0,    1,    0]
predicted = [0.95, 0.90, 0.85, 0.81, 0.78, 0.70]
```



### Plot ROC curve
```
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

## Interpretation of ROC Curve

The ROC curve helps in understanding the trade-offs between TPR and FPR. The closer the curve follows the top-left corner, the better the model's performance. The AUC value quantifies this performance; an AUC of 1 indicates a perfect model, while an AUC of 0.5 suggests a model no better than random guessing.

### Applications and Considerations

ROC curves are widely used in various fields such as medicine, radiology, and machine learning to evaluate diagnostic tests and classification models. However, they may not be ideal for highly imbalanced datasets, where precision-recall curves might be more informative.

In summary, the ROC curve is a powerful tool for evaluating the performance of binary classifiers, providing insights into the trade-offs between sensitivity and specificity at different thresholds.


# Explanation of the above code
Great questions, Richard. These concepts click much better once you see what's happening under the hood. Let me walk through each piece.

## What `roc_curve(y_true, y_pred)` Does

This function takes your true labels and predicted **probabilities** (not hard 0/1 predictions), and it systematically tries every possible threshold to classify predictions as 0 or 1. For your data:

```
y_true = [   1,    1,    0,    0,    1,    0]
y_pred = [0.95, 0.90, 0.85, 0.81, 0.78, 0.70]
```

It sorts by predicted probability and then walks through thresholds. At each threshold, anything **≥ threshold** is predicted as class 1, anything below is predicted as class 0. For each threshold it computes:

- **TPR (True Positive Rate / Recall):** Of all actual positives, how many did we catch?  `TP / (TP + FN)`
- **FPR (False Positive Rate):** Of all actual negatives, how many did we falsely flag?  `FP / (FP + TN)`

Here's what that looks like concretely with your data (3 actual positives, 3 actual negatives):

| Threshold | Predicted as 1 | TP | FP | TPR | FPR |
|-----------|----------------|----|----|-----|-----|
| > 0.95 | none | 0 | 0 | 0.0 | 0.0 |
| 0.95 | [0.95] | 1 | 0 | 0.33 | 0.0 |
| 0.90 | [0.95, 0.90] | 2 | 0 | 0.67 | 0.0 |
| 0.85 | [0.95, 0.90, 0.85] | 2 | 1 | 0.67 | 0.33 |
| 0.81 | [0.95, 0.90, 0.85, 0.81] | 2 | 2 | 0.67 | 0.67 |
| 0.78 | [0.95, 0.90, 0.85, 0.81, 0.78] | 3 | 2 | 1.0 | 0.67 |
| 0.70 | all six | 3 | 3 | 1.0 | 1.0 |

The function returns three arrays: `fpr`, `tpr`, and `thresholds` (which your code discards into `_`). Each row in the table above becomes one point on the ROC curve.

## What `auc(fpr, tpr)` Does

This simply computes the **area under the curve** formed by those (FPR, TPR) points. It's the geometric area under the ROC plot. A perfect model has AUC = 1.0 (the curve hugs the top-left corner). A random coin-flip model has AUC = 0.5 (the diagonal dashed line in your plot). So AUC gives you a single number summarizing how well the model separates the two classes **across all possible thresholds**.

## Where Thresholds Come Into Play

This is the key insight: **a model that outputs probabilities doesn't make decisions by itself — YOU choose the threshold that converts probabilities into decisions.** The ROC curve shows you the trade-off at every possible threshold:

- **High threshold (e.g., 0.95):** Very conservative — few false positives, but you miss a lot of actual positives (low TPR, low FPR). Upper-left area of the table.
- **Low threshold (e.g., 0.70):** Very aggressive — you catch all positives, but you also flag many negatives incorrectly (high TPR, high FPR). Bottom of the table.

For your employee retention project, this is directly relevant. If you set a low threshold, you'd flag more employees as flight risks (catching more actual leavers but also incorrectly flagging loyal employees). A high threshold means you'd only flag the most obvious cases, missing some who actually leave.

The ROC curve lets you visualize this entire spectrum so you can pick the threshold that best fits your business needs, rather than being locked into the default 0.5.

One thing worth noting: your code assigns thresholds to `_`, but you can capture them with `fpr, tpr, thresholds = roc_curve(y_true, y_pred)` and print them out to see the exact values sklearn computed. That can be a useful exercise to verify the table above.