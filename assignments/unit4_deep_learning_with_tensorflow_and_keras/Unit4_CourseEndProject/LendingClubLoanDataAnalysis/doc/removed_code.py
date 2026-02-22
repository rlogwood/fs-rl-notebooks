me.print_classification_metrics(y_test, y_pred)

tu.print_heading("FINAL MODEL PERFORMANCE SUMMARY")

# Get final predictions with best threshold
best_threshold = results['best_threshold']['threshold']
y_pred_proba = results['y_pred_proba']
y_pred_final = (y_pred_proba > best_threshold).astype("int32")
cm_final = confusion_matrix(y_test, y_pred_final)

print(tu.bold_text(f"\nUsing Optimal Threshold: {best_threshold}"))
print(tu.bold_text("\nFinal Confusion Matrix:"))
print(cm_final)

print(tu.bold_text("\nFinal Classification Report:"))
print(classification_report(
    y_test, y_pred_final,
    target_names=['Paid (0)', 'Default (1)'],
    zero_division=0
))

# Calculate improvement
defaults_caught = cm_final[1, 1]
total_defaults = cm_final[1, 0] + cm_final[1, 1]
recall_final = defaults_caught / total_defaults
roc_auc = roc_auc_score(y_test, y_pred_proba)

tu.print_heading("KEY ACHIEVEMENTS")
print(f"✓ Model now detects {recall_final * 100:.1f}% of actual defaults")
print(f"✓ Caught {defaults_caught} out of {total_defaults} defaults in test set")
print(f"✓ AUC-ROC Score: {roc_auc:.4f} (Good discriminative ability)")


# # MODEL IMPROVEMENTS
#
# ## Exploring Alternative Approaches to Improve Performance
#
# Our baseline neural network achieved AUC=0.659. Let's try different approaches:
#
# 1. **Feature Engineering** - Create new predictive features
# 2. **Random Forest** - Often works better on tabular data
# 3. **XGBoost** - Industry standard for structured data
# 4. **Model Comparison** - Compare all approaches
