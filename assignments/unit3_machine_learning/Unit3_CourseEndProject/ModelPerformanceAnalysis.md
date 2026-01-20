# Model Performance and Anlysis

# Table of Contents
<!-- TOC -->
* [Model Performance and Anlysis](#model-performance-and-anlysis)
* [Table of Contents](#table-of-contents)
* [Performance Termnology](#performance-termnology)
    * [Confusion Matrix Terms](#confusion-matrix-terms-)
    * [Rate Terms](#rate-terms-)
    * [Visual Reference (Confusion Matrix)](#visual-reference-confusion-matrix)
    * [Employee Turnover Example](#employee-turnover-example-)
    * [Classification Metrics Summary](#classification-metrics-summary)
    * [Metric Definitions](#metric-definitions)
    * [Key Contrasts](#key-contrasts-)
    * [How to Choose the Right Metric](#how-to-choose-the-right-metric-)
* [How to Choose the Right Metric](#how-to-choose-the-right-metric--1)
    * [Use Recall when:](#use-recall-when-)
    * [Use Precision when:](#use-precision-when-)
    * [Use F1-Score when:](#use-f1-score-when-)
    * [Use Accuracy when:](#use-accuracy-when-)
    * [Use ROC-AUC when:](#use-roc-auc-when-)
* [Best model metric for employee turnover](#best-model-metric-for-employee-turnover-)
    * [We prioritize Recall because:](#we-prioritize-recall-because-)
* [Understaning Recal/Specificity](#understaning-recalspecificity)
  * [Recall = Sensitivity (Same Metric, Different Names)](#recall--sensitivity-same-metric-different-names)
  * [Why Two Names?](#why-two-names)
    * ["Recall" (Machine Learning/Information Retrieval)](#recall-machine-learninginformation-retrieval)
    * ["Sensitivity" (Medical/Diagnostic Testing)](#sensitivity-medicaldiagnostic-testing)
  * [The Intuition Behind "Sensitivity"](#the-intuition-behind-sensitivity)
  * [For Employee Retention](#for-employee-retention)
  * [The Companion Metric: Specificity](#the-companion-metric-specificity)
  * [Quick Reference](#quick-reference)
<!-- TOC -->

# Performance Termnology
[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                                 
### Confusion Matrix Terms                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                 
  | Term | Name           | Definition                                                                 |                                                                                                                                                         
  |------|----------------|----------------------------------------------------------------------------|                                                                                                                                                         
  | TP   | True Positive  | Model predicted positive, and it was actually positive (correct)           |                                                                                                                                                         
  | TN   | True Negative  | Model predicted negative, and it was actually negative (correct)           |                                                                                                                                                         
  | FP   | False Positive | Model predicted positive, but it was actually negative (Type I error)      |                                                                                                                                                         
  | FN   | False Negative | Model predicted negative, but it was actually positive (Type II error)     |                                                                                                                                                         
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)
### Rate Terms                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                 
  | Term | Name                | Formula           | Definition                                          |                                                                                                                                                       
  |------|---------------------|-------------------|-----------------------------------------------------|                                                                                                                                                       
  | TPR  | True Positive Rate  | TP / (TP + FN)    | Proportion of actual positives correctly identified (same as Recall/Sensitivity) |                                                                                                                          
  | FPR  | False Positive Rate | FP / (FP + TN)    | Proportion of actual negatives incorrectly flagged as positive |                                                                                                                                            
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)
###  Visual Reference (Confusion Matrix)
  
|                     | Predicted Positive | Predicted Negative |                                                                                                                                                                                                                   
|---------------------|--------------------|--------------------|                                                                                                                                                                                                                   
| **Actual Positive** | TP                 | FN                 |
| **Actual Negative** | FP                 | TN                 |                                                                                                                                                                                                                   
 
[↑ Back to TOC](#table-of-contents)
###   Employee Turnover Example                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                 
  | Term | In Your Context                                                    |                                                                                                                                                                                  
  |------|--------------------------------------------------------------------|                                                                                                                                                                                  
  | TP   | Predicted employee would leave, and they did leave                 |                                                                                                                                                                                  
  | TN   | Predicted employee would stay, and they did stay                   |                                                                                                                                                                                  
  | FP   | Predicted employee would leave, but they actually stayed           |                                                                                                                                                                                  
  | FN   | Predicted employee would stay, but they actually left (costly!)    |                                                                                                                                                                                  
  | TPR  | % of employees who left that the model correctly identified        |                                                                                                                                                                                  
  | FPR  | % of employees who stayed that the model incorrectly flagged       |                                                                                                                                                                                  
                                                                                             

[↑ Back to TOC](#table-of-contents)
### Classification Metrics Summary

  | Metric      | Formula                                         | Question It Answers                                        |                                                                                                                                 
  |-------------|-------------------------------------------------|------------------------------------------------------------|                                                                                                                                 
  | Accuracy    | (TP + TN) / Total                               | What % of all predictions were correct?                    |                                                                                                                                 
  | Precision   | TP / (TP + FP)                                  | When I predict positive, how often am I right?             |                                                                                                                                 
  | Recall      | TP / (TP + FN)                                  | What % of actual positives did I catch?                    |                                                                                                                                 
  | F1-Score    | 2 × (Precision × Recall) / (Precision + Recall) | What's the balanced trade-off between Precision and Recall? |                                                                                                                                
  | ROC-AUC     | Area under TPR vs FPR curve                     | How well does the model rank/separate classes overall?     |                                                                                                                                 
                                                                                                                                                                                                                                                                 
  (TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative)                                                                                                                                                                                     

[↑ Back to TOC](#table-of-contents)
### Metric Definitions
  | Metric | Definition |                                                                                                                                                                                                                                        
  |--------|------------|                                                                                                                                                                                                                                        
  | Accuracy | Percentage of all predictions that are correct |                                                                                                                                                                                                  
  | Precision | Of predicted positives, how many are actually positive |                                                                                                                                                                                         
  | Recall | Of actual positives, how many did we correctly identify |                                                                                                                                                                                           
  | F1-Score | Harmonic mean of Precision and Recall (balances both) |                                                                                                                                                                                           
  | ROC-AUC | Model's ability to distinguish between classes across all thresholds |   

> NOTE: ROC-AUC - "Area Under the Receiver Operating Characteristic Curve."     
>  - ROC: Receiver Operating Characteristic                                                                                                                                                                                                
>  - AUC: Area Under the Curve    
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)
### Key Contrasts                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                 
  | Aspect               | Accuracy   | Precision        | Recall             | F1-Score | ROC-AUC        |                                                                                                                                                      
  |----------------------|------------|------------------|--------------------|----------|----------------|                                                                                                                                                      
  | Class imbalance      | Misleading | Robust           | Robust             | Robust   | Robust         |                                                                                                                                                      
  | Threshold-dependent  | Yes        | Yes              | Yes                | Yes      | No             |                                                                                                                                                      
  | Focuses on           | Overall correctness | Avoiding false alarms | Catching all positives | Balance  | Ranking ability |                                                                                                                                   

[↑ Back to TOC](#table-of-contents)
### How to Choose the Right Metric                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                 
  | Metric    | Use When                                      | Examples                                           |                                                                                                                                             
  |-----------|-----------------------------------------------|----------------------------------------------------|                                                                                                                                             
  | Recall    | False negatives are costly                    | Disease detection, fraud, employee turnover        |                                                                                                                                             
  | Precision | False positives are costly                    | Spam filtering, expensive targeted interventions   |                                                                                                                                             
  | F1-Score  | Both error types matter equally               | General text classification, sentiment analysis    |                                                                                                                                             
  | Accuracy  | Classes are balanced, errors have equal cost  | Simple binary tasks with balanced data             |                                                                                                                                             
  | ROC-AUC   | Need threshold-independent comparison         | Credit scoring, risk stratification                |         

 
[↑ Back to TOC](#table-of-contents)
# How to Choose the Right Metric                                                                                                                                                                                                                                  
  The choice depends on business cost of errors:                                                                                                                                                                                                                  

[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                                 
### Use Recall when:                                                                                                                                                                                                                                                
  - False negatives are costly (missing a positive case is dangerous)                                                                                                                                                                                            
  - Examples: Disease detection, fraud detection, employee turnover (your case), security threats                                                                                                                                                                
  - Rationale: Better to flag more cases and investigate than miss one                                                                                                                                                                                           
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                                 
### Use Precision when:                                                                                                                                                                                                                                             
  - False positives are costly (acting on wrong predictions wastes resources)                                                                                                                                                                                    
  - Examples: Spam filtering (don't want legitimate emails in spam), targeted marketing (expensive interventions)                                                                                                                                                
  - Rationale: Only act when highly confident                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                                 
### Use F1-Score when:                                                                                                                                                                                                                                              
  - Both types of errors matter roughly equally                                                                                                                                                                                                                  
  - You need a single balanced metric for comparison                                                                                                                                                                                                             
  - Examples: General text classification, sentiment analysis                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                                 
### Use Accuracy when:                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                 
  - Classes are balanced (roughly 50/50 split)                                                                                                                                                                                                                   
  - All errors have equal cost                                                                                                                                                                                                                                   
  - Examples: Simple binary tasks with balanced data                                                                                                                                                                                                             
 
[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                                 
### Use ROC-AUC when:                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                 
  - You need to compare models before choosing a threshold                                                                                                                                                                                                       
  - You want to understand discrimination ability across all thresholds                                                                                                                                                                                          
  - You're building a probability-based ranking system                                                                                                                                                                                                           
  - Examples: Credit scoring, risk stratification                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                 
[↑ Back to TOC](#table-of-contents)                                                                                                                                                                                                                                                           
# Best model metric for employee turnover                                                                                                                                                                                                                              
[↑ Back to TOC](#table-of-contents)
### We prioritize Recall because:                                                                                                                                                                                                            
  - Missing an at-risk employee (false negative) → they leave → costs 50-200% of salary                                                                                                                                                                          
  - Flagging a stable employee (false positive) → unnecessary retention effort → minor cost                                                                                                                                                                      
                                                                                                                                                                                                                                                                 
  The asymmetric cost makes Recall the primary metric, with ROC-AUC as secondary to confirm overall model quality.

[↑ Back to TOC](#table-of-contents)
# Understaning Recal/Specificity
[↑ Back to TOC](#table-of-contents)
## Recall = Sensitivity (Same Metric, Different Names)

They're **exactly the same calculation**, just used in different fields:

```
Recall/Sensitivity = True Positives / (True Positives + False Negatives)
                   = TP / (TP + FN)
```

[↑ Back to TOC](#table-of-contents)
## Why Two Names?
[↑ Back to TOC](#table-of-contents)
### "Recall" (Machine Learning/Information Retrieval)
Comes from the question: **"Of all the relevant items, how many did we recall/retrieve?"**

In your case: Of all employees who left, how many did your model recall/remember to flag?
- 264 out of 398 employees who left = 66.3% recall

[↑ Back to TOC](#table-of-contents)
### "Sensitivity" (Medical/Diagnostic Testing)
Comes from the question: **"How sensitive is this test to detecting the condition?"**

In medical testing: Of all patients with the disease, how many did the test detect?
- A pregnancy test with 95% sensitivity catches 95% of actual pregnancies

[↑ Back to TOC](#table-of-contents)
## The Intuition Behind "Sensitivity"

A **highly sensitive** test/model is very good at detecting positive cases - it's sensitive to their presence. It rarely misses them (few False Negatives).

Think of it like a sensitive smoke detector:
- **High sensitivity**: Goes off easily, catches all fires (few misses) but many false alarms
- **Low sensitivity**: Misses some fires (more False Negatives)

[↑ Back to TOC](#table-of-contents)
## For Employee Retention
Your model has **66.3% sensitivity/recall** for detecting departures:

```python
Sensitivity = 264 / (264 + 134) = 0.663
```

This means your model is **moderately sensitive** to employees leaving - it detects about 2 out of 3 departures but misses 1 out of 3.

[↑ Back to TOC](#table-of-contents)
## The Companion Metric: Specificity

While we're here, "sensitivity" has a common companion in medical contexts:

**Specificity** = True Negatives / (True Negatives + False Positives)
- In ML, this is often just called "True Negative Rate"
- Measures: Of all negative cases (stayed), how many did you correctly identify?

Your specificity:
```python
Specificity = 1646 / (1646 + 355) = 82.3%
```
[↑ Back to TOC](#table-of-contents)
## Quick Reference
| ML Term | Medical Term | What It Measures |
|---------|--------------|------------------|
| Recall | Sensitivity | How many positives you catch |
| True Negative Rate | Specificity | How many negatives you correctly identify |
| Precision | Positive Predictive Value | When you predict positive, how often you're right |

**Bottom line:** When you see "sensitivity" in papers or documentation, just think "recall" - they're interchangeable!
  
[↑ Back to TOC](#table-of-contents)
