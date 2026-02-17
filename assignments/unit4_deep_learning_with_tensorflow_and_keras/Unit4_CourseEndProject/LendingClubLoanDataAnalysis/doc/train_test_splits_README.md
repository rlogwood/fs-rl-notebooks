
# Train/Test Split

We split the data the first time 70/30, putting 30% into X_test and y_test,
then we split the 70% again into 80% X_train and y_train and 20% 
into X_val and y_val. 

The training data is used for training, 
and the 20% X_val and y_val are used to evaluate the accuracy of the model. Then we use the first split X_test and y_test to see how well the model predicts outcomes. Do I understand this correctly?

```python
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, stratify=y_temp
)

```


Here's the breakdown with the loan data:
```
Full Dataset: 9,578 samples
    ↓
Split 1 (70/30):
    ├─ 70% → 6,704 samples (train+val combined)
    └─ 30% → 2,874 samples (X_test, y_test) ✓ HELD OUT
    
    6,704 samples
        ↓
    Split 2 (80/20):
        ├─ 80% → 5,363 samples (X_train, y_train) ✓ TRAINING
        └─ 20% → 1,341 samples (X_val, y_val)     ✓ VALIDATION
```


**Purpose of each set:**

1. **X_train, y_train (5,363):** Train the model - the model learns patterns from this data

2. **X_val, y_val (1,341):** Monitor performance during training
   - Early stopping watches validation metrics
   - Helps prevent overfitting
   - Never used for training, only for monitoring

3. **X_test, y_test (2,874):** Final evaluation after all training is complete
   - Model has NEVER seen this data
   - Gives unbiased estimate of real-world performance
   - Used only once at the very end

