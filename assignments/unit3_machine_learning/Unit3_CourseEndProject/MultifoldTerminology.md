# Stratified K-Fold Cross-Validation

Stratified means each fold preserves the same class proportion as the original dataset.

## Example with Your Turnover Data

Say your training data has 24% employees who left (class 1) and 76% who stayed (class 0):

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|--------|--------|--------|--------|--------|--------|
| StratifiedKFold | 24% / 76% | 24% / 76% | 24% / 76% | 24% / 76% | 24% / 76% |
| Regular KFold | 30% / 70% | 18% / 82% | 25% / 75% | 22% / 78% | 25% / 75% |

## Why It Matters

With regular KFold on imbalanced data:
- Some folds might have very few positive cases by chance
- Model performance varies wildly between folds
- Cross-validation scores become unreliable

With StratifiedKFold:
- Every fold is a representative sample
- Consistent training/validation conditions
- More reliable performance estimates

## The Term "Stratified"

Comes from stratified sampling in statistics - dividing a population into subgroups (strata) and sampling proportionally from each. Here, the "strata" are your class labels (left vs stayed).

For imbalanced classification like employee turnover, always use StratifiedKFold over regular KFold.

## Naming Conventions for N-Folds Constants

| Convention | Example | Pros | Cons |
|------------|---------|------|------|
| `N_FOLDS` | `N_FOLDS = 5` | Clear, concise; "N" is mathematical convention for count | Could be confused with other "N" variables |
| `K_FOLDS` | `K_FOLDS = 5` | Matches "k-fold" terminology in ML literature | Less intuitive if unfamiliar with convention |
| `NUM_FOLDS` | `NUM_FOLDS = 5` | Explicit, self-documenting | Slightly longer |
| `CV_FOLDS` | `CV_FOLDS = 5` | Ties it explicitly to cross-validation | Redundant if only used for CV |
| `N_SPLITS` | `N_SPLITS = 5` | Matches sklearn's parameter name exactly | Less domain-specific |

**Recommendation:** `N_FOLDS` or `K_FOLDS` are both widely accepted. `K_FOLDS` aligns with the standard "k-fold cross-validation" terminology in ML literature.
