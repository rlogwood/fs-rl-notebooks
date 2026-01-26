# Understanding R2 Scores

R² is calculated as:

**R² = 1 − (SS_res / SS_tot)**

Where:
- **SS_res** (residual sum of squares) = Σ(yᵢ − ŷᵢ)² — the squared errors between actual values and your model's predictions
- **SS_tot** (total sum of squares) = Σ(yᵢ − ȳ)² — the squared differences between actual values and the mean of y

## Intuition

The formula is essentially asking: "How much better is my model than just predicting the mean every time?"

- **SS_tot** represents total variance in your data
- **SS_res** represents the error your model still makes
- The ratio SS_res/SS_tot is the proportion of variance your model *failed* to explain
- Subtracting from 1 flips it to the proportion your model *did* explain

## Quick Examples

| R² | Meaning |
|-----|---------|
| 1.0 | Perfect predictions (SS_res = 0) |
| 0.0 | Model is no better than predicting the mean |
| Negative | Model is *worse* than just predicting the mean |

That last point surprises people—R² can go negative if your model is badly misspecified, though this is rare with properly trained models.


### Definitions
- **Variance** in this context refers to how spread out your actual bike rental values are from 
            their average. SS_tot captures this—if rentals range from 10 to 10,000 per day, 
            there's high variance to explain. If they're always around 500, there's little variance. R² tells you what fraction of that spread your model accounts for.

- **ŷ (y-hat)** is the predicted value from your model for each observation. 
            So if your model predicts 847 rentals for a given day, that's ŷ for that row. 
            The actual rental count that day is y. The difference (y − ŷ) is your residual/error for 
            that observation.

### Explantion
SS_tot measures the variance in your actual y values (bike rentals). This spread exists before you build any model. Some days have 200 rentals, others have 5,000—that variation is just a fact about your data.
R² then asks: "Of that pre-existing spread in y, how much does my model successfully capture?"
Think of it this way:

Before modeling: Your rental counts vary a lot around their mean. That's SS_tot.
After modeling: Your predictions (ŷ) try to track those actual values. The leftover error—where predictions miss—is SS_res.
R² interpretation: If SS_res is small relative to SS_tot, your model captured most of the original spread. If SS_res is nearly as large as SS_tot, your model didn't explain much—the predictions aren't tracking the real variation.

So R² = 0.622 means: "62% of the variation that already existed in daily bike rentals is accounted for by your model's predictions. The other 38% remains unexplained."
The model doesn't introduce spread—it attempts to mirror the spread that's already there. R² measures how well it succeeds.

### Additional Resources
- [search define R2](https://www.bing.com/search?q=define+r2)
- [StatQuest: R-squared](https://www.youtube.com/watch?v=ZkjP5RJLQF4)
