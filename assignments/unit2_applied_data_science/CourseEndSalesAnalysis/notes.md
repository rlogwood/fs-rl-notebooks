
# Additional Observations
## Daily, Weekly and Monthly sales
- Shows hightest sales at the start the last month of the 4th quarter
- Better sales at the beginning of the 4th quarter 
- Recommendation, monitor these trends to determine the cause. Consider increased marketing during slower periods in the quarter to see how they affect outcomes
- Sales are fairly uniform throughout the week

## Average Sales by time of day
- Shows that sales volume isn't affected by time of day

## Total Sales by State
- There is statistically significant difference in sales between states
- We see the hightest sales in VIC and NSW
- Review the marketing strategies used for these higest performers and try applying them to low performing states to see if sales can be increased
- Looking at the transaction ranges for the various states, it would make sense to obtain data about median incomes. Income differences may best explain the high sales in the best performing states. 


## State-wise Sales by Demographic Group
- Within a given state there seems to be remarkable uniformity among the groups.


## Demograpic group effect on Sales
- In total Men out perform other groups and Seniors come in last. Is there something in the marketing that appeals more to men or is the product just more suited for them. 
- That said, the diffences between groups is not huge.
- Variance analysis shows there is is not a significant difference between groups:
  - F-statistic: 0.2826
  - P-value: 0.838035
  -  No significant difference between groups (p >= 0.05)

## Data normalization reveals
- Unit and Sales are capturing the same information, probably because they differ only by a constant factor, price. 
- So it's sufficient to look at only one of these values to capture the correlation with other features.
