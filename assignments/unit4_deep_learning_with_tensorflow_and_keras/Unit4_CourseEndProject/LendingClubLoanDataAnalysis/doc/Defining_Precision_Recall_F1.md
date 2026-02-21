# Defining Precision, Recall, and F1 Score



**Precision** answers: *"When my model said someone would leave, how often was it right?"*

It's about the **trustworthiness of positive predictions**. If precision is 80%, then 8 out of 10 people flagged as "will leave" actually left. The other 2 were false alarms.

**Recall** answers: *"Of all the people who actually left, how many did my model catch?"*

It's about **completeness of detection**. If recall is 70%, the model caught 70 out of 100 actual departures — and missed 30.

So both are measures of accuracy, just from different angles. Precision looks at it from the prediction side ("of my positive calls, how many were correct?"), recall looks at it from the reality side ("of the actual positives, how many did I find?").

**F1 Score** is the harmonic mean of precision and recall — a single number that balances both. It punishes you hard if either one is low. So you can't game it by just predicting "everyone leaves" (high recall, terrible precision) or being extremely selective (high precision, terrible recall).

**Threshold** is the probability cutoff your model uses to decide "yes" vs "no." Your model outputs a probability (say 0.63), and the threshold determines where you draw the line. Default is usually 0.5, but that's arbitrary.

**Optimal threshold** is where you deliberately shift that cutoff based on what matters most. In your retention project, you've been lowering it to ~0.3–0.4 because missing a real departure ($50K+ cost) is far worse than a false alarm (~$500 intervention). Lowering the threshold means you're telling the model "be more aggressive about flagging people" — which pushes **recall up** but **precision down**.

Here's a quick analogy: imagine a smoke detector.

- **High recall, low precision** = sensitive detector that catches every fire but also goes off when you make toast. You'd want this in a hospital.
- **High precision, low recall** = detector that only alarms for real fires but sometimes sleeps through small ones. Dangerous.

For retention prediction, you want the sensitive detector — which is exactly why you optimize for recall with a lower threshold.

Does that framing click better?


**Harmonic mean** is a type of average that heavily penalizes low values. Unlike a regular (arithmetic) mean where 0.9 and 0.1 averages to 0.5, the harmonic mean of 0.9 and 0.1 gives you ~0.18. It essentially says "you're only as strong as your weakest link." This is why F1 can't be gamed by sacrificing one metric for the other.

Here are the formulas:

**Precision:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

True positives divided by everything the model *called* positive. "Of my positive predictions, how many were actually positive?"

**Recall:**

$$\text{Recall} = \frac{TP}{TP + FN}$$

True positives divided by everything that *was actually* positive. "Of the real positives, how many did I find?"

**F1 Score:**

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

That's the harmonic mean formula applied to precision and recall. The 2× is there because the general harmonic mean of two values is $\frac{2ab}{a+b}$.

Quick concrete example — say your model flags 100 employees as likely to leave:

- 60 actually left → TP = 60
- 40 didn't → FP = 40
- 20 who left were missed → FN = 20

Then: Precision = 60/100 = 0.60, Recall = 60/80 = 0.75, F1 = 2 × (0.60 × 0.75)/(0.60 + 0.75) = **0.667**

Notice how F1 (0.667) sits closer to the lower value (0.60) than the higher one — that's the harmonic mean doing its job.