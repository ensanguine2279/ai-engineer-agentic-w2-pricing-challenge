# Results from the 200-items pricing test


| Model / method            | Avg. absolute error | MSE       | R²        | Cost / 1K | Time   |
| ------------------------- | ------------------- | --------- | --------- | --------- | ------ |
| GPT-4.1 Nano              | **$64.31**          | 16,805    | 23.5%     | $0.012    | 899 ms |
| GPT-5.6 Luna              | **$51.18**          | 11,526    | 47.6%     | $0.030    | 1.11 s |
| Jev 1.13                  | $61.71              | 10,917    | 50.3%     | $0.082    | 309 ms |
| **Jev — two-pass median** | **$52.36**          | **6,993** | **68.2%** | $0.097    | 303 ms |


## 1. The interesting result isn't simply "Jev beats GPT"

The first `Jev` implementation gives: $61.71 error / 50.3% R²

versus Luna: $51.18 error / 47.6% R²

So Luna has lower absolute error, while `Jev` has slightly better R².

But after changing how the `Jev` probability distribution is interpreted, the result changes substantially:

`Jev` two-pass median → $52.36 error, 68.2% R².

That's the strongest result in the experiment.

The important insight is therefore:

> `Jev`'s underlying prediction appears considerably more useful than the original decoding method was extracting.

The problems identified was taking the `argmax` or mean of the distribution isn't necessarily appropriate, and the two-resolution/median approach was designed to extract the distribution more effectively.

## 2. `Jev`'s biggest advantage is speed

This is probably the most surprising part of the experiment.


|       | Luna   | Jev two-pass |
| ----- | ------ | ------------ |
| Error | $51.18 | $52.36       |
| R²    | 47.6%  | **68.2%**    |
| Time  | 1.11 s | **303 ms**   |


The experiment suggests that `Jev` can produce a structured prediction roughly 3.7× faster than Luna, while the two-pass method actually has better R².

That's potentially much more important than the raw price error.

For a production system processing millions of products, latency and cost matter enormously.

## 3. But there is an important apples-to-oranges issue

There is a big methodological caveat to be addressed before drawing a strong conclusion.

GPT is asked:

> "Estimate the price of this product. Respond with the price, no explanation."

It then produces a continuous number such as `$250`.

`Jev`, however, is given **explicit $20 price buckets** and returns probabilities over those choices. The code then converts those probabilities back into a price estimate.

So the models aren't solving *exactly* the same statistical problem.

`Jev` has been given a representation specifically suited to the evaluation.

That doesn't invalidate the experiment—in fact, **that's arguably the point of** `Jev`**'s architecture**—but it means that the conclusion would have to be phrased carefully.

## 4. The R² result is particularly interesting

The results:

**GPT-4.1 Nano: 23.5%**

**Luna: 47.6%**

**Jev: 50.3%**

**Jev two-pass median: 68.2%**

There is a very substantial improvement.

R² is measuring how much of the variation in actual prices is captured by the predictions. The jump from 47.6% to 68.2% is especially notable because the average absolute error barely changes from Luna's $51.18 to `Jev`'s $52.36.

That tells me the **distributional approach is producing better-calibrated relative predictions**, even though the average dollar error isn't dramatically better.

## 5. The cost comparison is less favorable to Jev

This is where `Jev` doesn't look as attractive:

GPT-4.1 Nano: $0.012 / 1K
Luna: $0.030 / 1K
Jev: $0.082 / 1K
Jev two-pass: $0.097 / 1K

So the two-pass Jev experiment costs roughly:

8× GPT-4.1 Nano

and about:

3.2× Luna

per 1K according to the reported figures.

That is a significant trade-off.

## The overall interpretation

`Jev` demonstrates a strong advantage when the task is formulated as structured probabilistic decision-making rather than free-form numerical generation.

The most compelling evidence is not that the basic `Jev` model beats GPT on raw error—it doesn't.

It's that changing the extraction of `Jev`'s probability distribution produces a large improvement in R² while retaining extremely low latency.

The result to highlight is therefore:

Jev two-pass median: 68.2% R², $52.36 error, ~303 ms

versus

Luna: 47.6% R², $51.18 error, ~1.11 s.

That is a genuinely interesting result.

The next thing to do next before deciding that `Jev` is better for this task, would be to run a much more rigorous comparison:

- Same 1,000–10,000 completely unseen products.
- GPT models get the same price-bucket formulation as Jev.
- Jev and GPT get exactly the same product information.
- Compare MAE, RMSE, R² and median absolute error.
- Measure latency and cost per 1,000 predictions.
- Break results down by price range, e.g. <$20, $20–100, $100–500, >$500.
- Measure calibration—whether a 70% confidence prediction is actually correct about 70% of the time.

That would turn the current experiment from an interesting demo into a much stronger model evaluation.

## Caveats
- Results are based on a single run of 200 test items; re-running could shift numbers slightly due to inherent model variance (temperature/sampling effects), though seed=42 was set for the LLM calls to improve reproducibility.

- r² can be sensitive to a handful of expensive outlier items given the $0–$1000 price range in this dataset; a stratified look at 
error by price band would clarify whether any model struggles specifically with higher-priced items.

- Cost and latency figures reflect OpenRouter pricing/response times at the time of this run, both of which can change.