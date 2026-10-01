# Problem with single pass Jev

Plain `jev()` asks one question with $20-wide buckets, picks the bucket with the highest probability (via `answer.choice`), and returns its midpoint. That's an `argmax` read — "give me your single best guess." There are two ways an `argmax` (or a naive mean) can mislead when the distribution is spread out and noisy:


## The Argmax Problem

`Argmax` only looks at which single bucket has the highest number — it throws away everything about how the rest of the probability mass is arranged.

Jev doesn't return one number, it returns a probability for every one of the 50 buckets (in the single-pass case). If the model is confident, that probability is concentrated: one or two buckets hold most of the mass, and argmax picks correctly. But if the model is uncertain, or the true price genuinely sits in a gray zone between buckets, the probability gets spread thin across many buckets instead of piled into one.

#### A concrete example

Say the buckets around the true price look like this:

```
$180-$200: 0.13
$200-$220: 0.15   <- argmax picks this one
$220-$240: 0.12
$240-$260: 0.11
$260-$280: 0.09
... (rest of the mass thinly spread across another 20+ buckets)
```

`Argmax` just returns $200-$220, because 0.15 is the single largest number. But look at what it ignored: 0.85 of the total probability lives somewhere else. The bucket it picked only barely edged out its neighbors ($180-$200 at 0.13, $220-$240 at 0.12), and there's nothing in the data suggesting $200-$220 is meaningfully more likely than those neighbors. It just happened to have the local maximum.

#### Why this happens in practice

A model that's genuinely torn between, say, $190 and $230 doesn't return one big number; it returns a soft bump around the middle of its uncertainty, split across several adjacent $20 buckets. If that bump straddles a bucket boundary, whichever half of the bump lands slightly more inside one bucket than the other "wins" the argmax, even by a razor-thin margin (0.15 vs 0.13), while the model's actual belief was "somewhere around $190-$230," not "definitely $200-$220."

#### Why this specifically hurts on a spread-out distribution

If the model were confident (say one bucket at 0.7 and everything else near zero), `argmax` and the true center of mass would basically coincide anyway, so it wouldn't matter which method you used. The problem only bites when confidence is low: with 50 buckets, an uncertain guess might have its winning bucket at just 10-15% of the mass, comfortably beaten in total by all the other buckets combined. Argmax still confidently reports that one bucket as "the answer," discarding the fact that the model was actually quite unsure and leaning across several neighboring buckets.

#### How the median avoids this

The median asks a different question entirely: "at what dollar value does cumulative probability cross 50%?" That's a property of the whole shape of the distribution, not just its single tallest bar. In the example above, the median would land somewhere in the middle of that whole thinly-spread cluster (close to where the real mass is concentrated), rather than snapping to whichever bucket happened to eke out a narrow local win. It's using all 50 numbers instead of throwing away 49 of them.


## The Mean Problem

The mean gives weight to every bucket in proportion to its probability, including a long tail of buckets that individually look negligible but collectively aren't.

#### Where the "rounding floor" comes from

Jev returns a probability for all 50 buckets, and those probabilities have to sum to 1. But probability values from a model like this aren't reported with infinite precision, they get rounded to a floor of 0.01. That means even a bucket the model considers essentially impossible doesn't get reported as 0.0000003, it gets rounded up to the smallest representable value, 0.01.

So imagine the true price is around $200, and the model is reasonably confident about that. But out of the 50 buckets, dozens of far-away ones (say buckets up near $700, $800, $900) each get floored to 0.01, purely as a rounding artifact, not because the model genuinely thinks there's meaningful probability there.

#### Why a mean is vulnerable to this

A probability-weighted mean is:
```
mean_price = Σ (bucket_midpoint × bucket_probability)
```

Every bucket contributes to that sum in proportion to its distance from zero, multiplied by its probability. A single far-away bucket at 0.01 probability seems trivial on its own. But if there are, say, 20 buckets way out near $700-$900 that all get floored to 0.01, that's 20 × 0.01 = 0.20, a full 20% of total probability mass, sitting at prices roughly $500+ away from where the real answer is. Meanwhile the "genuine" cluster of probability near $200 might only account for the remaining 80%.

#### A rough numeric illustration

Say the real signal is:
- 80% of mass concentrated around $200 (weighted average ≈ $200)
- 20% of mass is actually rounding noise, spread across far buckets averaging around $800

```
mean ≈ 0.8 × 200 + 0.2 × 800 = 160 + 160 = $320
```

The mean comes out at $320, dragged well above the $200 that the model actually believes, purely because of the floored tail. None of that 20% represents real conviction, it's an artifact of how probabilities get reported, but the mean can't tell the difference between "the model genuinely thinks there's a 20% chance it costs $800" and "the model rounded 50 tiny near-zero numbers up to 0.01 each."

#### This is the opposite failure mode from argmax

- `Argmax` ignores everything except the single tallest bucket, so it's blind to the shape of the rest of the distribution and can be fooled by a narrow, barely-there local maximum.
- `Mean` does the opposite: it accounts for every bucket, weighted by probability, so a large number of small, spurious probabilities (the floored tail) can collectively out-vote the real signal, dragging the estimate toward the tail.

## Why the median avoids this

The median only cares about where the 50% cumulative-probability crossing point falls, not how far the tail stretches or how much total mass sits in it. Even if there's a long tail of floored buckets way out past $700, as long as they're a minority of total probability (in the example, 20%, well under 50%), the median calculation never gets anywhere near them: the running cumulative sum reaches 50% while still inside the real $200-ish cluster, and the tail is irrelevant to where that crossing point lands. A mean has to "average in" every stray data point; a median just needs to find the middle, and a symmetric or one-sided tail barely moves where the middle is.

