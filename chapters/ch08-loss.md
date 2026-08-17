# Chapter 8 — The loss

## Where we are

The model produces beliefs; the loss grades them. This handful of lines inside the
training loop defines what "good" means for the entire system — everything else
(autograd, Adam, the architecture) exists to push this one number down. It's also
where your old knowledge of loss functions reconnects: this is cross-entropy, the
standard loss for anything that predicts categories.

**Covers:** `microgpt.py:160-169`

## The code

```python
    # Forward the token sequence through the model, building up the computation graph all the way to the loss   # :160
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.
```

## Walkthrough

**The setup** (`:163-164`): recall ch 1's training pairs. For `'emma'` the model is
asked five questions — after BOS predict `e`, after `e` predict `m`, after `m` predict
`m`, after `m` predict `a`, after `a` predict BOS (stop). Each `gpt()` call answers
one question with 27 logits; softmax turns them into probabilities.

**The grade** (`:167`) is almost insultingly simple:

> Look up **the probability the model assigned to the correct answer**. Take
> `-log` of it. Done.

The 26 wrong-answer probabilities are never touched — they're punished implicitly,
because probabilities sum to 1: every point of belief wasted on wrong answers is
belief the right answer didn't get.

**Why -log, and not something milder like `1 - p`?** Feel the shape of it
(lab prints this table):

| p(correct) | -log p |
|-----------:|-------:|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 1/27 (guessing) | 3.296 |
| 0.01 | 4.605 |
| 0.001 | 6.908 |

- Perfect confidence in the truth → loss 0. There's no reward for lucky dice —
  only for *assigning probability* to what actually came next.
- The penalty **explodes** as p → 0: being confidently wrong is catastrophically
  expensive, so the model learns to hedge realistically rather than to gamble. `1-p`
  would cap the pain at 1 and barely care about the difference between p=0.01 and
  p=0.001; -log cares a lot.
- Deeper reading: `-log p` is the *surprise* of an outcome (rare = surprising = big).
  Minimizing average surprise ⇔ maximizing the probability the model assigns to the
  data ⇔ "maximum likelihood estimation," the classical statistics framing. Three
  vocabularies, one quantity.

**The average** (`:169`) makes documents of different lengths comparable: loss-per-
prediction, not per-name. Note `(1/n) * sum(losses)` happens in `Value`-land — the
mean is *part of the computation graph*, so each position's gradient automatically
arrives scaled by `1/n`.

**Calibrate your eye with the uniform baseline.** A model that knows nothing should
spread belief evenly: p = 1/27 everywhere, loss = `-log(1/27) ≈ 3.296`. Run
`microgpt.py` and look at step 1: **3.3660**. That's your proof the initialization
story from ch 4 worked (tiny weights → near-uniform output), and your ruler for
progress: by step 1000 the loss wanders around ~2.0–2.5, i.e. the model gives the true
next character e^-2.2 ≈ 11% on average — triple random guessing. When you train any
model, ever: compute the dumb baseline first, or the loss number means nothing.

**One vocabulary note:** "cross-entropy loss" in PyTorch (`F.cross_entropy`) fuses
softmax + pick-target + -log into one call. Lines `:166-167` are that call, unrolled.
What frameworks call "NLL" (negative log-likelihood) is the same thing modulo where
the softmax lives.

## Terminology

- **Cross-entropy loss** — `-log p(correct class)`, averaged; *the* loss for
  classification and language modeling.
  [Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy) ·
  [StatQuest: Neural Networks Part 6 — Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU)
  — friendly and concrete.
- **Negative log-likelihood / maximum likelihood** — same objective in statistics
  clothing: choose parameters making the observed data most probable.
  [Wikipedia](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
- **Surprise / information content** — `-log p` as "how surprised you are"; the
  bridge to information theory.
  [Wikipedia](https://en.wikipedia.org/wiki/Information_content)
- **Perplexity** — `e^loss`: "the model is as confused as if choosing evenly among
  *this many* options." Uniform baseline: e^3.296 = 27 (exactly the vocab). The
  standard headline metric for language models.
  [Wikipedia](https://en.wikipedia.org/wiki/Perplexity)
- **Objective / loss function** — the number a learning system minimizes; the formal
  definition of the task. [Wikipedia](https://en.wikipedia.org/wiki/Loss_function)
- **Classification** — predicting one of a fixed set of categories. Next-token
  prediction is 27-way classification, done at every position.

## Lab

```bash
python3 labs/ch08_loss.py                    # trains ~150 steps midway; under a minute
python3 labs/ch08_loss.py 0.99 0.001         # explore: your own p's for the -log table
```

The -log(p) feel-table, then the punchline: an untrained model's measured average loss
lands within a hair of the theoretical 3.296 baseline, and 150 training steps pull it
decisively below. Also converts the trained loss back to "average p on the truth" so
the abstract number becomes a probability you can feel.

## Exercises

1. **Perplexity by hand:** microgpt finishes around loss ≈ 2.2. What perplexity is
   that? Sanity-check: is it sensible that a good letter-predictor for English names
   is "as confused as ~9 options" when 27 exist?
2. **Best possible loss ≠ 0:** even a perfect model of names can't reach loss 0 on
   this data. Find concrete pairs of names in `data/names.txt` that prove irreducible
   uncertainty exists (hint: shared prefixes, different continuations — `ann`/`anna`,
   or the first letter after BOS).
3. **Loss of the truthful die:** a model predicting the next letter after `'j'`
   assigns p=0.6 to `a`, 0.3 to `o`, 0.1 to everything else combined. Data comes in:
   `a`, `o`, `a`. Average loss? Now a "braver" model says p=1.0 on `a`: average loss
   on the same three outcomes? (Infinity counts.) Moral?

Solutions: `solutions/ch08_solutions.py`

---
Next: [Chapter 9 — The training loop](ch09-training-loop.md)
