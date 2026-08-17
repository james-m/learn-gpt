# Chapter 9 — The training loop

## Where we are

Everything assembles here. One thousand times: pick a name, run the forward pass to a
loss (ch 5–8), run `backward()` for gradients (ch 2–3), then *use* those gradients to
nudge all 4,192 parameters. The nudging rule is the Adam optimizer — the industry
default, implemented in seven lines.

**Covers:** `microgpt.py:146-184`

## The code

```python
# Let there be Adam, the blessed optimizer and its buffers      # :146
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # ... forward pass to `loss` (chapter 8) ...                # :160-169

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
```

## Walkthrough

**The core idea first — gradient descent.** After `backward()`, every parameter knows
`p.grad`: "if I grew slightly, the loss would change by this much per unit." So move
each parameter a small step *against* its gradient:

```python
p.data -= learning_rate * p.grad     # this alone would work! (plain SGD)
```

That's it — that's how neural networks learn. Everything else in the block is about
making these steps *better sized*. The **learning rate** (0.01) is the compromise
between crawling (too small) and overshooting the valley you're descending into (too
big); it's the hyperparameter practitioners fuss over most.

**Stochastic**: each step's gradient comes from ONE name, not the whole dataset — a
noisy estimate of the "true" downhill direction, a different name each step
(`step % len(docs)` just cycles; with 32K names and 1K steps, no name repeats). Noisy
-but-cheap beats exact-but-slow, spectacularly — this is the "S" in SGD and the reason
the lab's loss curve jitters while trending down. Real training averages a **batch**
of examples per step to tame the noise; microgpt uses batch size 1, the purest form.

**Adam = SGD with two exponential moving averages per parameter.** For each `p`:

- `m` (**first moment**) — a running average of recent gradients (`beta1=0.85` ≈
  "remember roughly the last 6–7 steps"). Using `m` instead of the raw gradient
  smooths the single-name noise, and keeps moving through flat spots — hence
  "momentum," a ball rolling with inertia rather than a hiker re-deciding each step.
- `v` (**second moment**) — a running average of gradients *squared*: a per-parameter
  measure of "how big is this gradient, typically?" Dividing the step by `sqrt(v)`
  gives every parameter its own effective step size: rarely-touched parameters (say,
  weights only `z` and `q` excite) get big confident nudges when they finally receive
  signal; constantly-hammered parameters get gentler ones. This *adaptivity* is why
  Adam gets moving so much faster than SGD here — the lab races them head-to-head.
- `m_hat`, `v_hat` — **bias correction**. The buffers start at 0, so early on they're
  dragged toward zero (after one step, `m` is only 0.15·grad). Dividing by
  `1 - beta**(step+1)` — which starts at 0.15 and rises to 1 — exactly un-drags them.
  A detail, but a famous one: it's the difference between Adam and its buggy
  ancestors.
- `eps_adam` avoids 0/0 when a gradient has simply never arrived.

Then the two bookkeeping lines you already understand from ch 3:
**`p.grad = 0`** (fresh accumulation next step — forget this and gradients from every
step pile up; a rite-of-passage bug) — and note the update touches `p.data` directly,
*outside* autograd. Optimization isn't part of the differentiated computation; it's
surgery on the leaves between graphs. Each step builds a fresh graph, and the old one
becomes garbage for Python to collect.

**Learning-rate decay** (`:175`): `lr_t` slides linearly 0.01 → 0 across training.
Big steps early for coarse progress; small steps late for fine-tuning without
sloshing. Schedules (linear, cosine, warmup+decay) are standard practice — this is
the simplest respectable one.

**What "knowledge" looks like now:** after 1000 steps, nothing about the *code*
changed — only the 4,192 floats. `wte['e']` drifted from noise to a vector that means
something; attention weights learned where to look. Training a frontier model is this
identical loop with more data, more parameters, and batches — run for months on
thousands of GPUs.

## Terminology

- **Gradient descent** — walk downhill on the loss surface in parameter space.
  [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent) ·
  [3Blue1Brown: "Gradient descent, how machines learn"](https://www.youtube.com/watch?v=IHZwWFHWa-w)
  — the canonical visual intuition.
- **Stochastic gradient descent (SGD)** — gradient from a random example (or small
  batch) instead of all data.
  [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- **Optimizer** — the update rule mapping gradients to parameter changes (SGD, Adam,
  and a whole zoo). [Overview: Ruder, "An overview of gradient descent optimization
  algorithms"](https://www.ruder.io/optimizing-gradient-descent/) — the classic
  survey, very readable.
- **Adam** — adaptive moments: momentum + per-parameter step sizes + bias correction.
  [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) ·
  [Paper](https://arxiv.org/abs/1412.6980) (§2's pseudocode = these seven lines).
- **Momentum / moving average** — smoothed memory of recent gradients.
  [Wikipedia](https://en.wikipedia.org/wiki/Momentum_(machine_learning)) ·
  [distill.pub: "Why Momentum Really Works"](https://distill.pub/2017/momentum/) —
  interactive and gorgeous (goes deeper than needed; skim).
- **Learning rate & schedule** — step size and its trajectory over training.
  [Wikipedia](https://en.wikipedia.org/wiki/Learning_rate)
- **Batch / epoch** — examples per step / passes over the dataset. microgpt: batch 1,
  and (1000 steps, 32K names) well under one epoch.
- **Training step / iteration** — one full forward-backward-update cycle.

## Lab

```bash
python3 labs/ch09_training.py          # Adam
python3 labs/ch09_training.py --sgd    # same everything, dumber update rule
```

240 steps, ASCII loss curve bucketed by 20s. Run both and put the curves side by
side: same model, same data order, same gradients — the only difference is those seven
Adam lines, and the gap is dramatic.

## Exercises

1. **Learning-rate safari:** run the lab (fewer steps is fine) with
   `learning_rate=0.0001`, `0.01`, and `1.0`. Describe each curve in one line. Which
   failure is scarier in practice, and why? (Consider: does the 1.0 curve *look*
   obviously broken?)
2. **Kill the decay:** replace `lr_t` with constant `learning_rate`. Compare final-40
   -step average loss to stock over ~400 steps. Is decay's benefit visible this early,
   and what does that tell you about when schedules matter?
3. **Forget to zero:** comment out `p.grad = 0` in a copy of `common.py`'s `train`
   and watch the loss for ~50 steps. Explain the exact mechanism of the explosion
   using ch 3's accumulation semantics.
4. **Read Adam like a mechanic:** for a parameter whose gradient is *always exactly
   0.1*, what update size does Adam settle into (ignore bias correction and eps)?
   Notice anything surprising about how it relates to the gradient's magnitude?
   (Answer: `lr_t` — Adam normalizes magnitude away; only the *sign consistency*
   matters.)

Solutions: `solutions/ch09_solutions.py`

---
Next: [Chapter 10 — Inference](ch10-inference.md)
