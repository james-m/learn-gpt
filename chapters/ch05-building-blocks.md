# Chapter 5 — Building blocks: `linear`, `softmax`, `rmsnorm`

## Where we are

Three tiny functions that the model (`gpt()`, ch 6–7) calls over and over. Learn these
and the transformer stops being exotic: it's these three plus `relu`, arranged in a
particular order.

**Covers:** `microgpt.py:92-106`

## The code

```python
# Define the model architecture: a function mapping tokens and parameters to logits over what comes next   # :92
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

## Walkthrough

### `linear` — the workhorse

Input: vector `x` (length `nin`). For each row `wo` of the matrix, output one
**weighted sum**: `wo[0]*x[0] + wo[1]*x[1] + ...`. That's a *dot product* — and if you
squint, each output is one classic textbook "neuron" (weights × inputs, summed), just
without a bias term or activation. Stack `nout` of them and you've mapped a vector of
size `nin` to size `nout`; that's all a **matrix multiplication** is.

Two readings worth holding simultaneously:
- *Per output:* each row of `w` is one learned "detector" scoring the whole input.
- *Per layer:* `linear` is every neuron of a layer computed at once. When ch 4's table
  says `mlp_fc1` is 16→64, it means: 64 detectors, each reading 16 numbers.

This is >99% of all arithmetic in any GPT. When you hear that LLMs run on GPUs doing
"matmuls" — this loop is what they're accelerating.

### `softmax` — scores → probabilities

Takes any real-valued vector ("**logits**" — raw, unnormalized scores) and returns
positive numbers summing to exactly 1. Recipe: exponentiate each (making everything
positive, and *amplifying gaps*), then divide by the total. Properties that matter:

- Order-preserving: the biggest logit gets the biggest probability.
- It's a "soft" version of argmax — instead of crowning one winner it distributes
  belief, which keeps everything differentiable so gradients can flow through it.
- Shift-invariant: adding a constant to *all* logits changes nothing (the `exp`s all
  scale by the same factor, which cancels in the division). That licenses the
  `max_val` subtraction on `:98` — mathematically a no-op, practically essential
  because `math.exp(1000)` overflows a float. Classic numerical-stability trick;
  every serious framework does it.

Softmax appears twice in microgpt with two different hats on: inside attention
(ch 6, "how much should I look at each previous position?") and at the output
(ch 8/10, "probability of each next token"). Same function both times. It's also
where **temperature** will hook in (ch 10): divide the logits before softmax to
sharpen or flatten the result.

### `rmsnorm` — volume control

Computes the vector's RMS (root-mean-square — "average loudness" of its entries) and
divides every entry by it, so the output always has RMS ≈ 1. Two facts:

- The *direction* of the vector — the ratios between entries, where the information
  lives — is untouched. Only the overall scale is standardized. (The lab shows a
  quiet and a loud vector normalizing to the *identical* result.)
- The `1e-5` is a guard so an all-zeros vector doesn't divide by zero. Note
  `** -0.5` — this multiplies by the *reciprocal* square root, i.e. divides by RMS.

Why bother? Deep nets are products of many layers; without periodic re-standardizing,
activations (and therefore gradients — remember, `*`'s local grads are the other
operand) can compound exponentially louder or quieter, making training unstable.
Normalization keeps every layer receiving inputs in a predictable range. GPT-2 used
LayerNorm (which also subtracts the mean and has learned scale/shift parameters);
RMSNorm is the modern simplification (used by Llama and friends) — and here it's the
zero-parameter version.

The comment on `:93` is your Rosetta stone to real GPTs: microgpt = GPT-2 minus
biases, with LayerNorm→RMSNorm and GeLU→ReLU. Cosmetic differences; same machine.

## Terminology

- **Linear layer / fully-connected layer / matrix multiply** — same thing, three
  communities. [Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication) ·
  [3Blue1Brown: "But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk)
  — frames these weighted sums as neurons and layers.
- **Dot product** — sum of elementwise products; also a *similarity* measure between
  vectors (big when they point the same way) — that reading becomes load-bearing in
  attention, ch 6. [Wikipedia](https://en.wikipedia.org/wiki/Dot_product)
- **Logits** — raw pre-softmax scores. [Wikipedia](https://en.wikipedia.org/wiki/Logit)
- **Softmax** — logits → probability distribution.
  [Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)
- **Numerical stability** — writing math to avoid float overflow/underflow/precision
  loss; the max-subtraction is the canonical example.
  [Wikipedia](https://en.wikipedia.org/wiki/Numerical_stability)
- **RMSNorm / LayerNorm** — activation-scale standardizers.
  [RMSNorm paper](https://arxiv.org/abs/1910.07467) ·
  [Wikipedia (normalization in DL)](https://en.wikipedia.org/wiki/Normalization_(machine_learning))
- **Bias (the neuron kind)** — the `+ b` term classic neurons add and microgpt omits;
  normalization layers make it largely redundant in transformers.

## Lab

```bash
python3 labs/ch05_building_blocks.py
python3 labs/ch05_building_blocks.py 4 4 0    # explore: your own softmax logits
```

Hand-crafted `linear` rows (an averager, a differencer), softmax with bars, proof of
shift-invariance, the actual `OverflowError` the trick prevents, a temperature
preview, and the quiet/loud rmsnorm demo.

## Exercises

1. **Softmax by hand:** compute softmax([1, 1, 3]) with a calculator (no code), to
   2 decimal places. Then check with the lab's `softmax` (wrap floats in `Value`).
2. **Why not just normalize?** A simpler "probabilities" recipe: divide each logit by
   the sum of logits. Give two concrete inputs where this breaks badly and softmax
   doesn't. (Hint: think signs, and think all-equal-but-negative.)
3. **rmsnorm grads flow:** Unlike a naive `x / constant`, rmsnorm's scale *depends on
   x*, so gradients flow through the scale too. Build `y = rmsnorm([a, b])`, backward
   from `y[0]`, and check `a.grad` numerically. Convince yourself autograd handled the
   product rule for you — nothing special was needed.

Solutions: `solutions/ch05_solutions.py`

---
Next: [Chapter 6 — Attention](ch06-attention.md)
