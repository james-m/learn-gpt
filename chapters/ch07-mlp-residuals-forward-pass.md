# Chapter 7 — MLP, residuals, and the full forward pass

## Where we are

The second half of `gpt()`: after attention *gathers* information from the past, a
small two-layer neural network (the MLP) *processes* it. Wrap both in residual
connections, project to vocabulary scores, and the forward pass is complete — every
line of the model, accounted for.

**Covers:** `microgpt.py:135-144` (plus a second look at `:112`, `:116`, `:134`)

## The code

```python
        # 2) MLP block                                         # :135
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])                  # :143
    return logits
```

## Walkthrough

**The MLP** (`:138-140`) is the most classical object in the file — a plain two-layer
neural network, the thing "multilayer perceptron" has meant since the 1980s: expand 16
→ 64 (`fc1`), apply a nonlinearity (`relu`), contract 64 → 16 (`fc2`). Where attention
moved information *between positions*, the MLP computes *within* one position — it
can't see anyone else. A useful division of labor to memorize:

> **attention = communicate, MLP = compute.**

**Why the nonlinearity is non-negotiable:** chain `linear` after `linear` and algebra
collapses them into one equivalent `linear` — a hundred stacked linear layers have
exactly the power of one. `relu` breaks the collapse: by zeroing negatives it lets
the network implement *conditional* logic ("if this combination of features is
present, then..."). Each of the 64 hidden units is a little feature detector that
fires or stays silent; `fc2` recombines the firings. And the 4× expansion? Computing
in a roomier space before compressing back — standard in every transformer (GPT-2
likewise used 4×; it's a convention, not a law).

**Residual connections** (`:134`, `:141`) — the unassuming `x + block(x)` pattern that
made deep networks trainable, twice per layer here. Two readings:

- *Forward:* each block contributes an **adjustment** to a running signal rather than
  wholesale replacing it. Karpathy's image: a "residual highway" — the vector cruises
  along; attention and MLP are on-ramps merging refinements in.
- *Backward (the important one):* recall from ch 3 that `+` has local grads `(1, 1)` —
  it passes gradient through **undiminished**. So the loss's gradient flows straight
  down the highway to the earliest layers no matter how deep the stack; blocks only
  *add* signal, never gate it. Without residuals, deep-network gradients must survive
  every intermediate transformation and tend to die (or blow up) on the way — the
  "vanishing gradient" problem that capped network depth for decades. With them,
  100-layer GPTs train. (This also explains the cryptic comment on `:112`: rmsnorm
  before the highway starts is *not* redundant, because gradients arriving via the
  highway see it differently than a post-block norm would.)

**Pre-norm** (`:117`, `:137`): each block normalizes its *input* (`rmsnorm` after
saving `x_residual`) — so the highway itself is never normalized, only the on-ramp
computations. This "pre-LN" arrangement trains more stably than the original
transformer's post-block placement, and is what GPT-2 and successors settled on.

**With `n_layer > 1`** the whole attention+MLP pair just repeats: communicate,
compute, communicate, compute... Each round lets information hop further and be
processed more. microgpt uses one layer; GPT-2 had 12–48; frontier models, ~100.
Same loop body.

**The finale** (`:143`): `lm_head` maps the final 16-dim vector to 27 numbers — one
score per vocabulary token. These are the **logits**; softmax away from being
next-token probabilities. Note what the model returns: not a choice, not a letter — a
*scored ballot*. Training (ch 8) grades the ballot; generation (ch 10) draws from it.

**The whole architecture on one napkin:**

```
token id ─→ wte ┐
                ├─(+)─→ rmsnorm ─→ [ attention block ]─(+)─→ [ MLP block ]─(+)─→ lm_head ─→ 27 logits
position ─→ wpe ┘        ↑residual────────────┘   ↑residual──────┘      (×n_layer)
```

## Terminology

- **MLP / feed-forward network (FFN)** — the expand-nonlinear-contract block; "FFN" in
  most transformer papers.
  [Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron) ·
  [3Blue1Brown: "How might LLMs store facts"](https://www.youtube.com/watch?v=9-Jl0dxWQs8)
  — argues the MLP blocks are where facts live; great intuition for "compute."
- **Activation function / nonlinearity** — the function between linear layers that
  prevents collapse; here ReLU, in GPT-2 GeLU (a smooth cousin).
  [Wikipedia](https://en.wikipedia.org/wiki/Activation_function)
- **Residual (skip) connection** — `x + block(x)`; the gradient highway.
  [Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network)
- **Vanishing / exploding gradients** — what residuals + normalization jointly solve.
  [Wikipedia](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- **Pre-norm vs post-norm** — where the normalization sits relative to the residual
  add; modern GPTs are pre-norm.
- **Language-model head (`lm_head`)** — final projection to per-token scores.
- **Hidden layer / hidden units** — the 64-wide intermediate inside the MLP; "hidden"
  because it's neither input nor output, just internal scratch space.

## Lab

```bash
python3 labs/ch07_forward_trace.py
python3 labs/ch07_forward_trace.py z --pos 7    # explore: any char, any position
```

Pushes one token through the entire forward pass with a printout at every stage:
vector lengths (watch 16 → 64 → 16 → 27), RMS values (watch rmsnorm snap them to
~1.0), how many hidden units relu silenced, and the final near-uniform probabilities
of an untrained model.

## Exercises

1. **Prove the collapse:** with `w1 = [[2,0],[0,3]]` and `w2 = [[1,1],[1,-1]]`, find
   the single matrix `w` such that `linear(linear(x, w1), w2) == linear(x, w)` for all
   x. Verify on a couple of inputs. That's why no-relu ⇒ one big linear layer.
2. **Ablate the residuals:** in a copy of `labs/common.py`'s `gpt`, delete both
   `+ x_residual` adds, retrain 200 steps (same seed), and compare loss curves against
   the stock model. Even at depth 1 you'll typically see slower/worse training — and
   reflect: why would the gap become catastrophic at depth 12?
3. **Count the compute:** roughly how many scalar multiplies does one call to `gpt()`
   make in the MLP block vs. the attention block (at position 5, say)? Which dominates
   — and does your answer change late in a long sequence?

Solutions: `solutions/ch07_solutions.py`

---
Next: [Chapter 8 — The loss](ch08-loss.md)
