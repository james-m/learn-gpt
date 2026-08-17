# Chapter 4 — Parameters & initialization

## Where we are

Here the model's memory is allocated: 4,192 `Value`s, each initialized to a small
random number. Right now they know nothing. Training (ch 9) will nudge each one
thousands of times; whatever the finished model "knows" is stored entirely in these
numbers. This chapter is about what each matrix is *for* and why they start random.

**Covers:** `microgpt.py:74-90`

## The code

```python
# Initialize the parameters, to store the knowledge of the model   # :74
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")
```

## Walkthrough

**The four knobs** (`:75-78`) are the *hyperparameters* — set by the programmer, not
learned:

- `n_embd = 16`: every token is represented internally as a list of 16 floats (an
  *embedding vector*). This is the "width" of every pipe in the model.
- `block_size = 16`: the most context positions the model can ever see (the longest
  name is 15 chars + BOS — a deliberate fit).
- `n_layer = 1`, `n_head = 4`: depth of the transformer stack and number of parallel
  attention heads (both get real meaning in ch 6–7).

**A "matrix" here is honest Python**: a list of `nout` rows, each a list of `nin`
`Value`s. In ch 5 you'll see `linear(x, w)` compute `nout` weighted sums of an
`nin`-long input — so read every shape below as **input size → output size**.

**The cast of matrices** (run `labs/ch04_parameters.py` for this table live):

| name | shape | params | job |
|------|-------|-------:|-----|
| `wte` | 27×16 | 432 | **token embedding table** — row *i* is the learned 16-number "meaning" of token *i*. Not multiplied; just indexed: `wte[token_id]`. |
| `wpe` | 16×16 | 256 | **position embedding table** — row *p* is the learned meaning of "being at position *p*". How the model tells `ana` from `aan`. |
| `layer0.attn_wq/wk/wv` | 16×16 ×3 | 768 | make the query/key/value vectors for attention (ch 6) |
| `layer0.attn_wo` | 16×16 | 256 | recombine the heads' outputs (ch 6) |
| `layer0.mlp_fc1` | 16→64 | 1024 | MLP expansion (ch 7) |
| `layer0.mlp_fc2` | 64→16 | 1024 | MLP contraction (ch 7) |
| `lm_head` | 16→27 | 432 | final projection: internal vector → one score per vocab token (ch 7) |
| **total** | | **4192** | matches the program's printout |

An *embedding table* is worth dwelling on, because it's how discrete symbols enter a
world of continuous math: you cannot do calculus on "the letter e", but you can on 16
floats, and gradient descent can *learn* what those floats should be. After training,
similar-behaving tokens end up with similar vectors — meaning becomes geometry.

**Why random, why tiny?** (`random.gauss(0, 0.08)` — bell-curve noise, mean 0, typical
size ±0.08.)

- *Random* breaks symmetry. If two neurons started with identical weights they'd
  receive identical gradients and stay clones forever — you'd have paid for 4,192
  parameters and gotten far fewer distinct ones. Randomness makes them specialize
  differently.
- *Tiny* keeps early outputs mild: near-uniform prediction probabilities (as ch 8's
  baseline check confirms) and no saturated/exploding activations. Real frameworks
  tune the std per-matrix (e.g. `1/sqrt(n)` schemes); one small constant is fine at
  this scale.

**`state_dict` and `params`**: the dict (named after PyTorch's convention for exactly
this) is the organized view — matrices by role. `params` is the flat view: one long
list of all 4,192 `Value`s, in whatever order. The optimizer (ch 9) doesn't care what
a parameter *does* — it treats them as an undifferentiated bag of numbers to nudge.
Note that both views hold references to the *same* `Value` objects — Python aliasing
doing real work.

## Terminology

- **Parameter / weight** — a number the model learns. Contrast **hyperparameter** — a
  number the human picks (`n_embd`, learning rate, ...).
  [Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
- **Embedding** — representing a discrete symbol as a learned vector of floats.
  [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding) ·
  [Jay Alammar: "The Illustrated Word2vec"](https://jalammar.github.io/illustrated-word2vec/)
  — the classic gentle intro to "meaning as geometry."
- **Positional embedding** — same trick for positions, so order is visible to a model
  that otherwise treats context as a set (becomes crucial in ch 6).
- **Gaussian / normal distribution** — the bell curve behind `random.gauss`.
  [Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- **Initialization & symmetry breaking** — why training must start from small random
  values. [Wikipedia](https://en.wikipedia.org/wiki/Weight_initialization)
- **state_dict** — PyTorch's name for "dict of parameter tensors by name"; saving one
  to disk *is* saving the model. [PyTorch docs](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## Lab

```bash
python3 labs/ch04_parameters.py
```

Prints the full matrix census with roles, shows how the count scales as you widen
(`n_embd`: roughly quadratic) or deepen (`n_layer`: linear) the model, and dumps the
raw initial embedding of `'e'` — sixteen meaningless little numbers, before training
gives them a job.

## Exercises

1. **Count without running:** With `n_layer=2` and everything else default, how many
   parameters? Compute it from the table above by hand, then check with
   `init_model(vocab_size, n_layer=2)`.
2. **The missing matrix:** GPT-2 shares one matrix for `wte` and `lm_head` ("weight
   tying") — microgpt keeps them separate. What do the two matrices' shapes have to do
   with each other, and how many parameters would tying save here?
3. **Symmetry in action (extreme edition):** Initialize a model with `std=0` (all
   zeros) and train ~20 steps (use `labs/common.py`'s `train`). Predict first: does
   the loss improve slowly, or not at all? Then run it, and explain the result by
   tracing what the forward pass computes when every weight is zero — and what that
   means for every gradient. (This is symmetry's worst case: not just clones, a
   fixed point training can never leave.)

Solutions: `solutions/ch04_solutions.py`

---
Next: [Chapter 5 — Building blocks: linear, softmax, rmsnorm](ch05-building-blocks.md)
