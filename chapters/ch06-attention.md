# Chapter 6 — Attention

## Where we are

The heart of the transformer, and the reason this architecture beat everything before
it. So far every piece has processed *one* position's vector in isolation; attention is
the step where a position gets to **look back at everything before it and pull in what
it needs**. If you fully absorb one chapter in this course, make it this one.

**Covers:** `microgpt.py:108-134` (the first half of `gpt()`)

## The code

```python
def gpt(token_id, pos_id, keys, values):                       # :108
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
```

## Walkthrough

**Entering the model** (`:109-112`): a token arrives as a bare id. Two table lookups —
`wte[token_id]` ("what am I?") and `wpe[pos_id]` ("where am I?") — get **added**
elementwise into one 16-number vector `x`. From here to the end, the model is just
transformations of `x`.

**The problem attention solves.** Predicting the next letter from *only* the current
letter would make the model a lookup table of letter pairs. To do better, position 4 of
`emma` needs to know there was an `e` at the start and a double-`m` in the middle.
But context length varies, and "what matters" depends on content. Attention's answer:
let each position *ask* for what it needs.

**The library metaphor** for the three vectors every position computes (`:118-120`):

- **query** `q` — the question I'm asking ("looking for vowels behind me")
- **key** `k` — my label, how I appear to searchers ("I'm an early-position vowel")
- **value** `v` — the payload I hand over if someone attends to me

All three are just `linear` projections of the same `x` — three learned "views" of one
vector. The names are database/dictionary vocabulary on purpose: attention is a *soft*
dictionary lookup. Where a Python dict matches a query to one exact key, attention
scores the query against *every* key (dot product — big when vectors align, ch 5) and
takes a softmax-weighted **blend of all the values**. Fuzzy, differentiable `dict.get`.

**The mechanics** (`:129-131`), for the position currently being processed:

1. `attn_logits[t]` = how well my query matches the key at past position `t` — one dot
   product per visible position, scaled by `1/sqrt(head_dim)` (with `head_dim=4`,
   dot products of 4-term sums have a natural size of ~`sqrt(4)`; dividing keeps the
   logits O(1) so softmax starts soft rather than saturated).
2. `softmax` → **attention weights**: a probability distribution over past positions.
   "Where am I looking, and how hard." This is the single most interpretable object in
   the transformer — the lab prints these matrices.
3. `head_out[j]` = weighted average of everyone's values. If I attend 0.8 to the `e`
   and 0.2 to BOS, my output is 0.8·(e's value) + 0.2·(BOS's value).

**The KV cache** (`:121-122`) — microgpt's slickest move, easy to miss. `gpt()`
processes ONE token per call. Each call computes this position's `k` and `v` and
**appends them to lists that persist across calls** (`keys`, `values` are created
once per sequence, outside — `microgpt.py:161`, `:190`). So when position 4 runs,
`keys[li]` holds keys for positions 0–4 — the model literally *cannot* peek at the
future, because the future hasn't been computed yet. Two birds, one design:

- **Causality.** Textbook implementations process all positions at once and must
  *mask* the attention matrix (set future logits to −∞) to forbid look-ahead. The
  cache gets causality *structurally* — the lab's triangular matrices with no masking
  code anywhere.
- **Efficiency.** This is the same "KV cache" that makes real LLM chatbots usable:
  generating token N reuses all cached keys/values instead of recomputing the whole
  history. What's a memory-hungry optimization at scale is here just... the natural
  way to write it.

**Multi-head** (`:124-132`): rather than one attention over 16 dims, slice q/k/v into
4 independent chunks of 4 (`q[hs:hs+head_dim]`), run the identical procedure per
chunk, and concatenate the outputs back to 16. Each head has its own little q/k/v
subspace, so each can learn a *different* lookup pattern simultaneously — one head
tracking the previous letter, another watching BOS/position-1 (names' first letters
are very informative), another counting vowels. With one head you'd get one question
per position; four heads, four parallel questions. `wo` (`:133`) then lets the heads'
findings talk to each other — without it, head 0's output could only ever influence
dims 0–3.

**What attention is *not*:** there's no recurrence (no hidden state threaded through
time like an RNN) and no convolution windows. Any position can reach any other in one
hop, and all positions' attention could run in parallel — the two properties that made
transformers both stronger and vastly more parallelizable than what preceded them.

## Terminology

- **Attention / self-attention** — each position builds a weighted average over (its
  own sequence's) positions, with learned, content-dependent weights.
  [Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning)) ·
  [3Blue1Brown: "Attention in transformers, visually explained"](https://www.youtube.com/watch?v=eMlx5fFNoYc)
  — genuinely the best visual treatment in existence ·
  [Jay Alammar: "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)
  — the classic diagram-first walkthrough.
- **Query / Key / Value** — the three learned projections; soft-dictionary reading of
  attention. (Same vocabulary as key-value stores, deliberately.)
- **Scaled dot-product attention** — this exact recipe: `softmax(q·k / sqrt(d)) · v`,
  from the "Attention Is All You Need" paper.
  [Paper](https://arxiv.org/abs/1706.03762) (skim §3.2 after this chapter — it will
  read as a description of the code above).
- **Attention head / multi-head attention** — independent attention on a slice of the
  dims; parallel questions.
- **Causal (autoregressive) masking** — forbidding attention to the future; microgpt
  gets it via the KV cache instead of a mask.
  [Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)#Masked_attention)
- **KV cache** — storing past keys/values so each new token only computes its own.
  [Nice explainer](https://huggingface.co/blog/not-lain/kv-caching)
- **Context window (`block_size`)** — the maximum number of positions attention can
  span; why LLMs have a "context limit" at all (`wpe` runs out of rows).
- **Transformer** — the architecture this all adds up to.
  [Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) ·
  [bbycroft.net/llm](https://bbycroft.net/llm) — watch these exact q/k/v arrows move.

## Lab

```bash
python3 labs/ch06_attention.py                     # trains ~200 steps first; about a minute
python3 labs/ch06_attention.py sophia --steps 500  # explore: any word, any budget
```

Prints real attention matrices for `'emma'` — untrained (nearly uniform rows: no
opinions yet) vs. trained (structure emerges), all four heads. Note the triangular
shape: row `t` has exactly `t+1` entries, each row sums to 1.0, and the future is
simply absent. 200 steps is early training — expect *subtle* specialization here, and
try exercise 1 to sharpen it.

## Exercises

1. **More training, more structure:** bump the lab to 500+ steps. Do heads
   differentiate more? Which head watches the immediately-previous position hardest?
2. **Break causality (thought experiment first):** suppose at training time each
   position *could* see the next token's key/value. Why would training loss collapse
   toward zero while generation stays garbage? Explain in one paragraph; the mechanism
   matters more than running it.
3. **One head vs. four:** train two 200-step models, `n_head=1` vs `n_head=4` (same
   seed, same steps — `init_model` takes `n_head`). Compare final losses and sampled
   names. At this scale the gap is small — why would it grow with harder data?
4. **Read the scaling:** remove the `/ head_dim**0.5` in `labs/common.py`'s `gpt` and
   retrain. Watch the untrained attention matrices — are the rows still soft, or
   already spiky? Connect to ch 5's "softmax amplifies gaps."

Solutions: `solutions/ch06_solutions.py`

---
Next: [Chapter 7 — MLP, residuals, and the full forward pass](ch07-mlp-residuals-forward-pass.md)
