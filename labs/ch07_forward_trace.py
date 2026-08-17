"""Lab 07 — The full forward pass (chapters/ch07-mlp-residuals-forward-pass.md)

Run:  python3 labs/ch07_forward_trace.py

What to look for:
  * one token goes in as an id, comes out as vocab_size scores (logits)
  * the vector STAYS 16 numbers wide the whole way through, except the brief
    4x expansion inside the MLP and the final projection to vocab size
  * residual connections: the block's output is 'input + adjustment'
"""
import math
from common import load_names, build_vocab, init_model, linear, softmax, rmsnorm

docs = load_names()
uchars, BOS, vocab_size = build_vocab(docs)
model = init_model(vocab_size)
sd = model['state_dict']

def stats(x):
    d = [v.data for v in x]
    rms = math.sqrt(sum(v * v for v in d) / len(d))
    return f"len={len(d):<3} rms={rms:6.3f}  first3=[{', '.join(f'{v:+.2f}' for v in d[:3])} ...]"

# This is microgpt.py's gpt() (lines 108-144) with a print after every stage.
token_id, pos_id = uchars.index('e'), 1   # pretend 'e' arrives at position 1
keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
n_head, head_dim = model['n_head'], model['head_dim']

print(f"input: token 'e' (id {token_id}) at position {pos_id}\n")
tok_emb = sd['wte'][token_id]
pos_emb = sd['wpe'][pos_id]
print(f"token embedding      {stats(tok_emb)}")
print(f"position embedding   {stats(pos_emb)}")
x = [t + p for t, p in zip(tok_emb, pos_emb)]
print(f"sum of the two       {stats(x)}")
x = rmsnorm(x)
print(f"after rmsnorm        {stats(x)}")

for li in range(model['n_layer']):
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, sd[f'layer{li}.attn_wq'])
    k = linear(x, sd[f'layer{li}.attn_wk'])
    v = linear(x, sd[f'layer{li}.attn_wv'])
    keys[li].append(k); values[li].append(v)
    print(f"\n--- attention block (layer {li}) ---")
    print(f"q                    {stats(q)}")
    print(f"k (cached)           {stats(k)}")
    print(f"v (cached)           {stats(v)}")
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
        if h == 0:
            print(f"head 0 output        {stats(head_out)}   (4 numbers per head)")
    print(f"4 heads concatenated {stats(x_attn)}")
    x = linear(x_attn, sd[f'layer{li}.attn_wo'])
    print(f"after wo mix         {stats(x)}")
    x = [a + b for a, b in zip(x, x_residual)]
    print(f"+ residual           {stats(x)}   <- input + attention's adjustment")

    x_residual = x
    x = rmsnorm(x)
    h1 = linear(x, sd[f'layer{li}.mlp_fc1'])
    print(f"\n--- MLP block (layer {li}) ---")
    print(f"fc1 expand           {stats(h1)}   <- 16 -> 64, room to compute")
    h1 = [xi.relu() for xi in h1]
    dead = sum(1 for v in h1 if v.data == 0)
    print(f"after relu           {stats(h1)}   ({dead}/64 zeroed by relu)")
    x = linear(h1, sd[f'layer{li}.mlp_fc2'])
    print(f"fc2 contract         {stats(x)}   <- 64 -> 16, back to highway width")
    x = [a + b for a, b in zip(x, x_residual)]
    print(f"+ residual           {stats(x)}")

logits = linear(x, sd['lm_head'])
print(f"\nlm_head              {stats(logits)}   <- one score per vocab token")
probs = softmax(logits)
best = max(range(vocab_size), key=lambda i: probs[i].data)
label = 'BOS' if best == BOS else repr(uchars[best])
print(f"highest-probability next token: {label} at p={probs[best].data:.3f}")
print("(untrained, so that's noise — every prob is near 1/27 ~ 0.037)")
