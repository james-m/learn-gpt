"""Solutions — Chapter 7 (MLP, residuals, forward pass).   Run: python3 solutions/ch07_solutions.py
(Trains two small models — takes a minute or two.)"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import (load_names, build_vocab, init_model, train, linear, rmsnorm,
                    Value, gpt)
import common

# --- Exercise 1: two linears collapse into one ------------------------------
# linear(x, w1) then linear(., w2) is the matrix product w = w2 @ w1:
#   w[i][j] = sum_k w2[i][k] * w1[k][j]
# w1 = [[2,0],[0,3]], w2 = [[1,1],[1,-1]]  ->  w = [[2,3],[2,-3]]
w1 = [[Value(2.0), Value(0.0)], [Value(0.0), Value(3.0)]]
w2 = [[Value(1.0), Value(1.0)], [Value(1.0), Value(-1.0)]]
w  = [[Value(2.0), Value(3.0)], [Value(2.0), Value(-3.0)]]
for xv in ([1.0, 1.0], [5.0, -2.0]):
    x = [Value(v) for v in xv]
    two = [v.data for v in linear(linear(x, w1), w2)]
    one = [v.data for v in linear(x, w)]
    print(f"1. x={xv}: two linears {two} == one linear {one}")
    assert two == one
# No relu in between => the stack has exactly the expressive power of ONE matrix.

# --- Exercise 2: ablate the residual connections ----------------------------
# A gpt() clone with both `+ x_residual` adds removed:
def gpt_no_residual(model, token_id, pos_id, keys, values, attn_capture=None):
    sd = model['state_dict']
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(model['n_layer']):
        x = rmsnorm(x)
        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        x_attn = []
        hd = model['head_dim']
        for h in range(model['n_head']):
            hs = h * hd
            q_h = q[hs:hs+hd]
            k_h = [ki[hs:hs+hd] for ki in keys[li]]
            v_h = [vi[hs:hs+hd] for vi in values[li]]
            al = [sum(q_h[j] * k_h[t][j] for j in range(hd)) / hd**0.5 for t in range(len(k_h))]
            aw = common.softmax(al)
            x_attn.extend([sum(aw[t] * v_h[t][j] for t in range(len(v_h))) for j in range(hd)])
        x = linear(x_attn, sd[f'layer{li}.attn_wo'])          # no + x_residual
        x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])               # no + x_residual
    return linear(x, sd['lm_head'])

random.seed(42)
docs = load_names(); random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)

print("\n2. residuals vs none (150 steps each):")
for label, fwd in (('with residuals', gpt), ('no residuals  ', gpt_no_residual)):
    common_gpt = common.gpt
    common.gpt = fwd            # swap the forward used by common.doc_loss/train
    m = init_model(vocab_size)
    losses = train(m, docs, uchars, BOS, num_steps=150)
    common.gpt = common_gpt
    print(f"   {label}: last-30 avg loss {sum(losses[-30:])/30:.4f}")
# At depth 1 the gap is usually modest — one block's gradients survive without a
# highway. At depth 12 the no-residual path multiplies 24 transformations of the
# gradient; anything slightly <1 in 'gain' vanishes exponentially, anything >1
# explodes. Residuals make the identity the default and each block a small correction.

# --- Exercise 3: multiply census, MLP vs attention (position 5) -------------
# MLP: fc1 64x16 + fc2 16x64 = 2048 multiplies.
# Attention: q,k,v,wo projections 4 * (16x16) = 1024; the actual attention math at
# position t=5 (6 visible positions): logits 4 heads * 6 pos * 4 dims = 96, weighted
# sum another 96 -> 192.  MLP (2048) > all of attention (1216) — and the PROJECTIONS
# dwarf the attention mixing itself.
# Does it change late in a sequence? mixing grows linearly with t (at t=15: 512),
# still short of 2048 here. But scale the model up: mixing goes as context * n_embd,
# projections as n_embd^2 — with 100K-token contexts the mixing term wins, which is
# why long-context inference is attention-bound (and why the KV cache + attention
# variants like flash/sparse attention are where the engineering goes).
print("\n3. per-position multiplies at t=5: MLP 2048 vs attention 1024 (proj) + 192 (mix)")
