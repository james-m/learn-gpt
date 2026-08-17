"""Solutions — Chapter 6 (Attention).   Run: python3 solutions/ch06_solutions.py
(Trains three small models — takes a couple of minutes.)"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import load_names, build_vocab, init_model, gpt, train, softmax, Value

random.seed(42)
docs = load_names()
random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)

# --- Exercise 1: more training -> more structure ----------------------------
# (Answer by experiment; we use 500 steps.) The head whose rows put the most weight
# on column t-1 is the "previous-letter head". Quantify instead of eyeball:
model = init_model(vocab_size)
print("1. training 500 steps...")
train(model, docs, uchars, BOS, num_steps=500,
      on_step=lambda s, l: print(f"   step {s+1:3d} | loss {l:.3f}", end='\r'))
print()
word = 'emma'
tokens = [BOS] + [uchars.index(c) for c in word]
keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
cap = []
for pos, t in enumerate(tokens):
    gpt(model, t, pos, keys, values, attn_capture=cap)
for h in range(model['n_head']):
    rows = [c['weights'] for c in cap if c['head'] == h and c['pos'] > 0]
    prev = sum(r[len(r) - 2] for r in rows) / len(rows)   # weight on position t-1
    bos_w = sum(r[0] for r in rows) / len(rows)           # weight on BOS (position 0)
    print(f"   head {h}: avg weight on previous position {prev:.2f}, on BOS {bos_w:.2f}")
print("   whichever head scores highest on the first column is your previous-letter head.")

# --- Exercise 2: why leaking the future collapses training loss -------------
# If position t can attend to position t+1's key/value, the v-vector of t+1 CONTAINS
# (a projection of) the identity of the very token the model is asked to predict.
# Gradient descent will rapidly discover the trivial strategy "attend to t+1, copy its
# value through wo -> lm_head" — driving train loss toward 0 without learning anything
# about names. At generation time there IS no t+1 yet (the token hasn't been sampled),
# so the crutch vanishes and output quality is garbage. This is 'data leakage' in its
# purest form, and it's why causal masking (or microgpt's append-only cache) is not an
# optimization but a correctness requirement.
print("\n2. see comments — leaking t+1's k/v lets the model copy the answer at train time")

# --- Exercise 3: n_head = 1 vs 4 --------------------------------------------
print("\n3. head-count comparison (200 steps each):")
for nh in (1, 4):
    m = init_model(vocab_size, n_head=nh)
    losses = train(m, docs, uchars, BOS, num_steps=200)
    print(f"   n_head={nh}: last-30-step avg loss {sum(losses[-30:])/30:.4f}")
# Expect a modest gap (sometimes noisy at this scale). Why it grows with harder data:
# one head = one attention distribution per position — a single question. Rich text
# needs several simultaneous, DIFFERENT lookups (syntax, coreference, position, topic),
# which is exactly what independent per-head q/k/v subspaces buy.

# --- Exercise 4: what the 1/sqrt(head_dim) scaling does ---------------------
# No retraining needed to see the mechanism: dot products of d independent ~N(0,1)
# terms have typical size ~sqrt(d). Feed softmax logits ~3x bigger and it saturates.
rng = random.Random(0)
for d in (4, 64):
    q = [rng.gauss(0, 1) for _ in range(d)]
    ks = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(5)]
    raw = [sum(qj * kj for qj, kj in zip(q, k)) for k in ks]
    scaled = [x / d ** 0.5 for x in raw]
    p_raw = softmax([Value(x) for x in raw])
    p_scl = softmax([Value(x) for x in scaled])
    print(f"4. head_dim={d:2d}: max prob unscaled {max(p.data for p in p_raw):.2f}, "
          f"scaled {max(p.data for p in p_scl):.2f}")
print("   unscaled attention starts life spiky (near one-hot): tiny gradients to the")
print("   losers, slow learning. Scaling keeps it soft so training can shape it.")
