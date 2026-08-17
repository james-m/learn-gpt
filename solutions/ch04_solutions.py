"""Solutions — Chapter 4 (Parameters & initialization).   Run: python3 solutions/ch04_solutions.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import load_names, build_vocab, init_model, train

docs = load_names(500)
uchars, BOS, vocab_size = build_vocab(docs)

# --- Exercise 1: parameter count at n_layer=2 -------------------------------
# Fixed matrices: wte 27*16=432, wpe 16*16=256, lm_head 27*16=432       -> 1120
# Per layer: 4 attention matrices 16*16 (=1024) + fc1 64*16 + fc2 16*64 (=2048) -> 3072
# n_layer=2: 1120 + 2*3072 = 7264
m2 = init_model(vocab_size, n_layer=2)
print(f"1. n_layer=2 -> {len(m2['params'])} params (hand count: 1120 + 2*3072 = 7264)")
assert len(m2['params']) == 7264

# --- Exercise 2: weight tying -----------------------------------------------
# wte is vocab_size x n_embd (27x16): row i = embedding OF token i.
# lm_head is also vocab_size x n_embd: row i = the vector x is DOTTED with to score
# token i as the next token. Same shape, mirror-image jobs (read a token in / score a
# token out) — so GPT-2 uses one matrix for both. Savings here: 27*16 = 432 params,
# ~10% of the model. Bonus: gives the embedding rows extra training signal.
print(f"2. tying wte and lm_head would save 27*16 = {27*16} params of {4192}")

# --- Exercise 3: std=0 — total gradient death -------------------------------
# Prediction: the loss does not move AT ALL. Trace: all weights zero
#   -> tok_emb + pos_emb = 0-vector -> every linear() outputs zeros -> logits all 0
#   -> softmax uniform -> loss = -log(1/27) = 3.2958 exactly, every step.
# Gradients: dloss/dlogits is NOT zero (probs - onehot), but pushing it further down:
#   every path from any parameter to the loss passes through a multiplication by some
#   OTHER zero-valued weight or zero-valued activation, so every parameter's grad is
#   0 * something = 0. Zero grads -> Adam's update is 0 -> the parameters stay zero.
# A perfect fixed point: training is stuck on a knife's edge forever. This is the
# extreme version of the symmetry argument — random init exists to avoid this whole
# family of self-reinforcing coincidences.
m0 = init_model(vocab_size, std=0)
losses = train(m0, docs, uchars, BOS, num_steps=15)
print(f"3. std=0 losses over 15 steps: {sorted(set(round(l, 4) for l in losses))}")
print("   frozen at exactly -log(1/27) = 3.2958. Not slow learning — NO learning.")
