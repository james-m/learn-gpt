"""Lab 06 — Attention (chapters/ch06-attention.md)

Run:  python3 labs/ch06_attention.py     (trains briefly first — about a minute)

What to look for:
  * the attention matrix is lower-triangular: position t can only look at 0..t,
    because the KV cache literally doesn't contain the future yet
  * an UNTRAINED head attends almost uniformly; a trained head has opinions
  * different heads learn different patterns
"""
import random
from common import (load_names, build_vocab, init_model, gpt, train)

random.seed(42)
docs = load_names()
random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)

def attention_for(model, word):
    """Run `word` through the model, capturing every head's attention weights."""
    tokens = [BOS] + [uchars.index(ch) for ch in word]
    keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
    capture = []
    for pos_id, token_id in enumerate(tokens):
        gpt(model, token_id, pos_id, keys, values, attn_capture=capture)
    return tokens, capture

def print_head(tokens, capture, head):
    labels = ['BOS' if t == BOS else uchars[t] for t in tokens]
    rows = [c for c in capture if c['head'] == head]
    print(f"    head {head}:  (rows = position doing the looking, cols = position looked at)")
    print("          " + "".join(f"{l:>6}" for l in labels))
    for c in rows:
        w = c['weights']
        cells = "".join(f"{w[t]:>6.2f}" if t < len(w) else "     ." for t in range(len(tokens)))
        print(f"    {labels[c['pos']]:>5} {cells}")

word = 'emma'

model = init_model(vocab_size)
tokens, capture = attention_for(model, word)
print(f"=== UNTRAINED model, word {word!r} ===")
print_head(tokens, capture, head=0)
print("    ('.' = future positions: no weight exists for them at all — causality for free)\n")

print("training 200 steps (be patient, this is pure-Python scalar math)...")
train(model, docs, uchars, BOS, num_steps=200,
      on_step=lambda s, l: print(f"  step {s+1:3d} | loss {l:.3f}", end='\r'))
print()

tokens, capture = attention_for(model, word)
print(f"\n=== TRAINED model, word {word!r} ===")
for h in range(model['n_head']):
    print_head(tokens, capture, h)
    print()
print("each row sums to 1.0 (it's a softmax). Compare heads: they specialize —")
print("some stare at the previous letter, some at BOS, some spread out.")
