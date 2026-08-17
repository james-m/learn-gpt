"""Lab 04 — Parameters & initialization (chapters/ch04-parameters-and-initialization.md)

Run:      python3 labs/ch04_parameters.py
Explore:  python3 labs/ch04_parameters.py --n-embd 32 --n-layer 2 --n-head 8
          (census any model shape you like; defaults match microgpt.py)

What to look for:
  * where microgpt's "num params: 4192" printout comes from, matrix by matrix
  * how the count scales when you grow n_embd or n_layer
  * what the raw initial weights look like (small Gaussian noise around 0)
"""
import argparse
from common import load_names, build_vocab, init_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--n-embd', type=int, default=16, help="embedding width (default 16)")
parser.add_argument('--n-layer', type=int, default=1, help="transformer layers (default 1)")
parser.add_argument('--n-head', type=int, default=4, help="attention heads (default 4)")
args = parser.parse_args()
if args.n_embd % args.n_head:
    parser.error(f"n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head}) — "
                 "each head gets an equal slice (microgpt.py:79)")

docs = load_names()
uchars, BOS, vocab_size = build_vocab(docs)

model = init_model(vocab_size, n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head)
sd = model['state_dict']

print(f"vocab_size={vocab_size}  n_embd={model['n_embd']}  n_layer={model['n_layer']}  "
      f"n_head={model['n_head']}  head_dim={model['head_dim']}\n")

print(f"{'matrix':<18}{'shape':<12}{'params':>8}   role")
roles = {
    'wte':     'token embedding: one row (a learned vector) per token id',
    'wpe':     'position embedding: one row per position 0..block_size-1',
    'lm_head': 'maps final 16-dim vector to a score per vocabulary token',
    'attn_wq': 'makes the "query" vector (what am I looking for?)',
    'attn_wk': 'makes the "key" vector (what do I contain?)',
    'attn_wv': 'makes the "value" vector (what do I hand over if attended to?)',
    'attn_wo': 'mixes the concatenated head outputs back together',
    'mlp_fc1': 'MLP expand: n_embd -> 4*n_embd',
    'mlp_fc2': 'MLP contract: 4*n_embd -> n_embd',
}
total = 0
for name, mat in sd.items():
    nout, nin = len(mat), len(mat[0])
    count = nout * nin
    total += count
    short = name.split('.')[-1]
    print(f"{name:<18}{f'{nout}x{nin}':<12}{count:>8}   {roles.get(short, '')}")
print(f"{'':<18}{'':<12}{'-'*8}")
note = "(matches microgpt.py's 'num params' printout)" if (args.n_embd, args.n_layer, args.n_head) == (16, 1, 4) \
       else "(custom shape — microgpt.py's defaults give 4192)"
print(f"{'total':<18}{'':<12}{total:>8}   {note}")

# --- how the count scales --------------------------------------------------
print("\nparameter count vs. width and depth:")
for n_embd in (16, 32, 64):
    for n_layer in (1, 2):
        m = init_model(vocab_size, n_layer=n_layer, n_embd=n_embd)
        print(f"  n_embd={n_embd:<3} n_layer={n_layer}: {len(m['params']):>7,} params")
print("width scales roughly quadratically (the NxN matrices), depth linearly.")

# --- what initialization looks like ----------------------------------------
row = [p.data for p in sd['wte'][uchars.index('e')]]
print(f"\ninitial embedding for 'e' ({args.n_embd} random numbers, gauss(0, 0.08)):")
print('  [' + ', '.join(f'{x:+.3f}' for x in row) + ']')
print("meaningless now — training will slowly sculpt these into something useful.")
