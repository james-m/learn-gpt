"""Lab 10 — Inference & temperature (chapters/ch10-inference.md)

Run:      python3 labs/ch10_sampling.py     (trains briefly first — about a minute)
Explore:  python3 labs/ch10_sampling.py --temps 0.01 3.0 --steps 500 --samples 12
          (your own temperature ladder, training budget, and samples per rung)

What to look for:
  * generation is just the forward pass in a loop: feed a token, get a
    probability distribution, roll the dice, feed the result back in
  * temperature: low T -> plays the favorites (repetitive, 'safe' names),
    high T -> adventurous (novel but sometimes unpronounceable)
"""
import argparse
import random
from common import load_names, build_vocab, init_model, gpt, softmax, train, sample_name

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--temps', nargs='+', type=float, default=[0.1, 0.5, 1.0, 1.5],
                    help="temperatures to sweep (default: 0.1 0.5 1.0 1.5)")
parser.add_argument('--steps', type=int, default=300, help="training steps (default 300)")
parser.add_argument('--samples', type=int, default=8, help="samples per temperature (default 8)")
args = parser.parse_args()
if any(t <= 0 for t in args.temps):
    parser.error("temperatures must be > 0 (T=0 would divide the logits by zero; "
                 "'greedy' is the LIMIT as T->0, not a value you can plug in)")

random.seed(42)
docs = load_names()
random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)
model = init_model(vocab_size)

print(f"training {args.steps} steps so samples aren't pure noise...")
train(model, docs, uchars, BOS, num_steps=args.steps,
      on_step=lambda s, l: print(f"  step {s+1:3d} | loss {l:.3f}", end='\r'))
print("\n")

# --- one generation step, dissected ----------------------------------------
keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
logits = gpt(model, BOS, 0, keys, values)
probs = softmax(logits)
top = sorted(range(vocab_size), key=lambda i: -probs[i].data)[:5]
print("fed BOS at position 0 — model's top-5 picks for a name's FIRST letter:")
for t in top:
    label = 'BOS' if t == BOS else uchars[t]
    print(f"  {label}: {probs[t].data:.3f}")
print("generation samples from this distribution, appends the pick, repeats.\n")

# --- temperature sweep -----------------------------------------------------
for T in args.temps:
    rng = random.Random(1234)  # same dice for every T — differences are all T's doing
    names = [sample_name(model, uchars, BOS, temperature=T, rng=rng) for _ in range(args.samples)]
    print(f"T={T:<4}: {'  '.join(n if n else '(empty)' for n in names)}")

print("\nT divides the logits before softmax. T->0 approaches 'always pick the")
print("argmax' (greedy); T=1 is the model's honest distribution; T>1 flattens it")
print("toward uniform noise. microgpt.py ships with T=0.5: mildly conservative.")
