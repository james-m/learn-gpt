"""Lab 08 — The loss (chapters/ch08-loss.md)

Run:      python3 labs/ch08_loss.py
Explore:  python3 labs/ch08_loss.py --steps 300           # longer training, lower loss
          python3 labs/ch08_loss.py 0.99 0.5 0.0001      # your own p's for the -log table

What to look for:
  * loss = -log(probability the model gave the RIGHT answer). Only that one
    probability matters; confidence about wrong answers hurts indirectly
    (probabilities must sum to 1).
  * an untrained model's loss lands almost exactly at -log(1/vocab_size):
    the loss of pure uniform guessing. That's your 'is training working?' baseline.
  * a few gradient steps push it below the baseline.
"""
import argparse
import math
import random
from common import load_names, build_vocab, init_model, doc_loss, train

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('probs', nargs='*', type=float, default=[0.9, 0.5, 0.25, 1 / 27, 0.01, 0.001],
                    help="probabilities for the -log(p) table (each in (0, 1])")
parser.add_argument('--steps', type=int, default=150, help="training steps (default 150)")
args = parser.parse_args()
if any(not 0 < p <= 1 for p in args.probs):
    parser.error("probabilities must be in (0, 1] — p=0 gives -log(0) = infinity, which is the point, "
                 "but math.log refuses to print it for you")

random.seed(42)
docs = load_names()
random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)

# --- what -log(p) feels like ----------------------------------------------
print("loss = -log(p_correct):")
for p in args.probs:
    print(f"  model gave correct answer p={p:<9.5f} -> loss {-math.log(p):7.3f}")
print("being confidently wrong (tiny p on the truth) is punished brutally;")
print("-log(p) -> infinity as p -> 0.\n")

# --- the uniform baseline --------------------------------------------------
baseline = -math.log(1 / vocab_size)
print(f"uniform-guessing baseline: -log(1/{vocab_size}) = {baseline:.4f}")

model = init_model(vocab_size)
untrained = sum(doc_loss(model, d, uchars, BOS).data for d in docs[:20]) / 20
print(f"untrained model, avg loss over 20 names: {untrained:.4f}  (~ the baseline — it knows nothing)")

# --- training moves it -----------------------------------------------------
print(f"\ntraining {args.steps} steps...")
train(model, docs, uchars, BOS, num_steps=args.steps,
      on_step=lambda s, l: print(f"  step {s+1:3d} | loss {l:.3f}", end='\r'))
trained = sum(doc_loss(model, d, uchars, BOS).data for d in docs[:20]) / 20
print(f"\nsame 20 names after training: avg loss {trained:.4f}")
print(f"\nbaseline {baseline:.3f} -> {trained:.3f}: the model now assigns the true next")
print(f"character p ~ {math.exp(-trained):.3f} on average, vs {1/vocab_size:.3f} for guessing.")
