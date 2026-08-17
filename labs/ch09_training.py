"""Lab 09 — The training loop (chapters/ch09-training-loop.md)

Run:      python3 labs/ch09_training.py          (Adam, the default — ~1 min)
          python3 labs/ch09_training.py --sgd    (plain SGD for comparison)
Explore:  python3 labs/ch09_training.py --lr 1.0 --steps 100    # learning-rate safari
          python3 labs/ch09_training.py --sgd --lr 0.0001       # watch nothing happen

What to look for:
  * the loss curve: noisy (each step sees ONE name) but trending down
  * Adam vs SGD: same gradients, different update rule — Adam adapts a
    per-parameter step size and gets moving much faster here
"""
import argparse
import random
from common import load_names, build_vocab, init_model, train, bar

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--sgd', action='store_true', help="plain SGD instead of Adam")
parser.add_argument('--steps', type=int, default=240, help="training steps (default 240)")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate (default 0.01)")
args = parser.parse_args()
optimizer = 'sgd' if args.sgd else 'adam'
NUM_STEPS = args.steps

random.seed(42)
docs = load_names()
random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)
model = init_model(vocab_size)

print(f"training {NUM_STEPS} steps with {optimizer.upper()}, lr={args.lr}...")
try:
    losses = train(model, docs, uchars, BOS, num_steps=NUM_STEPS, optimizer=optimizer,
                   learning_rate=args.lr,
                   on_step=lambda s, l: print(f"  step {s+1:3d} | loss {l:.3f}", end='\r'))
except (OverflowError, ValueError) as e:
    print(f"\ntraining EXPLODED: {type(e).__name__}: {e}")
    print("congratulations, you've found a too-large learning rate — the parameters")
    print("swung so hard the math left its safe domain (ch 9, exercise 1).")
    raise SystemExit(0)
print()

# --- ASCII loss curve: average each bucket of 20 steps ---------------------
print(f"\nloss curve ({optimizer}, averaged over buckets of 20 steps):")
hi = max(losses)
for i in range(0, NUM_STEPS, 20):
    chunk = losses[i:i + 20]
    avg = sum(chunk) / len(chunk)
    print(f"  steps {i + 1:3d}-{i + len(chunk):3d}: {avg:5.3f} |{bar(avg, width=50, lo=0, hi=hi)}")

print(f"\nfirst-20 avg {sum(losses[:20]) / 20:.3f} -> last-20 avg {sum(losses[-20:]) / 20:.3f}")
print("individual steps are noisy — each sees a single name — but the trend is what")
print("matters. Now run the other optimizer flag and compare the two curves.")
