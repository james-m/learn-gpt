"""Lab 09 — The training loop (chapters/ch09-training-loop.md)

Run:  python3 labs/ch09_training.py          (Adam, the default — ~1 min)
      python3 labs/ch09_training.py --sgd    (plain SGD for comparison)

What to look for:
  * the loss curve: noisy (each step sees ONE name) but trending down
  * Adam vs SGD: same gradients, different update rule — Adam adapts a
    per-parameter step size and gets moving much faster here
"""
import sys
import random
from common import load_names, build_vocab, init_model, train, bar

optimizer = 'sgd' if '--sgd' in sys.argv else 'adam'
NUM_STEPS = 240

random.seed(42)
docs = load_names()
random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)
model = init_model(vocab_size)

print(f"training {NUM_STEPS} steps with {optimizer.upper()}...")
losses = train(model, docs, uchars, BOS, num_steps=NUM_STEPS, optimizer=optimizer,
               on_step=lambda s, l: print(f"  step {s+1:3d} | loss {l:.3f}", end='\r'))
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
