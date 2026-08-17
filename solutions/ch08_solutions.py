"""Solutions — Chapter 8 (The loss).   Run: python3 solutions/ch08_solutions.py"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import load_names

# --- Exercise 1: perplexity of loss 2.2 -------------------------------------
loss = 2.2
print(f"1. perplexity = e^{loss} = {math.exp(loss):.1f}")
# ~9 effective options out of 27. Sensible: given a few letters of a name, the next
# letter is genuinely uncertain (vowel? consonant? stop?) but far from uniform —
# 'q' is essentially never next, 'a' often is. A single-digit perplexity on a
# 27-symbol alphabet means the model captured most of the easy structure.

# --- Exercise 2: irreducible uncertainty in the data ------------------------
# If two names share a prefix but diverge (or one ENDS where the other continues),
# then after that prefix the true distribution has multiple correct answers — no
# model can put probability 1 on both. Find real examples:
docs = load_names()
names = set(docs)
# the cleanest case: a name that is a strict prefix of another name
# (cheap way to find them: does name + one more letter also exist?)
prefix_pairs = []
for n in sorted(names):
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        if n + ch in names:
            prefix_pairs.append((n, n + ch))
            break
    if len(prefix_pairs) >= 5:
        break
print("2. names that are strict prefixes of other names:", prefix_pairs)
print("   after seeing the shorter one, both 'BOS (stop)' and 'continue' are correct")
print("   answers with nonzero true probability -> the best achievable loss is > 0.")
print("   (Also: after BOS alone, EVERY first letter in the data must get probability.)")

# --- Exercise 3: truthful die vs brave die ----------------------------------
# Truthful model: p(a)=0.6, p(o)=0.3. Outcomes a, o, a:
truthful = -(math.log(0.6) + math.log(0.3) + math.log(0.6)) / 3
print(f"3. truthful model: avg loss {truthful:.4f}")
# Brave model: p(a)=1.0 -> outcomes a (loss 0), o (-log 0 = INFINITY), a (0).
print("   brave model: avg loss = (0 + inf + 0)/3 = INFINITY")
print("   moral: cross-entropy rewards honest calibration, not boldness. One")
print("   confidently-wrong prediction outweighs any number of perfect ones, so the")
print("   optimal strategy is to report your true uncertainty. (This is why trained")
print("   LMs are 'calibrated' — and why loss can never reach 0 on uncertain data.)")
