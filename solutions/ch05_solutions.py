"""Solutions — Chapter 5 (Building blocks).   Run: python3 solutions/ch05_solutions.py"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import Value, softmax, rmsnorm

# --- Exercise 1: softmax([1, 1, 3]) by hand ---------------------------------
# e^1 = 2.718, e^1 = 2.718, e^3 = 20.086; total = 25.52
# -> [2.718/25.52, 2.718/25.52, 20.086/25.52] = [0.106, 0.106, 0.787]
probs = softmax([Value(1.0), Value(1.0), Value(3.0)])
print("1. softmax([1,1,3]) =", [round(p.data, 3) for p in probs], "(hand: [0.106, 0.106, 0.787])")

# --- Exercise 2: why not just divide by the sum? ----------------------------
# Failure A — signs: logits can be negative. naive([2, -2]) = [2/0, -2/0]: divide by
#   zero. Even without hitting zero exactly, a "probability" of -0.5 is meaningless.
# Failure B — all-negative flips the ranking: naive([-1, -3]) = [-1/-4, -3/-4]
#   = [0.25, 0.75] — the WORSE logit (-3) gets the HIGHER probability!
# softmax fixes both with exp(): every score becomes positive first, order preserved.
def naive(logits):
    s = sum(logits)
    return [l / s for l in logits]
print("2. naive([-1,-3]) =", naive([-1.0, -3.0]), "<- ranking inverted; softmax:",
      [round(p.data, 3) for p in softmax([Value(-1.0), Value(-3.0)])])

# --- Exercise 3: gradients flow through rmsnorm's scale ---------------------
# y = rmsnorm([a, b]); y[0] = a * (mean(a^2,b^2) + 1e-5)^-0.5. The scale term
# CONTAINS a, so dy0/da has two pieces (product rule): the obvious 'scale' piece and
# a negative correction through ms. Autograd composed it all from +, *, ** records.
a, b = Value(3.0), Value(4.0)
y = rmsnorm([a, b])
y[0].backward()
h = 1e-6
def y0(av, bv):
    ms = (av * av + bv * bv) / 2
    return av * (ms + 1e-5) ** -0.5
numeric = (y0(3.0 + h, 4.0) - y0(3.0, 4.0)) / h
print(f"3. dy0/da: autograd {a.grad:.6f} vs numeric {numeric:.6f}")
assert abs(a.grad - numeric) < 1e-4
# For intuition: the naive guess (just 1/rms ~ 0.283) is WRONG; the true value is
# smaller, because growing `a` also grows the rms, which pushes y[0] back down.
print(f"   (naive 1/rms would be {1/math.sqrt((9+16)/2):.3f} — the ms-route correction is real)")
