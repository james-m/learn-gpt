"""Lab 05 — Building blocks: linear, softmax, rmsnorm (chapters/ch05-building-blocks.md)

Run:  python3 labs/ch05_building_blocks.py

What to look for:
  * linear() is just rows of weighted sums (a matrix-vector multiply)
  * softmax turns any scores into probabilities; subtracting the max changes nothing
    mathematically but prevents exp() from overflowing
  * rmsnorm rescales a vector to a standard "loudness" without changing its direction
"""
import math
from common import Value, linear, softmax, rmsnorm, bar

# --- linear ---------------------------------------------------------------
x = [Value(1.0), Value(2.0)]
w = [[Value(0.5), Value(0.5)],    # output 0 = 0.5*x0 + 0.5*x1  (an average)
     [Value(1.0), Value(-1.0)],   # output 1 = x0 - x1          (a difference)
     [Value(0.0), Value(3.0)]]    # output 2 = 3*x1
out = linear(x, w)
print("linear([1, 2], w) with 3 output rows:")
for i, o in enumerate(out):
    print(f"  out[{i}] = {o.data:+.1f}   (row {i} of w, dotted with x)")
print("each output is one learned 'weighted opinion' about the input vector\n")

# --- softmax --------------------------------------------------------------
logits = [Value(2.0), Value(1.0), Value(0.1)]
probs = softmax(logits)
print("softmax([2.0, 1.0, 0.1]):")
for l, p in zip(logits, probs):
    print(f"  logit {l.data:+.1f} -> prob {p.data:.3f}  {bar(p.data)}")
print(f"  sum of probs = {sum(p.data for p in probs):.6f} (always exactly 1)\n")

# shifting ALL logits by a constant changes nothing:
shifted = softmax([l + 100 for l in logits])
print("softmax([102.0, 101.0, 100.1]) — same gaps, same answer:")
print("  " + ", ".join(f"{p.data:.3f}" for p in shifted))

# ...which is exactly why the max-subtraction trick is safe. And necessary:
try:
    math.exp(1000)
except OverflowError as e:
    print(f"  math.exp(1000) without the trick -> OverflowError: {e}\n")

# temperature preview (chapter 10 makes real use of this):
for T in (0.5, 1.0, 2.0):
    p = softmax([l / T for l in logits])
    print(f"  T={T}: " + ", ".join(f"{q.data:.3f}" for q in p))
print("  low T sharpens the distribution, high T flattens it\n")

# --- rmsnorm --------------------------------------------------------------
quiet = [Value(0.01), Value(0.02), Value(-0.01), Value(0.02)]
loud = [Value(10.0), Value(20.0), Value(-10.0), Value(20.0)]
for label, vec in (("quiet", quiet), ("loud ", loud)):
    normed = rmsnorm(vec)
    rms = lambda v: math.sqrt(sum(x.data**2 for x in v) / len(v))
    print(f"rmsnorm({label}): rms {rms(vec):8.3f} -> {rms(normed):.3f}   "
          f"values {[round(x.data, 2) for x in normed]}")
print("both come out with rms ~1.0, and note: the two results are IDENTICAL —")
print("rmsnorm only fixes the volume; the direction (the information) is untouched.")
