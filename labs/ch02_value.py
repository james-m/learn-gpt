"""Lab 02 — Autograd I: the Value class (chapters/ch02-autograd-the-value-class.md)

Run:      python3 labs/ch02_value.py
Explore:  python3 labs/ch02_value.py 5 2 -1        # your own a, b, c in L=(a*b+c).relu()
          python3 labs/ch02_value.py 2 -3 1        # make the relu gate slam shut

What to look for:
  * every arithmetic op returns a NEW Value that remembers its inputs (children)
    and the "local gradient" — the derivative of the op w.r.t. each input
  * .backward() fills in .grad on every node
  * the numerical check at the end: autograd agrees with finite differences
"""
import argparse
from common import Value

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('values', nargs='*', type=float, default=[2.0, -3.0, 10.0],
                    metavar='A B C', help="leaf values for L=(a*b+c).relu() (default: 2 -3 10)")
args = parser.parse_args()
if len(args.values) != 3:
    parser.error("give exactly three numbers: a b c")
av, bv, cv = args.values

# --- build a tiny expression: L = (a * b + c).relu() -----------------------
a = Value(av)
b = Value(bv)
c = Value(cv)

ab = a * b
s = ab + c
L = s.relu()

print(f"forward pass for a={av}, b={bv}, c={cv}:")
print(f"  ab = a*b        -> data={ab.data}, children=(a, b), local_grads={ab._local_grads}")
print(f"  s  = ab+c       -> data={s.data},  children=(ab, c), local_grads={s._local_grads}")
print(f"  L  = s.relu()   -> data={L.data},  children=(s,),   local_grads={L._local_grads}")

# --- backward pass ---------------------------------------------------------
# Expected by the nudge argument: if the relu is open (s > 0), dL/da = b,
# dL/db = a, dL/dc = 1. If the relu is shut (s <= 0), everything is 0.
gate = 1.0 if s.data > 0 else 0.0
L.backward()
print("\nafter L.backward():")
print(f"  dL/da = {a.grad}   (expect b * gate = {bv * gate})")
print(f"  dL/db = {b.grad}   (expect a * gate = {av * gate})")
print(f"  dL/dc = {c.grad}   (expect 1 * gate = {gate})")
if gate == 0.0:
    print("  the relu input was <= 0: output clamped to 0, ALL gradients killed.")
    print("  nudging a, b, or c (slightly) changes nothing — a 'dead' path.")

# --- verify one gradient numerically (finite differences) ------------------
# The definition of a derivative: dL/da ~ (L(a+h) - L(a)) / h for tiny h.
def compute_L(av_, bv_, cv_):
    return max(0.0, av_ * bv_ + cv_)

h = 1e-6
numeric = (compute_L(av + h, bv, cv) - compute_L(av, bv, cv)) / h
print(f"\nnumerical dL/da = {numeric:.6f}   autograd said {a.grad}")
assert abs(numeric - a.grad) < 1e-3

# --- the relu 'gate', demonstrated on fixed values -------------------------
# 2*-3 + 1 = -5 -> relu -> 0: gradients die.
a2, b2, c2 = Value(2.0), Value(-3.0), Value(1.0)
L2 = (a2 * b2 + c2).relu()
L2.backward()
print(f"\nfixed demo, c=1 (relu input is -5, output 0): dL/da = {a2.grad}, dL/db = {b2.grad}")
print("relu's local gradient is 0 for negative inputs, so nothing flows back. 'Dead' path.")
