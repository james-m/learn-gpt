"""Lab 02 — Autograd I: the Value class (chapters/ch02-autograd-the-value-class.md)

Run:  python3 labs/ch02_value.py

What to look for:
  * every arithmetic op returns a NEW Value that remembers its inputs (children)
    and the "local gradient" — the derivative of the op w.r.t. each input
  * .backward() fills in .grad on every node
  * the numerical check at the end: autograd agrees with finite differences
"""
from common import Value

# --- build a tiny expression: L = (a * b + c).relu() -----------------------
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

ab = a * b        # 2 * -3 = -6
s = ab + c        # -6 + 10 = 4
L = s.relu()      # relu(4) = 4

print("forward pass:")
print(f"  ab = a*b        -> data={ab.data}, children=(a, b), local_grads={ab._local_grads}")
print(f"  s  = ab+c       -> data={s.data},  children=(ab, c), local_grads={s._local_grads}")
print(f"  L  = s.relu()   -> data={L.data},  children=(s,),   local_grads={L._local_grads}")

# --- backward pass ---------------------------------------------------------
L.backward()
print("\nafter L.backward():")
print(f"  dL/da = {a.grad}   (expect b = -3: nudging a by h moves L by -3h)")
print(f"  dL/db = {b.grad}   (expect a =  2)")
print(f"  dL/dc = {c.grad}   (expect      1)")

# --- verify one gradient numerically (finite differences) ------------------
# The definition of a derivative: dL/da ~ (L(a+h) - L(a)) / h for tiny h.
def compute_L(av, bv, cv):
    return max(0.0, av * bv + cv)

h = 1e-6
numeric = (compute_L(2.0 + h, -3.0, 10.0) - compute_L(2.0, -3.0, 10.0)) / h
print(f"\nnumerical dL/da = {numeric:.6f}   autograd said {a.grad}")
assert abs(numeric - a.grad) < 1e-3

# --- the relu 'gate' -------------------------------------------------------
# If the sum were negative, relu would output 0 and KILL all gradients upstream.
a2, b2, c2 = Value(2.0), Value(-3.0), Value(1.0)   # 2*-3 + 1 = -5 -> relu -> 0
L2 = (a2 * b2 + c2).relu()
L2.backward()
print(f"\nwith c=1 (relu input is -5, output 0): dL/da = {a2.grad}, dL/db = {b2.grad}")
print("relu's local gradient is 0 for negative inputs, so nothing flows back. 'Dead' path.")
