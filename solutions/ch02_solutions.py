"""Solutions — Chapter 2 (Autograd I: the Value class).   Run: python3 solutions/ch02_solutions.py"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import Value

# --- Exercise 1: y = x1*x2 + x2, at x1=4, x2=3 ------------------------------
# Nudge argument:
#   dy/dx1: nudging x1 by h changes x1*x2 by x2*h = 3h, and the +x2 term not at all -> 3
#   dy/dx2: nudging x2 by h changes x1*x2 by x1*h = 4h AND +x2 by h -> two routes, 4+1 = 5
x1, x2 = Value(4.0), Value(3.0)
y = x1 * x2 + x2
y.backward()
print(f"1. dy/dx1 = {x1.grad} (expected 3), dy/dx2 = {x2.grad} (expected 5 — two routes!)")
assert x1.grad == 3 and x2.grad == 5

# --- Exercise 2: add a tanh op ----------------------------------------------
# Exactly the pattern of Value.exp/log/relu: compute the result, record the child,
# record the local derivative. d/dx tanh(x) = 1 - tanh(x)^2.
def tanh(v):
    t = math.tanh(v.data)
    return Value(t, (v,), (1 - t * t,))

x = Value(0.7)
out = tanh(x)
out.backward()
h = 1e-6
numeric = (math.tanh(0.7 + h) - math.tanh(0.7)) / h
print(f"2. tanh(0.7): autograd grad {x.grad:.6f} vs numeric {numeric:.6f}")
assert abs(x.grad - numeric) < 1e-4

# --- Exercise 3: why 0**-1 and log(negative) can't occur in microgpt --------
# The dangerous ops and where the model uses them:
#   ** -1   (division) — used by softmax (`e / total`) and rmsnorm (`(ms + 1e-5) ** -0.5`).
#     * softmax's `total` is a sum of exp()s: every exp is > 0, so total > 0. Never zero.
#     * rmsnorm's `ms` is a mean of squares, >= 0, and the `+ 1e-5` guard makes it > 0.
#   log     — used ONLY on `probs[target_id]` in the loss. A softmax output is
#     exp(...)/total: strictly positive (it can UNDERFLOW to 0.0 in float math if the
#     logit gaps grow past ~745 — a numerical, not mathematical, hazard; microgpt's
#     mild weights never get near it).
#   ** (other-1) inside __pow__'s local grad: 0.0 ** -0.5 would blow up the BACKWARD
#     pass even when forward succeeded — again saved by the 1e-5 guard.
# Moral: the architecture quietly guarantees every op stays in its safe domain,
# and where math alone doesn't guarantee it (rmsnorm), a tiny epsilon does.
print("3. see comments — softmax's positivity and rmsnorm's 1e-5 keep every op in-domain")
