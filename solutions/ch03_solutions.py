"""Solutions — Chapter 3 (Autograd II: backward()).   Run: python3 solutions/ch03_solutions.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import Value

# --- Exercise 1: L = (a + b) * a, at a=2, b=3 -------------------------------
# Graph: s = a+b, L = s*a. Reverse sweep on paper:
#   L.grad = 1
#   L = s*a: s.grad += a * 1 = 2;  a.grad += s * 1 = 5   (first route into a)
#   s = a+b: a.grad += 1 * s.grad = 2  -> a.grad = 5+2 = 7;  b.grad += 1*2 = 2
# So dL/da = 7 (NOT just b: product rule via two routes), dL/db = 2.
a, b = Value(2.0), Value(3.0)
L = (a + b) * a
L.backward()
print(f"1. dL/da = {a.grad} (expected 7), dL/db = {b.grad} (expected 2)")
assert a.grad == 7 and b.grad == 2

# --- Exercise 2: calling backward() twice -----------------------------------
# Tempting wrong answer: "everything doubles." The truth is worse. Nothing resets
# between calls, so the second sweep's `+=` contributions are computed from grads
# that are ALREADY polluted. Trace it: after sweep 1, s.grad = 2. Sweep 2 adds
# another 2 (s.grad = 4), and then s's pass-down to a and b uses that inflated 4,
# not the honest 2. Final: a.grad = 7 + 5 + 4 = 16 (not 14), b.grad = 2 + 4 = 6
# (not 4). Corruption COMPOUNDS through intermediate nodes — the results aren't
# scaled versions of the truth, they're garbage. Hence: fresh graph every step,
# zero the leaves (microgpt.py:182), never re-run backward on a used graph.
a2, b2 = Value(2.0), Value(3.0)
L2 = (a2 + b2) * a2
L2.backward()
L2.backward()
print(f"2. after backward() twice: dL/da = {a2.grad} (not 7, not 14 — 16!), dL/db = {b2.grad} (6)")
assert a2.grad == 16 and b2.grad == 6

# --- Exercise 3: recursion depth --------------------------------------------
# build_topo recursion goes as deep as the LONGEST child-chain in the graph. Within
# one position's forward pass the long chains are things like softmax's `total`
# (a chain of 27 nested __add__ nodes from sum()) or a linear()'s 16-term sum —
# tens of nodes, not thousands. The graph is enormously WIDE (thousands of parameters
# — ch 4 counts them — feed in)
# but only modestly DEEP per position...
# ...except the chains COMPOUND across positions: position t's x depends on cached
# keys/values from all earlier positions, and the final loss sums all positions.
# Measure the actual depth (iteratively, so we don't smash the recursion limit
# ourselves) of a one-position graph vs a full-name loss:
def depth(root):
    seen, best, stack = {}, 0, [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            d = 1 + max((seen[c] for c in v._children), default=0)
            seen[v] = d
            best = max(best, d)
        elif v not in seen:
            seen[v] = 0
            stack.append((v, True))
            stack.extend((c, False) for c in v._children)
    return best

from common import load_names, build_vocab, init_model, doc_loss
docs = load_names(100)
uchars, BOS, vocab_size = build_vocab(docs)
model = init_model(vocab_size)
for name in ('al', 'gabriella'):
    loss = doc_loss(model, name, uchars, BOS)
    print(f"3. graph depth for loss of {name!r} ({len(name)} chars): {depth(loss)}")
print("   depth grows with sequence length (later positions read earlier positions'")
print("   cached k/v). Python's limit is 1000: microgpt survives because block_size=16")
print("   caps the compounding. Blow it up by raising block_size + training on long")
print("   documents — or by stacking many layers (n_layer multiplies the per-position")
print("   chain too). Real frameworks have no such limit: their 'backward' walks an")
print("   explicit graph iteratively instead of recursing the call stack.")
