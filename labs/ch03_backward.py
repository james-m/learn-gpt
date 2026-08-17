"""Lab 03 — Autograd II: backward() (chapters/ch03-autograd-backward.md)

Run:      python3 labs/ch03_backward.py
Explore:  python3 labs/ch03_backward.py 5 -1       # your own x, y in w = x*y + x

What to look for:
  * the topological sort: children always come before parents in `topo`,
    so walking it in reverse visits every node AFTER the nodes that depend on it
  * gradients ACCUMULATE with += — crucial when a value is used more than once
"""
import argparse
from common import Value

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('values', nargs='*', type=float, default=[3.0, 2.0],
                    metavar='X Y', help="leaf values for w = x*y + x (default: 3 2)")
args = parser.parse_args()
if len(args.values) != 2:
    parser.error("give exactly two numbers: x y")
xv, yv = args.values

# Give nodes names so we can print the topo order readably.
def show(name_map, v):
    return name_map.get(id(v), f"<tmp {v.data:.3g}>")

# --- 1. watch the topological sort happen ----------------------------------
x = Value(xv)
y = Value(yv)
u = x * y
w = u + x          # note: x is used TWICE (in u and directly in w)

names = {id(x): 'x', id(y): 'y', id(u): 'u=x*y', id(w): 'w=u+x'}

topo, visited = [], set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)
        topo.append(v)
build_topo(w)

print(f"w = x*y + x  with x={xv}, y={yv}  ->  w = {w.data}")
print("\ntopological order (children before parents):")
print("  " + "  ->  ".join(show(names, v) for v in topo))
print("backward() walks this list in REVERSE, so by the time it reaches a node,")
print("that node's .grad is already fully summed up from everything above it.\n")

# --- 2. gradient accumulation: why `child.grad += ...` uses += -------------
w.backward()
# w = x*y + x, so dw/dx = y + 1. The two 'routes' x takes to reach w
# (through u, and directly) each contribute, and += adds them together.
print(f"dw/dx = {x.grad}   (= y + 1 = {yv + 1}: a contribution of y through u, plus 1 directly)")
print(f"dw/dy = {y.grad}   (= x = {xv})")
assert x.grad == yv + 1 and y.grad == xv

# --- 3. what would go wrong with plain assignment --------------------------
# Simulate backward() with `=` instead of `+=`:
for v in [x, y, u, w]:
    v.grad = 0
w.grad = 1
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad = local_grad * v.grad   # BUG: overwrites instead of accumulating
print(f"\nwith '=' instead of '+=': dw/dx = {x.grad}  (WRONG — one of x's two routes was overwritten)")

# --- 4. this is also why the training loop must zero grads every step ------
# (microgpt.py:182 `p.grad = 0`) — otherwise step 2's gradients would be added
# on top of step 1's leftovers.
