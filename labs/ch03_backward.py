"""Lab 03 — Autograd II: backward() (chapters/ch03-autograd-backward.md)

Run:  python3 labs/ch03_backward.py

What to look for:
  * the topological sort: children always come before parents in `topo`,
    so walking it in reverse visits every node AFTER the nodes that depend on it
  * gradients ACCUMULATE with += — crucial when a value is used more than once
"""
from common import Value

# Give nodes names so we can print the topo order readably.
def show(name_map, v):
    return name_map.get(id(v), f"<tmp {v.data:.3g}>")

# --- 1. watch the topological sort happen ----------------------------------
x = Value(3.0)
y = Value(2.0)
u = x * y          # u = 6
w = u + x          # w = 9   <-- note: x is used TWICE (in u and directly in w)

names = {id(x): 'x', id(y): 'y', id(u): 'u=x*y', id(w): 'w=u+x'}

topo, visited = [], set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)
        topo.append(v)
build_topo(w)

print("topological order (children before parents):")
print("  " + "  ->  ".join(show(names, v) for v in topo))
print("backward() walks this list in REVERSE, so by the time it reaches a node,")
print("that node's .grad is already fully summed up from everything above it.\n")

# --- 2. gradient accumulation: why `child.grad += ...` uses += -------------
w.backward()
# w = x*y + x, so dw/dx = y + 1 = 3. The two 'routes' x takes to reach w
# (through u, and directly) each contribute, and += adds them together.
print(f"dw/dx = {x.grad}   (= y + 1: one contribution of y=2 through u, plus 1 directly)")
print(f"dw/dy = {y.grad}   (= x = 3)")
assert x.grad == 3.0 and y.grad == 3.0

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
