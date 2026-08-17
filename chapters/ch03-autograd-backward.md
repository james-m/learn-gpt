# Chapter 3 — Autograd II: `backward()`

## Where we are

Chapter 2 recorded the computation graph; this chapter plays it back. `backward()` is
backpropagation in 14 lines: starting from the final node (in training, the loss), it
walks the graph once and stamps every node with `grad` — *"if this number had been a
tiny bit bigger, the loss would have changed by this much."* For the 4,192 parameters,
that's the exact information needed to improve them.

**Covers:** `microgpt.py:59-72`

## The code

```python
    def backward(self):                                        # :59
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

## Walkthrough

**The chain rule, minus the notation.** Suppose the loss `L` depends on `w`, and `w`
was computed from `x`. If nudging `w` by `h` moves `L` by `2h` (`w.grad = 2`), and
nudging `x` by `h` moves `w` by `5h` (local grad 5), then nudging `x` moves `L` by
`5 · 2 = 10` h's. **Effects multiply along a path.** That's the entire chain rule, and
it's the last line of the code: `child.grad += local_grad * v.grad`.

Two more facts complete the picture:

- **Paths add.** If `x` reaches the loss through several routes, each route's
  contribution sums — hence `+=`, not `=`. (Every parameter matrix is used at every
  position of the sequence, so this isn't an edge case; it's the norm.)
- **The root starts at 1** (`self.grad = 1`): nudging the loss moves the loss 1:1.

**Why the topological sort?** The multiply-then-pass-down step for node `v` uses
`v.grad` — so `v.grad` had better be *finished* (all routes summed) before `v` passes
blame to its children. A **topological order** guarantees exactly that: `build_topo`
is a depth-first traversal that appends a node only after all its children, so `topo`
ends with the loss; walking it in `reversed()` order means every node is processed
after all its *parents* have contributed to it. One linear sweep, every gradient exact.
No iteration, no convergence, no approximation.

**Cost intuition:** the backward sweep touches each graph edge once, doing one
multiply-add — so getting all 4,192 gradients costs about the same as one forward
pass. This near-free-ness (formally ~2× forward cost) is *the* reason deep learning is
computationally feasible at all.

**Two practical consequences you'll meet again in ch 9:**
1. `grad` fields are never reset here — accumulation is a feature *within* one
   backward pass but a bug *across* training steps, so the training loop must zero
   them (`microgpt.py:182`, `p.grad = 0`).
2. Everything computed during the forward pass is kept alive (the graph holds
   references) until backward finishes — this is why training big models eats memory.

## Terminology

- **Backpropagation** — this algorithm: one reverse sweep of the chain rule over the
  computation graph. [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) ·
  [3Blue1Brown: "What is backpropagation really doing?"](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
  (intuition) and [the calculus follow-up](https://www.youtube.com/watch?v=tIeHLnjs5U8) ·
  [colah: "Calculus on Computational Graphs"](https://colah.github.io/posts/2015-08-Backprop/)
  — short, precise, uses these exact ideas.
- **Chain rule** — derivatives multiply along a chain of dependencies.
  [Wikipedia](https://en.wikipedia.org/wiki/Chain_rule)
- **Gradient** — the collection of all these derivatives, one per parameter; "the
  direction that increases the loss fastest." Descending it is ch 9's job.
  [Wikipedia](https://en.wikipedia.org/wiki/Gradient)
- **Topological sort** — ordering a dependency graph so every node comes after the
  things it depends on. Same idea as `make`, import resolution, or spreadsheet
  recalculation. [Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting)
- **Reverse-mode vs. forward-mode AD** — going loss→inputs (as here) gets *all* input
  gradients in one sweep; the alternative direction would need one sweep *per
  parameter*. 4,192 sweeps vs. 1 — that's the whole argument.
  [Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)

## Lab

```bash
python3 labs/ch03_backward.py
python3 labs/ch03_backward.py 5 -1    # explore: your own x y
```

Builds `w = x*y + x` (note `x` used twice), prints the actual topological order, shows
`dw/dx = y + 1` with both routes summed — then deliberately re-runs the sweep with `=`
instead of `+=` so you can see the wrong answer that produces.

## Exercises

1. **Trace by hand:** For `L = (a + b) * a` with `a=2, b=3`, draw the graph, run the
   reverse sweep on paper, and find `dL/da` and `dL/db`. (Careful: `a` has two routes;
   the answer for `a` is *not* just `b`.) Check with `Value`.
2. **Double backward bug:** Call `L.backward()` twice in a row on the same graph.
   What do the grads read now, and why exactly?
3. **Recursion depth:** `build_topo` recurses once per node along a path. Training a
   name builds paths thousands of nodes long — Python's default recursion limit is
   1000, yet microgpt doesn't crash. Look at the *shape* of the graph for one
   position's forward pass and explain why the deepest recursion stays manageable.
   Then estimate: what change to the model would blow it up?

Solutions: `solutions/ch03_solutions.py`

---
Next: [Chapter 4 — Parameters & initialization](ch04-parameters-and-initialization.md)
