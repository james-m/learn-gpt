# Chapter 2 — Autograd I: the `Value` class

## Where we are

The most important 40 lines in the file. `Value` wraps a single Python float and
overloads its arithmetic so that *doing math secretly records the math* — building a
graph of what depended on what. That recording is what makes training possible: it's
how the model will later answer "which of the thousands of numbers inside me (ch 4
meets them as *parameters*) made things worse, and by how much?" This chapter covers
the recording (the *forward pass*); chapter 3 covers the playback (`backward()`).

**Covers:** `microgpt.py:29-57`

## The code

```python
# Let there be Autograd to recursively apply the chain rule through a computation graph   # :29
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    # ... remaining dunders: __rsub__, __rmul__, __truediv__, __rtruediv__ (:54-57)
```

## Walkthrough

**The Python part** (which you already know): `__add__`, `__mul__` etc. are operator
overloads, so `a * b + c` works on `Value`s. The `__r*__` variants handle
`3 * v` (float on the left). `__sub__` and `__truediv__` don't record anything new —
they're *derived*: `a - b` is `a + (-b)`, `a / b` is `a * b**-1`. Only five ops truly
exist: `+`, `*`, `**const`, `log`, `exp`, `relu`. The entire GPT is built from these.

**The recording.** Every operation returns a **new** `Value` holding:
- `data` — the ordinary numeric result;
- `_children` — which `Value`(s) it was computed *from*;
- `_local_grads` — the **derivative of this op with respect to each input**, as a
  plain number, computed on the spot.

So `a * b` remembers `(a, b)` and `(b.data, a.data)`. Run a whole expression and you've
grown a linked structure — a *computation graph* — whose leaves are your inputs and
whose root is the final result. Nothing is ever mutated; the graph only grows.

**What's a "local gradient," concretely?** It answers: *if I nudge this input by a
tiny amount `h`, how many `h`'s does this op's output move?* No calculus course
needed — you can check every entry in the table by nudging:

| op | output | nudge effect | local grad(s) |
|----|--------|--------------|---------------|
| `a + b` | `a+b` | nudge either input by `h` → output moves by exactly `h` | `(1, 1)` |
| `a * b` | `a*b` | nudge `a` by `h` → output moves by `b·h` | `(b, a)` |
| `a ** n` | `aⁿ` | e.g. `a²`: `(a+h)² ≈ a² + 2a·h` | `n·aⁿ⁻¹` |
| `a.exp()` | `eᵃ` | the curve whose steepness equals its own height | `eᵃ` |
| `a.log()` | `ln a` | flattens out as `a` grows | `1/a` |
| `a.relu()` | `max(0,a)` | passes nudges through if `a>0`, else blocks them | `1` or `0` |

Try one: `a=3, b=4`, so `a*b=12`. Nudge `a` to `3.001` → `12.004`. The output moved 4
`h`'s, and indeed the local grad for `a` is `b = 4`.

**relu deserves a note** — it's the only *nonlinear* op in the model's layers (ch 7
explains why one is required). Its "derivative" is a gate: inputs above zero pass
gradient through untouched; inputs below zero pass nothing. `float(self.data > 0)` is
that gate in one expression.

**Why scalars?** Real frameworks (PyTorch etc.) do this identically, but per *tensor*
(array) instead of per number, which is thousands of times faster. Scalar autograd is
the same idea with the training wheels visible — perfect for reading, hopeless for
speed. That asymmetry is the "everything else is just efficiency" from the docstring.

## Terminology

- **Derivative** — how much a function's output moves per tiny nudge of its input; the
  slope. [Wikipedia](https://en.wikipedia.org/wiki/Derivative) ·
  [3Blue1Brown, Essence of Calculus ch. 2](https://www.youtube.com/watch?v=9vKqVkMQHKk)
  — if you refresh one math concept for this course, make it this one.
- **Automatic differentiation (autograd)** — computing exact derivatives of code by
  recording the operations as they run, rather than by symbolic algebra or
  approximation. [Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)
- **Computation graph** — the recorded network of operations linking inputs to output.
  [Deep dive: colah's "Calculus on Computational Graphs"](https://colah.github.io/posts/2015-08-Backprop/)
- **Local gradient** — one edge's derivative in that graph: the op's output w.r.t. one
  direct input, ignoring everything else.
- **ReLU (rectified linear unit)** — `max(0, x)`; the standard "cheap nonlinearity."
  [Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- **micrograd** — Karpathy's earlier standalone version of exactly this class, with a
  legendary lecture building it from scratch.
  [Repo](https://github.com/karpathy/micrograd) ·
  [Video: "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0)
  — the ideal companion to chapters 2–3.

## Lab

```bash
python3 labs/ch02_value.py
python3 labs/ch02_value.py 2 -3 1    # explore: your own a b c (these shut the relu gate)
```

Builds `L = (a*b + c).relu()`, prints what each node recorded, runs `backward()`
(a preview of ch 3), and then **verifies the gradient numerically** by actually nudging
an input — the finite-difference check. If you take one habit from this course:
gradients are never mystical, you can always check them by nudging.

## Exercises

1. **Predict, then check:** For `y = x1 * x2 + x2` with `x1=4, x2=3` — what are
   `dy/dx1` and `dy/dx2`? Work it out by the nudge argument, then verify with `Value`.
2. **Add an op:** Implement `tanh` on `Value` (`math.tanh(x)`; its local grad is
   `1 - tanh(x)²`). Verify it against a finite difference. This is exactly how you'd
   extend the engine with GeLU, sigmoid, etc.
3. **Find the flaw:** `Value(0.0) ** -1` and `Value(-1.0).log()` both explode. Why is
   that acceptable for microgpt — which lines of the model guarantee these can't occur?
   (You'll be able to fully answer after ch 5; take a first guess now.)

Solutions: `solutions/ch02_solutions.py`

---
Next: [Chapter 3 — Autograd II: backward()](ch03-autograd-backward.md)
