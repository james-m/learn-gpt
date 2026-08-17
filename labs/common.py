"""Shared building blocks for the labs.

Everything in this file is copied from — or very lightly adapted from — ../microgpt.py,
so each lab can run standalone. (We can't just `import microgpt`, because importing that
file would kick off its full training run.) Comments note the microgpt.py lines each
piece comes from, so you can always diff against the original.

The one real difference: microgpt.py keeps the model in module-level globals
(state_dict, n_layer, ...). Here those live in a plain dict called `model` that gets
passed around, so labs can build several models side by side.

Pure Python, stdlib only — same spirit as the original.
"""

import math
import random
from pathlib import Path


# -----------------------------------------------------------------------------
# Dataset (microgpt.py:14-21) — with offline fallbacks so labs always run

def load_names(limit=None):
    """Return the names dataset as a list[str], trying in order:
    1. the vendored copy at ../data/names.txt
    2. downloading it (same URL microgpt.py uses)
    3. the small embedded sample in _names_sample.py
    """
    local = Path(__file__).resolve().parent.parent / 'data' / 'names.txt'
    if local.exists():
        docs = [line.strip() for line in open(local) if line.strip()]
    else:
        try:
            import urllib.request
            url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
            text = urllib.request.urlopen(url, timeout=10).read().decode()
            docs = [line.strip() for line in text.splitlines() if line.strip()]
        except Exception:
            from _names_sample import NAMES
            docs = list(NAMES)
    return docs[:limit] if limit else docs


# -----------------------------------------------------------------------------
# Tokenizer (microgpt.py:23-27)

def build_vocab(docs):
    """Return (uchars, BOS, vocab_size) exactly as microgpt.py builds them."""
    uchars = sorted(set(''.join(docs)))  # unique characters become token ids 0..n-1
    BOS = len(uchars)                    # one extra id for the Beginning of Sequence token
    vocab_size = len(uchars) + 1
    return uchars, BOS, vocab_size


# -----------------------------------------------------------------------------
# Autograd engine (microgpt.py:29-72) — copied verbatim

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

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
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
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


# -----------------------------------------------------------------------------
# Architecture helper functions (microgpt.py:92-106) — copied verbatim

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# -----------------------------------------------------------------------------
# Parameter initialization (microgpt.py:74-90), bundled into a `model` dict

def init_model(vocab_size, n_layer=1, n_embd=16, block_size=16, n_head=4, std=0.08, seed=42):
    rng = random.Random(seed)
    matrix = lambda nout, nin: [[Value(rng.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    state_dict = {'wte': matrix(vocab_size, n_embd),   # token embeddings
                  'wpe': matrix(block_size, n_embd),   # position embeddings
                  'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    return {'state_dict': state_dict, 'params': params, 'vocab_size': vocab_size,
            'n_layer': n_layer, 'n_embd': n_embd, 'block_size': block_size,
            'n_head': n_head, 'head_dim': n_embd // n_head}


# -----------------------------------------------------------------------------
# The model itself (microgpt.py:108-144), taking `model` instead of using globals.
# `attn_capture`, if given, is a list that receives one entry per (layer, head):
# the row of attention weights (as plain floats) this position computed.

def gpt(model, token_id, pos_id, keys, values, attn_capture=None):
    sd = model['state_dict']
    n_layer, n_head, head_dim = model['n_layer'], model['n_head'], model['head_dim']

    tok_emb = sd['wte'][token_id]                    # token embedding
    pos_emb = sd['wpe'][pos_id]                      # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]    # joint token and position embedding
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            if attn_capture is not None:
                attn_capture.append({'layer': li, 'head': h, 'pos': pos_id,
                                     'weights': [w.data for w in attn_weights]})
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, sd['lm_head'])
    return logits


# -----------------------------------------------------------------------------
# One document's loss (microgpt.py:155-169), reused by the training loop below

def doc_loss(model, doc, uchars, BOS, attn_capture=None):
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(model['block_size'], len(tokens) - 1)
    keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(model, token_id, pos_id, keys, values, attn_capture)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    return (1 / n) * sum(losses)


# -----------------------------------------------------------------------------
# Training loop (microgpt.py:146-184). optimizer='adam' matches the original;
# optimizer='sgd' is a deliberately simpler update for comparison in chapter 9.

def train(model, docs, uchars, BOS, num_steps=300, learning_rate=0.01,
          optimizer='adam', on_step=None):
    params = model['params']
    beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
    m = [0.0] * len(params)  # first moment buffer
    v = [0.0] * len(params)  # second moment buffer
    losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        loss = doc_loss(model, doc, uchars, BOS)
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
        for i, p in enumerate(params):
            if optimizer == 'adam':
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            else:  # plain stochastic gradient descent
                p.data -= lr_t * p.grad
            p.grad = 0
        losses.append(loss.data)
        if on_step:
            on_step(step, loss.data)
    return losses


# -----------------------------------------------------------------------------
# Inference (microgpt.py:186-200)

def sample_name(model, uchars, BOS, temperature=0.5, rng=None):
    rng = rng or random
    keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
    token_id = BOS
    out = []
    for pos_id in range(model['block_size']):
        logits = gpt(model, token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = rng.choices(range(model['vocab_size']), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        out.append(uchars[token_id])
    return ''.join(out)


# -----------------------------------------------------------------------------
# Small display helper used by several labs

def bar(x, width=40, lo=0.0, hi=1.0):
    """Render x as an ASCII bar, e.g. bar(0.5) -> '####################'."""
    n = int(round((x - lo) / (hi - lo) * width))
    return '#' * max(0, min(width, n))
