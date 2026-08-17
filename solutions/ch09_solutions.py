"""Solutions — Chapter 9 (The training loop).   Run: python3 solutions/ch09_solutions.py
(Several short training runs — takes a couple of minutes.)"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import load_names, build_vocab, init_model, train, doc_loss

random.seed(42)
docs = load_names(); random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)
STEPS = 120

# --- Exercise 1: learning-rate safari ---------------------------------------
print(f"1. learning-rate safari ({STEPS} steps each):")
for lr in (0.0001, 0.01, 1.0):
    m = init_model(vocab_size)
    try:
        losses = train(m, docs, uchars, BOS, num_steps=STEPS, learning_rate=lr)
        print(f"   lr={lr:<7}: first-20 avg {sum(losses[:20])/20:.3f} -> last-20 avg {sum(losses[-20:])/20:.3f}")
    except (OverflowError, ValueError) as e:
        print(f"   lr={lr:<7}: CRASHED mid-training ({type(e).__name__}: {e})")
# Typical readings:
#   0.0001 — a flat line barely below 3.3: crawling. Wastes compute but is honest
#            about it; you can SEE nothing is happening.
#   0.01   — healthy descent.
#   1.0    — the scary one: it may crash, but it may also 'train', bouncing around a
#            mediocre loss. A too-big LR that doesn't diverge LOOKS like a working
#            run with a worse architecture — much harder to diagnose than a flatline.

# --- Exercise 2: kill the learning-rate decay -------------------------------
# train() implements decay internally (lr_t = lr * (1 - step/num_steps)). Compare
# against a constant-lr variant done inline:
def train_const_lr(model, num_steps, lr=0.01):
    params = model['params']
    beta1, beta2, eps = 0.85, 0.99, 1e-8
    m = [0.0] * len(params); v = [0.0] * len(params)
    out = []
    for step in range(num_steps):
        loss = doc_loss(model, docs[step % len(docs)], uchars, BOS)
        loss.backward()
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            p.data -= lr * (m[i] / (1 - beta1**(step+1))) / ((v[i] / (1 - beta2**(step+1))) ** 0.5 + eps)
            p.grad = 0
        out.append(loss.data)
    return out

ma, mb = init_model(vocab_size), init_model(vocab_size)
la = train(ma, docs, uchars, BOS, num_steps=300)          # with decay
lb = train_const_lr(mb, 300)                              # constant lr
print(f"\n2. decay vs constant over 300 steps: last-40 avg {sum(la[-40:])/40:.4f} vs {sum(lb[-40:])/40:.4f}")
# Early on the two are nearly indistinguishable — decay's lr has only dropped ~30%.
# Decay pays off at the END of long runs, when small steps let the model settle into
# a minimum instead of orbiting it. Lesson: schedules are about the endgame.

# --- Exercise 3: forget to zero the gradients -------------------------------
def train_no_zero(model, num_steps, lr=0.01):
    params = model['params']
    out = []
    for step in range(num_steps):
        loss = doc_loss(model, docs[step % len(docs)], uchars, BOS)
        loss.backward()
        for p in params:
            p.data -= lr * p.grad        # plain SGD, and NO p.grad = 0
        out.append(loss.data)
    return out

m3 = init_model(vocab_size)
try:
    l3 = train_no_zero(m3, 40)
    print(f"\n3. no grad zeroing: losses {['%.2f' % x for x in l3[::8]]} ...")
except (OverflowError, ValueError) as e:
    print(f"\n3. no grad zeroing: CRASHED ({type(e).__name__})")
# Mechanism: backward() ACCUMULATES (ch 3's `+=`). Without zeroing, step k's update
# uses grad_1 + grad_2 + ... + grad_k — an ever-growing sum of stale directions. The
# effective step size grows without bound, parameters swing harder each step, and the
# loss explodes (or the math overflows). One line, catastrophic difference.

# --- Exercise 4: Adam with a perfectly steady gradient ----------------------
# If p.grad == 0.1 every step: m -> 0.1, v -> 0.01, so the update is
#   lr * m / sqrt(v) = lr * 0.1 / 0.1 = lr  — the gradient's SIZE cancels out!
# Adam steps by ~lr for any parameter whose gradient direction is consistent,
# whether the raw gradient is 0.1 or 0.0001. It's a sign-consistency detector with
# per-parameter units, which is why one global lr works across wildly different
# layers — and why lr is THE knob that matters for Adam.
print("\n4. steady grad 0.1: update settles to lr*0.1/sqrt(0.01) = lr — magnitude cancels.")
