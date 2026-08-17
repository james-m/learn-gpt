"""Solutions — Chapter 10 (Inference).   Run: python3 solutions/ch10_solutions.py
(Trains one model for ~300 steps first — about a minute.)"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import load_names, build_vocab, init_model, gpt, softmax, train, sample_name

random.seed(42)
docs = load_names(); random.shuffle(docs)
uchars, BOS, vocab_size = build_vocab(docs)
model = init_model(vocab_size)
print("training 300 steps...")
train(model, docs, uchars, BOS, num_steps=300,
      on_step=lambda s, l: print(f"  step {s+1:3d} | loss {l:.3f}", end='\r'))
print()

# --- Exercise 1: greedy decoding --------------------------------------------
# Greedy is deterministic: same start (BOS), same weights, same argmax at every
# step -> exactly ONE possible output, ever.
def greedy():
    keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
    token_id, out = BOS, []
    for pos_id in range(model['block_size']):
        logits = gpt(model, token_id, pos_id, keys, values)
        probs = softmax(logits)
        token_id = max(range(vocab_size), key=lambda i: probs[i].data)
        if token_id == BOS:
            break
        out.append(uchars[token_id])
    return ''.join(out)

runs = {greedy() for _ in range(5)}
print(f"1. greedy, 5 runs -> {runs} — {len(runs)} distinct name(s). Determinism demo done.")

# --- Exercise 2: complete a prefix ------------------------------------------
def complete(prefix, temperature=0.5, rng=None):
    rng = rng or random
    keys, values = [[] for _ in range(model['n_layer'])], [[] for _ in range(model['n_layer'])]
    # feed BOS + the prefix, filling the KV cache; only the LAST logits matter
    ids = [BOS] + [uchars.index(c) for c in prefix]
    for pos_id, token_id in enumerate(ids):
        logits = gpt(model, token_id, pos_id, keys, values)
    out = list(prefix)
    for pos_id in range(len(ids), model['block_size']):
        probs = softmax([l / temperature for l in logits])
        token_id = rng.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        out.append(uchars[token_id])
        logits = gpt(model, token_id, pos_id, keys, values)
    return ''.join(out)

print("2. completions of 'ka':", '  '.join(complete('ka') for _ in range(10)))

# --- Exercise 3: novelty audit ----------------------------------------------
names = set(docs)
for T in (0.5, 0.1):
    rng = random.Random(7)
    samples = [sample_name(model, uchars, BOS, temperature=T, rng=rng) for _ in range(50)]
    new = [s for s in samples if s and s not in names]
    print(f"3. T={T}: {len(new)}/50 samples are NOT in the dataset")
# Expect more novelty at T=0.5 than T=0.1. Low temperature retreats to the highest-
# probability letter sequences — which are high-probability precisely because they
# (or things like them) are common in the data, so samples collapse toward real,
# frequent names. Higher T explores lower-probability paths: more inventions (and
# more junk). Creativity and reliability are the same dial.

# --- Exercise 4: the capstone read ------------------------------------------
# No code. Open microgpt.py, read all 200 lines, and name the chapter for each:
#   :14-27 ch1 | :29-72 ch2-3 | :74-90 ch4 | :92-106 ch5 | :108-134 ch6 |
#   :135-144 ch7 | :146-149, :171-184 ch9 | :160-169 ch8 | :186-200 ch10.
# Anything that still resists: /explain <term> in Claude Code.
print("4. capstone: see the line->chapter map in this file's comments.")
