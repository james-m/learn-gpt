"""Solutions — Chapter 1 (Data & tokenization).   Run: python3 solutions/ch01_solutions.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'labs'))
from common import load_names, build_vocab

docs = load_names()
uchars, BOS, vocab_size = build_vocab(docs)

# --- Exercise 1: predict the token list for 'ava' ---------------------------
# a=0, v=21 (a is the 1st letter -> 0; v is the 22nd -> 21), BOS=26.
# Prediction: [26, 0, 21, 0, 26]
tokens = [BOS] + [uchars.index(ch) for ch in 'ava'] + [BOS]
print(f"1. tokens for 'ava': {tokens}")
assert tokens == [26, 0, 21, 0, 26]

# --- Exercise 2: how many names contain each character ----------------------
counts = {ch: sum(1 for d in docs if ch in d) for ch in uchars}
print("\n2. names containing each letter (sorted, rarest first):")
for ch, c in sorted(counts.items(), key=lambda kv: kv[1])[:6]:
    print(f"   {ch}: {c:6d}  ({100*c/len(docs):.2f}% of names)")
rarest = min(counts, key=counts.get)
print(f"   rarest: {rarest!r} — the model gets ~{counts[rarest]} looks at it in the whole")
print("   dataset (and microgpt only trains on 1000 names!), so anything involving")
print(f"   {rarest!r} is learned weakly. Data frequency = learning signal.")

# --- Exercise 3: what if a name contained 'é'? ------------------------------
# uchars.index('é') would raise ValueError: 'é' is not in list.
# It can never happen in microgpt because the vocabulary is built FROM the data:
# uchars = sorted(set(''.join(docs))) — every character that appears in any doc is,
# by construction, in uchars. The tokenizer can only be surprised by characters it
# never saw, and it tokenizes only the text it was built from.
# (Real systems hit this constantly — new text at inference time! That's why real
# tokenizers work on bytes or reserve an <unk> token: there must be SOME id for
# anything a user can type.)
try:
    uchars.index('é')
except ValueError as e:
    print(f"\n3. uchars.index('é') -> ValueError: {e}")
    print("   impossible in microgpt: the vocab is built from the very text it encodes.")
