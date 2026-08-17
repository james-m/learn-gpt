"""Lab 01 — Data & tokenization (chapters/ch01-data-and-tokenization.md)

Run:  python3 labs/ch01_tokenizer.py

What to look for:
  * the vocabulary is just the sorted unique characters of the dataset, plus BOS
  * "tokenizing" a name is nothing more than list-indexing in both directions
  * why the same BOS token can mean both "start" and "stop"
"""
from common import load_names, build_vocab

docs = load_names()
print(f"num docs: {len(docs)}   first few: {docs[:5]}")

uchars, BOS, vocab_size = build_vocab(docs)
print(f"\nuchars ({len(uchars)}): {''.join(uchars)}")
print(f"BOS id: {BOS}   vocab_size: {vocab_size}")

# --- encode: string -> list of ints (microgpt.py:157 does exactly this) ---
name = 'emma'
tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
print(f"\nencode {name!r:8} -> {tokens}")

# --- decode: list of ints -> string ---
decoded = ''.join(uchars[t] for t in tokens if t != BOS)
print(f"decode {tokens} -> {decoded!r}")
assert decoded == name, "round trip must be lossless"

# --- what the model actually trains on: (input, target) pairs ---
# At each position the model sees tokens[:i+1] and must predict tokens[i+1].
print(f"\ntraining pairs for {name!r} (input char -> target char):")
for i in range(len(tokens) - 1):
    inp = 'BOS' if tokens[i] == BOS else uchars[tokens[i]]
    tgt = 'BOS' if tokens[i + 1] == BOS else uchars[tokens[i + 1]]
    print(f"  pos {i}: {inp:>3} -> {tgt:>3}")

print("\nNote the two jobs of BOS: as an *input* at pos 0 it means 'a name is starting,")
print("predict the first letter'; as a *target* at the end it means 'predict that the")
print("name is over'. One special token, two roles.")
