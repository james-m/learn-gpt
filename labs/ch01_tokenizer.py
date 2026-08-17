"""Lab 01 — Data & tokenization (chapters/ch01-data-and-tokenization.md)

Run:      python3 labs/ch01_tokenizer.py
Explore:  python3 labs/ch01_tokenizer.py olivia     # tokenize any name you like
          python3 labs/ch01_tokenizer.py karpathy   # ...even one not in the dataset
          python3 labs/ch01_tokenizer.py émma       # ...even one the vocab can't hold
          (fun fact: 'zzyzx' and 'claude' ARE in the dataset — try them)

What to look for:
  * the vocabulary is just the sorted unique characters of the dataset, plus BOS
  * "tokenizing" a name is nothing more than list-indexing in both directions
  * why the same BOS token can mean both "start" and "stop"
  * (with a custom name) what the tokenizer does and doesn't care about
"""
import argparse
from common import load_names, build_vocab

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('name', nargs='?', default='emma', help="name to tokenize (default: emma)")
args = parser.parse_args()

docs = load_names()
print(f"num docs: {len(docs)}   first few: {docs[:5]}")

uchars, BOS, vocab_size = build_vocab(docs)
print(f"\nuchars ({len(uchars)}): {''.join(uchars)}")
print(f"BOS id: {BOS}   vocab_size: {vocab_size}")

name = args.name

# --- can the vocabulary even hold this name? --------------------------------
# This is chapter 1, exercise 3, live: uchars.index(ch) has no answer for a
# character the dataset never contained.
missing = sorted(set(ch for ch in name if ch not in uchars))
if missing:
    print(f"\n{name!r} contains characters outside the vocabulary: {missing}")
    try:
        [uchars.index(ch) for ch in name]
    except ValueError as e:
        print(f"uchars.index(...) -> ValueError: {e}")
    print("\nmicrogpt itself can never hit this: its vocab is built FROM the training")
    print("text, so it only ever tokenizes characters it has seen. Real tokenizers")
    print("must handle arbitrary user input, so they work on bytes or keep an <unk>")
    print("token — there has to be SOME id for anything you can type.")
    raise SystemExit(0)

if name not in set(docs):
    print(f"\nnote: {name!r} is NOT in the dataset — but every character of it is, so")
    print("it tokenizes just fine. The tokenizer knows characters, not names; 'is this")
    print("a real name' is the MODEL's problem (it answers with probabilities, ch 8).")

# --- encode: string -> list of ints (microgpt.py:157 does exactly this) ---
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
