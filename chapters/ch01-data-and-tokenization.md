# Chapter 1 — Data & tokenization

## Where we are

The first real code in the file. Two small blocks: load a list of names, and build a
two-way mapping between characters and integers. Everything downstream operates on
integers — the model never sees a letter again after this point.

**Covers:** `microgpt.py:14-27` (and the tokenize call at `microgpt.py:157`)

## The code

```python
# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)   # :14
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back   # :23
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
```

And later, inside the training loop, the actual tokenization of one name
(`microgpt.py:157`):

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

## Walkthrough

**The dataset** (`:15-21`) is 32,033 lowercase first names, one per line, from
Karpathy's earlier *makemore* project. Each name is a "document" — a tiny, complete
piece of text. (This repo vendors the file at `data/names.txt`; copy it to `input.txt`
so the download never runs.) The shuffle just mixes the alphabetically-sorted file so
consecutive training steps see varied names.

**The tokenizer** (`:24-26`) is three lines of pure Python you already understand —
the only new thing is the *vocabulary*:

- `''.join(docs)` smashes all names into one giant string; `set()` finds the unique
  characters; `sorted()` fixes their order. Result: `['a', 'b', ..., 'z']`. So token id
  0 *is* `'a'`, 25 *is* `'z'`, by list position. Encoding is `uchars.index(ch)`,
  decoding is `uchars[i]`. A tokenizer is genuinely just this: a lookup table.
- `BOS = 26` mints one extra id that corresponds to no character. It marks the
  *boundary* of a document.

**Why the boundary token matters** (`:157`): `'emma'` becomes
`[26, 4, 12, 12, 0, 26]` — BOS on both ends. This gives the model two crucial
training signals that raw characters can't:

1. *input* BOS at the front: "a name is starting — predict a plausible **first**
   letter" (with no BOS the model could never learn how names begin);
2. *target* BOS at the end: "after `emma`, predict **the name is over**" (with no
   end-marker, generation would never know when to stop).

One token, both jobs. Larger models do the same with tokens like `<|endoftext|>`.

**The scale of it.** `vocab_size = 27` here. GPT-2 used ~50,257 tokens; modern models
~100K+. Their tokens are *chunks* of words (`"transform"`, `"er"`) learned by an
algorithm called byte-pair encoding — a pure data-compression trick, still just a
lookup table at heart, only with a fancier procedure for choosing what goes in the
table. Character-level is the honest, simple end of the same spectrum.

## Terminology

- **Token / tokenization** — converting text to a sequence of integer ids from a fixed
  vocabulary, and back.
  [Wikipedia](https://en.wikipedia.org/wiki/Large_language_model#Tokenization) ·
  [Karpathy: "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE)
  (deep but excellent — for after this course).
- **Vocabulary (`vocab_size`)** — the set of all token ids the model can read or emit.
  It fixes the size of the embedding table (ch 4) and the output layer (ch 7).
- **BOS (Beginning of Sequence) / special token** — a token that carries structural
  meaning ("document boundary") rather than text.
  [Wikipedia](https://en.wikipedia.org/wiki/Large_language_model#Tokenization)
- **Byte-pair encoding (BPE)** — how real LLMs pick their subword vocabularies.
  [Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- **Document / corpus** — one training text; the collection of all of them.
  Here: one name; 32,033 names. [makemore repo](https://github.com/karpathy/makemore),
  where names.txt comes from.

## Lab

```bash
python3 labs/ch01_tokenizer.py
python3 labs/ch01_tokenizer.py <any-name>    # explore: your own input
```

Watch the encode/decode round trip, then the exact `(input → target)` pairs the model
will train on for `'emma'` — including both BOS roles. Chapter 8's loss is computed
on precisely these pairs. Then pass your own name: one that's not in the dataset
(`karpathy`), and one with a character the vocab can't hold (`émma`) — the second is
exercise 3, live.

## Exercises

1. **Warm-up:** Predict the token list for `'ava'` before running anything. Then verify
   with the lab code.
2. **Vocab archaeology:** Change the lab to print, for each character, how many names
   contain it. Which letter is rarest? (This is why the model will be slow to learn
   about `q` — it barely ever sees one.)
3. **Break it:** What happens at `uchars.index(ch)` if a name contained a character
   that wasn't in the training data — say `'émma'`? Why can this never happen in
   microgpt as written? (Look at how `uchars` is built. Then probe it for real:
   `python3 labs/ch01_tokenizer.py émma`.)

Solutions: `solutions/ch01_solutions.py`

---
Next: [Chapter 2 — Autograd I: the Value class](ch02-autograd-the-value-class.md)
