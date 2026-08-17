# Chapter 0 — Orientation: the whole thing at 30,000 feet

## Where we are

Before zooming into any single line, this chapter answers three questions: what problem
is `microgpt.py` solving, what are its moving parts, and what does it look like when it
runs? Everything here gets its own deep-dive chapter later — today is just the map.

## What a GPT actually does

Strip away everything and a GPT is a function with a strange but simple contract:

> **Given the tokens so far, output a probability for every possible next token.**

That's it. ChatGPT does it with ~100K possible tokens (word chunks) and hundreds of
billions of parameters; `microgpt.py` does it with 27 possible tokens (the letters
`a`–`z` plus one special marker) and 4,192 parameters. The *algorithm* is the same —
which is Karpathy's whole point:

> *"This file is the complete algorithm. Everything else is just efficiency."*

The model here learns from a list of 32,033 human first names (`emma`, `olivia`, ...)
and, once trained, can babble plausible *new* names that were never in the list. Being
able to invent `kaley` or `jaxon` means it internalized real structure: which letters
follow which, how names begin, how long they run, when they stop. Language modeling at
scale is the same trick applied to all human text.

## The moving parts (and their chapters)

`microgpt.py` is 200 lines with no imports beyond `os`, `math`, and `random`. It reads
top to bottom as five acts:

| Act | Lines | What happens | Chapter |
|-----|-------|--------------|---------|
| Dataset & tokenizer | 14–27 | names ↔ lists of integers | 1 |
| Autograd engine (`Value`) | 29–72 | derivatives computed automatically | 2–3 |
| Parameters | 74–90 | 4,192 random numbers that will become the "knowledge" | 4 |
| The model (`gpt()` + helpers) | 92–144 | tokens in → next-token scores out | 5–7 |
| Training & inference | 146–200 | nudge parameters to fit the data, then generate | 8–10 |

And the loop that ties them together, in pseudo-Python:

```python
for step in range(1000):
    name = next_training_example()            # "emma"
    loss = how_badly_model_predicts(name)     # forward pass  (ch 5-8)
    loss.backward()                           # who's to blame?  (ch 2-3)
    nudge_every_parameter_slightly()          # Adam optimizer  (ch 9)
# then:
generate_new_names()                          # inference  (ch 10)
```

## Run it

From the repo root:

```bash
cp data/names.txt input.txt   # microgpt.py expects input.txt in the cwd
python3 microgpt.py           # takes a few minutes — it's pure Python, on purpose
```

You'll see the loss fall from ~3.3 (random guessing) to ~2.0ish over 1000 steps, then
20 freshly hallucinated names. Keep that output around — several chapters refer back
to it.

## Terminology

- **Language model** — any system that assigns probabilities to sequences of text,
  usually framed as "predict the next piece."
  [Wikipedia](https://en.wikipedia.org/wiki/Language_model) ·
  [3Blue1Brown: "But what is a GPT?"](https://www.youtube.com/watch?v=wjZofJX0v4M) —
  the single best 25 minutes you can spend before this course.
- **GPT (Generatively Pre-trained Transformer)** — a language model with a specific
  architecture (the *transformer*, chapters 6–7) trained by next-token prediction.
  [Wikipedia](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) ·
  [bbycroft.net/llm](https://bbycroft.net/llm) — interactive 3D walkthrough of a GPT
  this small; you can literally see every matrix from this repo in it.
- **Parameter / weight** — one of the model's stored numbers, adjusted during training.
  All of the model's "knowledge" lives in them. `microgpt` has 4,192; GPT-2 had 1.5
  billion. [Wikipedia](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
- **Token** — the unit of text the model reads and writes. Here: single characters.
  In big models: chunks of words. (Chapter 1.)
- **Loss** — a single number measuring "how wrong were the predictions"; training is
  the art of pushing it down. (Chapter 8.)
- **Training vs. inference** — adjusting parameters to fit data, vs. using the frozen
  parameters to generate. (Chapters 9 and 10.)
- **Neural network** — if you want the general refresher before the GPT-specific
  material: [3Blue1Brown's series opener](https://www.youtube.com/watch?v=aircAruvnKk) ·
  [Karpathy's "Deep Dive into LLMs like ChatGPT"](https://www.youtube.com/watch?v=7xTGNNLPyMI)
  for the big-picture, general-audience version.

## Lab

No lab for chapter 0 — the "lab" is running `microgpt.py` itself (above).

## Check yourself

1. The model outputs 27 numbers at every step. What are they, and what must they sum to?
2. Why does inventing a *new* plausible name prove the model learned something, in a way
   that reciting `emma` wouldn't?
3. Roughly where in the 200 lines would you look to change how *creative* the generated
   names are? (Hint: it's a single constant. You'll meet it properly in chapter 10.)

*(Answers to these are woven through the coming chapters — no solutions file for ch0.)*

---
Next: [Chapter 1 — Data & tokenization](ch01-data-and-tokenization.md)
