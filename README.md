# learn-gpt

A guided, chapter-by-chapter course on how GPTs work, built around
[Andrej Karpathy's `microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
— a complete GPT (training *and* inference) in 200 lines of dependency-free Python:

> *"This file is the complete algorithm. Everything else is just efficiency."*

The course is designed to be taken **inside Claude Code**, with Claude acting as your
tutor. It assumes you're a strong Python developer whose neural-network math is rusty
— every term of art gets a plain-English definition and curated links out (Wikipedia
as a baseline plus the best layman-level video/article we could find).

## Start here

```bash
cp data/names.txt input.txt   # the dataset microgpt.py expects
python3 microgpt.py           # watch a GPT train and generate names (a few minutes)
```

Then open Claude Code in this repo and type:

```
/learn
```

Claude will teach the chapters in order, interactively — explaining in small chunks,
checking your understanding, and pointing you at the labs. Also available:

- `/quiz [N]` — get quizzed on chapter N (or `all`)
- `/explain <term>` — plain-English deep dive on any term (e.g. `/explain logits`)

Prefer self-study? The chapters below are ordinary markdown — read them directly and
run the labs yourself.

## The chapters

Each chapter covers one contiguous section of `microgpt.py`: the code, a line-by-line
walkthrough, a **Terminology** section with links, a runnable lab, and exercises
(solutions in `solutions/`).

| # | Chapter | Covers | Lab |
|---|---------|--------|-----|
| 0 | [Orientation](chapters/ch00-orientation.md) | the whole file, bird's-eye | run `microgpt.py` |
| 1 | [Data & tokenization](chapters/ch01-data-and-tokenization.md) | `microgpt.py:14-27` | `labs/ch01_tokenizer.py` |
| 2 | [Autograd I: the Value class](chapters/ch02-autograd-the-value-class.md) | `:29-57` | `labs/ch02_value.py` |
| 3 | [Autograd II: backward()](chapters/ch03-autograd-backward.md) | `:59-72` | `labs/ch03_backward.py` |
| 4 | [Parameters & initialization](chapters/ch04-parameters-and-initialization.md) | `:74-90` | `labs/ch04_parameters.py` |
| 5 | [Building blocks: linear, softmax, rmsnorm](chapters/ch05-building-blocks.md) | `:92-106` | `labs/ch05_building_blocks.py` |
| 6 | [Attention](chapters/ch06-attention.md) | `:108-134` | `labs/ch06_attention.py` |
| 7 | [MLP, residuals & the forward pass](chapters/ch07-mlp-residuals-forward-pass.md) | `:135-144` | `labs/ch07_forward_trace.py` |
| 8 | [The loss](chapters/ch08-loss.md) | `:160-169` | `labs/ch08_loss.py` |
| 9 | [The training loop](chapters/ch09-training-loop.md) | `:146-184` | `labs/ch09_training.py` |
| 10 | [Inference & sampling](chapters/ch10-inference.md) | `:186-200` | `labs/ch10_sampling.py` |

Plus [GLOSSARY.md](GLOSSARY.md) — every term in one place, A–Z, with links.

## Repo layout

```
microgpt.py        the star of the show — vendored verbatim from the gist, never edited
data/names.txt     the dataset (32K first names, from karpathy/makemore)
chapters/          the course text
labs/              standalone runnable demos (pure Python; shared code in labs/common.py)
solutions/         runnable answers to each chapter's exercises
GLOSSARY.md        master glossary with curated external links
CLAUDE.md          tutor instructions for Claude Code
.claude/commands/  /learn, /quiz, /explain
```

Everything runs on the Python standard library alone — no venv, no pip, nothing to
install. Labs that train do so for ≤300 steps and finish in about a minute.

## What's next, after chapter 10

- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
  — Karpathy's lecture series; start with the
  [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) (this course's ch 2–3,
  built live) and [makemore](https://github.com/karpathy/makemore) (this exact dataset,
  progressively fancier models).
- ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — microgpt's big
  sibling, in PyTorch with tensors and batches.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the same architecture, ready to
  train real models on real data.
- The papers, which will now read like descriptions of code you know:
  ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and
  [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

## Credits

- `microgpt.py` © [Andrej Karpathy](https://karpathy.ai/), from
  [this gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95),
  included verbatim for study purposes.
- `data/names.txt` from [karpathy/makemore](https://github.com/karpathy/makemore)
  (MIT license), originally sourced from ssa.gov.
- Course chapters, labs, and exercises were written for this repo.
