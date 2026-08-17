---
description: Plain-English deep dive on any term from the microgpt course
argument-hint: "<term>  (e.g. 'softmax', 'KV cache', 'perplexity')"
---

Explain a term for the microgpt course student (see CLAUDE.md: expert Python dev,
rusty NN math — terminology is the sticking point).

Term: "$ARGUMENTS"

1. If no term was given, ask for one.
2. Check `GLOSSARY.md` for the term (fuzzy-match generously — "attention weights"
   should find Attention) and the chapter it points to; skim that chapter section
   before answering. If the term isn't in the glossary, explain it anyway and note
   it's off the course's beaten path.
3. Structure the explanation:
   - **One sentence**, plain English, no jargon.
   - **An analogy** that fits, with its parts explicitly mapped back to code
     variables/lines in `microgpt.py`.
   - **A tiny worked numeric example** (2–3 element vectors, arithmetic shown) —
     runnable as a few lines of Python where possible.
   - **Where it lives in the code**: exact `microgpt.py:NN` reference, plus the
     matching lab in `labs/` if one demonstrates it.
   - **Links**: the GLOSSARY.md entry's external links (Wikipedia baseline + the
     layman resource), with a word on which to read first and why.
4. End with one check-understanding question about the term, and answer it if the
   student responds.
