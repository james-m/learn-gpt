---
description: Quiz the student on a microgpt course chapter
argument-hint: "[chapter number 0-10, or 'all']"
---

Quiz the student on the microgpt course (see CLAUDE.md for who they are: expert
Python dev, rusty NN math, terminology is the sticking point).

Requested scope: "$ARGUMENTS"

1. If no chapter was given, ask which chapter (or infer the most recently discussed
   chapter from the conversation). `all` means a cumulative quiz across everything
   covered so far.
2. Read the chapter file(s) in `chapters/` first so questions match what was
   actually taught.
3. Ask **5 questions, one at a time** — wait for each answer before showing the
   next. Mix these types:
   - terminology ("in one sentence, what's a logit?"),
   - predict-the-output ("what does softmax([0, 0, 0]) return?"),
   - read-the-code ("what breaks if we delete `p.grad = 0` at microgpt.py:182?"),
   - why-questions ("why divide by sqrt(head_dim)?").
   At least one question should show a short snippet from `microgpt.py`.
4. Grade gently: confirm what was right, fix what was wrong with a brief
   re-explanation (Socratic follow-up first if the miss is small), and link the
   relevant GLOSSARY.md term.
5. Finish with a score, a one-line diagnosis of the weakest area, and a pointer to
   the specific chapter section or lab to revisit. If they aced it, say so and
   point at the next chapter.
