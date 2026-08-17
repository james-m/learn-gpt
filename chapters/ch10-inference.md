# Chapter 10 — Inference: sampling new names

## Where we are

The payoff. Training is over; the 4,192 numbers are frozen. Now the model runs in a
loop, feeding on its own output — and names that exist nowhere in the dataset fall
out. This is the same mechanism by which ChatGPT writes: one token at a time, each
conditioned on everything so far, forever choosing from a probability distribution.

**Covers:** `microgpt.py:186-200`

## The code

```python
# Inference: may the model babble back to us                    # :186
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## Walkthrough

**The autoregressive loop.** Start each name with fresh, empty KV caches (`:190` —
a blank context) and feed BOS: "a name is starting." Then repeat:

1. `gpt()` → 27 logits for the next character (with all past k/v cached, each call
   only processes the newest token — ch 6's cache earning its keep);
2. softmax → probabilities;
3. **roll the dice** (`random.choices` with those weights);
4. feed the winner back in as the next input.

The model eats its own output — that's all "autoregressive" means. Generation ends
when the dice pick BOS ("the name is over" — ch 1's end-marker role, now load-bearing)
or after `block_size` characters, the hard ceiling from ch 4: `wpe` simply has no row
17. When an LLM "runs out of context," this is the wall it hit.

**Why roll dice instead of taking the argmax?** Greedy decoding (always the top
token) is deterministic: every sample would be the *same* name — and at each step a
mediocre-but-safe letter beats an interesting one. Sampling explores the learned
distribution: 20 samples, 20 different names, each plausible in proportion to how
much the model believes in it. (Real chatbots sample too — with refinements like
top-k/top-p that clip the distribution's absurd tail before rolling.)

**Temperature — the one-knob personality control** (`:195`). Divide all logits by
`T` *before* softmax:

- `T < 1` stretches the gaps between logits → softmax sharpens → favorites get
  favored harder. As T→0, sampling becomes argmax: safe, repetitive.
- `T = 1` — the model's honest, unretouched beliefs.
- `T > 1` shrinks the gaps → distribution flattens toward uniform → adventure and
  gibberish in equal measure.

From this repo's actual lab run (same dice each row — every difference is T):

```
T=0.1 : kalen  jaran  aman   amara  jaman  janan  jaran  aman
T=0.5 : vaas   jadiel enicani uanin janlen sanana jadie  avevan
T=1.0 : zaay   kiasga heogcaa zanlia ennlen jaanalise bmayy bealole
T=1.5 : zeay   miatea hindeea zanllayn niif jeanahtka cicyv caajtah
```

The "temperature" slider in LLM APIs and playgrounds is *exactly* this line of code.

**"Hallucinated" is the point here.** Karpathy's word choice in `:188` is a wink:
the model confidently produces names that don't exist — which is precisely what we
*want* from a generative model of names, and precisely the failure mode we *don't*
want when the same mechanism generates, say, legal citations. Same dice, different
stakes: an LLM never "looks up" an answer; it always rolls from its learned
distribution. (Also note the samples aren't memorized — grep `data/names.txt` for
them; most aren't there. With 4,192 parameters and 32K names, the model *couldn't*
memorize the data if it tried — compression forces generalization.)

**And that's the entire file.** Notice what inference *doesn't* use: no loss, no
gradients, no optimizer, no `backward()` — the entire autograd machinery goes idle.
`.data` flows through the same `Value` objects, the graph gets built and thrown away
unused. Forward pass + dice: that's deployment. You have now read, and hopefully
understood, every line of a working GPT.

## Terminology

- **Inference** — running a trained model (vs. training it). What you pay for by the
  token when calling an LLM API.
- **Autoregressive generation** — predict one token from the sequence so far, append
  it, repeat. [Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_model) ·
  [Jay Alammar: "The Illustrated GPT-2"](https://jalammar.github.io/illustrated-gpt2/)
  — walks this loop with big-model pictures.
- **Sampling / decoding strategies** — how to pick from the distribution: greedy
  (argmax), pure sampling (here), beam search, top-k, top-p/nucleus.
  [Nice explainer: Hugging Face on decoding](https://huggingface.co/blog/how-to-generate)
- **Temperature** — logit divisor controlling sharpness of the sampling distribution.
  [Wikipedia](https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning)
- **Greedy decoding / argmax** — always the single most likely token; deterministic.
- **Hallucination** — fluent, confident generation of things that aren't so; the
  flip side of generative sampling.
  [Wikipedia](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
- **Context window** — the `block_size` ceiling on how much past the model can
  condition on.

## Lab

```bash
python3 labs/ch10_sampling.py    # trains ~300 steps first; about a minute
```

Dissects one generation step (the actual top-5 first-letter picks with their
probabilities), then the temperature sweep above, with identical dice per row so
temperature's effect is isolated.

## Exercises

1. **Greedy microgpt:** change the lab's sampling to argmax (`max(range(vocab_size),
   key=lambda i: probs[i].data)`). How many distinct names can greedy produce from a
   BOS start? Run it and confirm.
2. **Seed the name:** write a `complete(prefix)` that feeds BOS + a prefix like
   `'ka'` through the model (filling the KV cache) and then samples the continuation.
   Generate ten completions of `'ka'`. (Everything you need is in `labs/common.py`;
   ~10 lines.)
3. **Novelty audit:** generate 50 names at T=0.5 and check each against
   `data/names.txt`. What fraction is genuinely new? Repeat at T=0.1. Explain the
   direction of the difference.
4. **Capstone — read the file cold:** close everything, open bare `microgpt.py`, and
   read all 200 lines top to bottom. For every line, name the chapter that explained
   it. Anything that still resists you — that's your `/explain` list.

Solutions: `solutions/ch10_solutions.py`

---
You made it. For where to go next (micrograd video, makemore, nanoGPT, the papers),
see the **What's next** section in [README.md](../README.md).
