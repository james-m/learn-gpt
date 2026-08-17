# Glossary

Every term the course leans on, A–Z. Each entry: plain-English definition, the chapter
where it's taught in context (with a worked example), and links out. Wikipedia is the
baseline reference; the second link, where present, is the best layman-level
treatment we know of. In Claude Code, `/explain <term>` expands any of these
interactively.

---

**Activation function / nonlinearity** — the non-linear function between linear
layers (here: ReLU) that stops a stack of matrices from collapsing into one matrix.
Ch 7 · [Wikipedia](https://en.wikipedia.org/wiki/Activation_function)

**Adam** — the standard optimizer: gradient descent plus a momentum average (`m`) and
a per-parameter step-size adaptation (`v`), with bias correction. Seven lines in
microgpt. Ch 9 · [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) ·
[paper](https://arxiv.org/abs/1412.6980)

**Attention / self-attention** — the mechanism letting each position build a weighted
average over earlier positions, with learned, content-dependent weights ("soft
dictionary lookup"). Ch 6 · [Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning)) ·
[3Blue1Brown video](https://www.youtube.com/watch?v=eMlx5fFNoYc)

**Attention head** — one independent attention running on a slice of the dimensions;
several heads = several parallel "questions" per position. Ch 6 ·
[Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

**Autograd (automatic differentiation)** — computing exact derivatives of code by
recording operations as they execute. The `Value` class. Ch 2–3 ·
[Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)

**Autoregressive** — generating one token at a time, each conditioned on all previous
output; the model "eats its own output." Ch 10 ·
[Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_model)

**Backpropagation** — one reverse sweep of the chain rule through the computation
graph, yielding every parameter's gradient at ~2× the cost of the forward pass.
Ch 3 · [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) ·
[3Blue1Brown video](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

**Batch** — how many training examples are averaged into one gradient step. microgpt
uses batch size 1 (pure stochastic). Ch 9

**Bias (neuron)** — the constant `+ b` in the classic neuron `w·x + b`. microgpt (like
several modern GPTs) omits biases entirely; normalization layers cover for them. Ch 5

**BOS (Beginning of Sequence)** — a special token marking document boundaries. As
input: "a name is starting"; as target: "the name is over." Ch 1

**Byte-pair encoding (BPE)** — the compression-style algorithm real LLMs use to build
subword vocabularies (vs. microgpt's characters). Ch 1 ·
[Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding) ·
[Karpathy's tokenizer video](https://www.youtube.com/watch?v=zduSFxRajkE)

**Chain rule** — effects multiply along a dependency chain: if x moves w by 5× and w
moves L by 2×, x moves L by 10×. The single line of math behind all of training.
Ch 3 · [Wikipedia](https://en.wikipedia.org/wiki/Chain_rule)

**Computation graph** — the recorded network of operations linking inputs to a final
value; built by `Value` ops, walked by `backward()`. Ch 2–3 ·
[colah's essay](https://colah.github.io/posts/2015-08-Backprop/)

**Context window (`block_size`)** — the maximum positions the model can attend over
(here 16; why LLMs have context limits — the position table runs out of rows).
Ch 4, 6, 10

**Cross-entropy loss** — `-log(probability assigned to the correct answer)`,
averaged. THE loss for classification and language modeling. Ch 8 ·
[Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy) ·
[StatQuest video](https://www.youtube.com/watch?v=6ArSys5qHAU)

**Derivative** — how much an output moves per tiny nudge of an input; a slope. Ch 2 ·
[Wikipedia](https://en.wikipedia.org/wiki/Derivative) ·
[3Blue1Brown, Essence of Calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk)

**Dot product** — sum of elementwise products of two vectors; doubles as a similarity
score (large when vectors align) — the reading attention is built on. Ch 5–6 ·
[Wikipedia](https://en.wikipedia.org/wiki/Dot_product)

**Embedding** — representing a discrete symbol (token, position) as a learned vector
of floats, so meaning becomes geometry and calculus applies. Ch 4 ·
[Wikipedia](https://en.wikipedia.org/wiki/Word_embedding) ·
[Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)

**Epoch** — one full pass over the training data. microgpt's 1000 steps × 1 name
don't even finish one. Ch 9

**Gradient** — all the partial derivatives of the loss w.r.t. the parameters,
bundled; points in the direction of fastest loss increase (so training steps the
other way). Ch 3, 9 · [Wikipedia](https://en.wikipedia.org/wiki/Gradient)

**Gradient descent** — repeatedly stepping parameters against their gradient:
`p -= lr * p.grad`. Ch 9 · [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent) ·
[3Blue1Brown video](https://www.youtube.com/watch?v=IHZwWFHWa-w)

**Greedy decoding / argmax** — always emitting the single most likely next token;
deterministic (and repetitive). Contrast: sampling. Ch 10

**Hallucination** — a generative model fluently producing things that aren't so —
the flip side of the same dice that produce novel names. Ch 10 ·
[Wikipedia](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))

**Hidden layer / hidden units** — internal vectors that are neither input nor output
(the MLP's 64-wide middle). Ch 7

**Hyperparameter** — a knob the human sets (`n_embd`, learning rate, temperature) as
opposed to a parameter the model learns. Ch 4 ·
[Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))

**Inference** — running a frozen, trained model to get outputs (vs. training). No
gradients, no optimizer. Ch 10

**KV cache** — keeping every past position's key and value vectors so each new token
only computes its own. In microgpt it also enforces causality: the future literally
isn't in the cache yet. Ch 6 · [explainer](https://huggingface.co/blog/not-lain/kv-caching)

**Language model** — a system assigning probabilities to text, framed as next-token
prediction. Ch 0 · [Wikipedia](https://en.wikipedia.org/wiki/Language_model) ·
[3Blue1Brown: "But what is a GPT?"](https://www.youtube.com/watch?v=wjZofJX0v4M)

**LayerNorm** — GPT-2's normalization (subtract mean, divide by std, learned
scale/shift); microgpt uses the simpler RMSNorm. Ch 5 ·
[Wikipedia](https://en.wikipedia.org/wiki/Normalization_(machine_learning))

**Learning rate (and schedule)** — the step-size multiplier on every update, and its
planned trajectory over training (microgpt: linear decay to zero). The most-tuned
hyperparameter in deep learning. Ch 9 ·
[Wikipedia](https://en.wikipedia.org/wiki/Learning_rate)

**Linear layer (fully-connected layer, matrix multiply)** — `nout` weighted sums of an
`nin`-vector; rows as learned "detectors"; >99% of all GPT arithmetic. Ch 5 ·
[Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication)

**lm_head** — the final linear layer mapping the internal vector to one score per
vocabulary token. Ch 7

**Local gradient** — the derivative of a single operation w.r.t. one of its direct
inputs, recorded at forward time (e.g. for `a*b`, they're `b` and `a`). Ch 2

**Logits** — raw, unnormalized scores before softmax. Ch 5, 7 ·
[Wikipedia](https://en.wikipedia.org/wiki/Logit)

**Loss function / objective** — the single number training minimizes; the formal
definition of the task. Ch 8 · [Wikipedia](https://en.wikipedia.org/wiki/Loss_function)

**Maximum likelihood** — the statistics framing of cross-entropy training: choose
parameters that make the observed data most probable. Ch 8 ·
[Wikipedia](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

**MLP (multilayer perceptron) / feed-forward network** — the expand → nonlinearity →
contract block; "compute" to attention's "communicate." Ch 7 ·
[Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron) ·
[3Blue1Brown on facts in MLPs](https://www.youtube.com/watch?v=9-Jl0dxWQs8)

**Momentum (first moment, `m`)** — an exponential moving average of recent gradients;
smooths single-example noise and coasts through flats. Ch 9 ·
[Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) ·
[distill.pub interactive](https://distill.pub/2017/momentum/)

**Multi-head attention** — see Attention head. Ch 6

**Numerical stability** — arranging math to avoid float overflow/underflow (softmax's
max-subtraction; rmsnorm's `1e-5`). Ch 5 ·
[Wikipedia](https://en.wikipedia.org/wiki/Numerical_stability)

**Optimizer** — the rule turning gradients into parameter updates (SGD, Adam, ...).
Ch 9 · [Ruder's survey](https://www.ruder.io/optimizing-gradient-descent/)

**Parameter / weight** — a learned number; the model's entire knowledge is its 4,192
of them. Ch 4 · [Wikipedia](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))

**Perplexity** — `e^loss`: "as confused as choosing evenly among this many options."
Uniform guessing over 27 tokens = perplexity 27. Ch 8 ·
[Wikipedia](https://en.wikipedia.org/wiki/Perplexity)

**Positional embedding (`wpe`)** — a learned vector per position, added to the token
embedding so order is visible. Ch 4, 6

**Pre-norm** — normalizing each block's input (rather than its output), leaving the
residual highway clean; the modern arrangement. Ch 7

**Query / Key / Value** — attention's three learned projections of each position:
the question asked, the label shown, the payload handed over. Ch 6

**ReLU** — `max(0, x)`; a gate that passes positives and zeroes negatives (gradient
included). Ch 2, 7 · [Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

**Residual (skip) connection** — `x + block(x)`; forward, blocks contribute
adjustments to a running signal; backward, `+`'s local grad of 1 gives gradients an
undiminished highway to early layers. Ch 7 ·
[Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network)

**Reverse-mode AD** — differentiating from the output back toward inputs, getting all
parameter gradients in one sweep (vs. one sweep per parameter forward-mode). Ch 3 ·
[Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)

**RMSNorm** — divide a vector by its root-mean-square: standardizes "loudness,"
preserves direction. Llama-era simplification of LayerNorm. Ch 5 ·
[paper](https://arxiv.org/abs/1910.07467)

**Sampling / decoding strategy** — how to pick a token from the model's probability
distribution: pure sampling (microgpt), greedy, top-k, top-p, beam search. Ch 10 ·
[Hugging Face explainer](https://huggingface.co/blog/how-to-generate)

**Scaled dot-product attention** — `softmax(q·k / sqrt(head_dim)) · v`; the scaling
keeps early attention soft so it can learn. Ch 6 ·
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

**SGD (stochastic gradient descent)** — gradient descent where each step's gradient
comes from a random example or small batch: noisy but cheap. Ch 9 ·
[Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

**Softmax** — exponentiate scores, divide by the total: any real vector becomes a
probability distribution; order-preserving, gap-amplifying, shift-invariant. Ch 5 ·
[Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)

**state_dict** — the named dict of parameter matrices (PyTorch's convention); saving
it = saving the model. Ch 4

**Surprise (information content)** — `-log p`: rare events are big surprises;
cross-entropy = average surprise. Ch 8 ·
[Wikipedia](https://en.wikipedia.org/wiki/Information_content)

**Symmetry breaking** — why initialization is random: identically-initialized units
receive identical gradients and never differentiate (`std=0` freezes training
entirely — ch 4's lab). Ch 4 ·
[Wikipedia](https://en.wikipedia.org/wiki/Weight_initialization)

**Temperature** — divide logits by `T` before softmax: `T<1` sharpens (safe,
repetitive), `T>1` flattens (creative, chaotic). The "creativity slider" in every LLM
API. Ch 10

**Token / tokenization** — the model's unit of text and the mapping text ↔ integer
ids. microgpt: single characters. Ch 1 ·
[Wikipedia](https://en.wikipedia.org/wiki/Large_language_model#Tokenization)

**Topological sort** — ordering a dependency graph so nodes come after their
dependencies; lets `backward()` finish each node's gradient before passing it on.
Ch 3 · [Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting)

**Training step / iteration** — one forward → backward → update cycle. Ch 9

**Transformer** — the architecture: embeddings, then repeated (attention + MLP)
blocks with residuals and normalization, then a head. Ch 6–7 ·
[Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) ·
[bbycroft.net/llm](https://bbycroft.net/llm)

**Vanishing / exploding gradients** — gradients shrinking or growing exponentially
through depth; solved jointly by residuals and normalization. Ch 7 ·
[Wikipedia](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

**Vocabulary (`vocab_size`)** — the fixed set of token ids (here 27); sets the size
of `wte` and `lm_head`. Ch 1

**Weight tying** — sharing one matrix between `wte` and `lm_head` (GPT-2 does;
microgpt doesn't). Ch 4
