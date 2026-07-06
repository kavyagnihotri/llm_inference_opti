# Presenter's Notes — Long-Context Inference
Kavyanjali Agnihotri · University of Tübingen · LLM Inference Optimization Seminar

Notes are organized slide-by-slide, matching `slides.html`. Each entry has: **Say** (what to add verbally beyond what's on the slide), **Numbers** (values to state out loud / point to), and **Q&A** (anticipated questions + short answers). Transitions are given at the end of each entry in *italics*.

---

## 1. Title Slide
**Say:** One-line framing: models now advertise 1M-token context windows (Llama 4, Gemini 1.5/2.5 Pro, Qwen 2.5). This talk is about what actually has to happen under the hood for that number to be real and usable — not just a marketing spec. Five techniques on the chip bar: two lossless (Ring, Striped), three lossy (NSA, SwiftKV, StreamingLLM).

*Transition: "Let's start with why anyone needs a million tokens in the first place."*

---

## 2. Why 1M Tokens?
**Say:**
- Context length growth timeline: GPT-1 (2018) ~512 tokens → Qwen-1 2K–8K → GPT-4-class 128K → current frontier 1M+ (Llama 4, Gemini 3 Pro, Qwen2.5).
- Three real workloads driving this, one per card: document analysis (whole codebase/contract at once, no chunking seams), agentic loops (tool call traces + scratchpad + history accumulate fast — an agent doing a 50-step task can burn hundreds of thousands of tokens just in transcript), multimodal (a single high-res image is thousands of tokens; a video is way more).
- Scale intuition: 1 token ≈ 0.75 English words. 128K tokens ≈ 96K words ≈ one full novel. Let that land — people don't intuitively know how big 128K is until you say "a novel."
- The point of the whole talk: the problem is not "can the model understand 1M tokens" (that's a training/architecture question), it's "can we *serve* it" — memory and compute at inference time.

**Numbers:** 512 (2018) → 2K/8K → 128K → 1M+. 128K ≈ 96K words ≈ full novel.

**Q&A:**
- *Why not just always use short context + retrieval (RAG)?* Chunking/retrieval introduces seam artifacts — cross-chunk reasoning breaks down, and agentic workflows need the full trace, not a retrieved snippet.

*Transition: "So why is serving that expensive? It comes down to one data structure: the KV cache."*

---

## 3. KV Cache: What It Is and Why It Costs
**Say:**
- Walk the attention equation first: Q = what a token is looking for, K = what every other token offers, V = the actual content. Softmax(QKᵀ/√d)V.
- Explain the KV cache from first principles: during autoregressive decoding, to generate token n+1 you need attention over all n previous tokens. Recomputing K and V for all previous tokens at every single decode step would be O(n²) *per token generated* — insane. Instead you cache K and V once when each token is first processed, so decoding drops to O(n) per step. That's the whole trick — you trade memory for compute.
- Point at the formula: Total KV size = 2 (K and V) × L (layers) × H_kv (KV heads, not query heads — matters a lot with GQA/MQA) × d (head dim) × bytes-per-element × n (sequence length) × b (batch size).
- Concrete anchor: Llama-3.1-8B-Instruct, FP32, works out to 0.25 MB per token. Use the slider live if presenting the interactive version — step through 1K → 1M and watch the number explode.
- Land on the table: 1K tokens = trivial (0.25GB), 128K = 32GB (an entire H100's memory, before you've even loaded model weights or done any generation), 1M = 250GB (needs 4 H100s just for cache, nothing else).

**Numbers:** 0.25 MB/token (Llama-3.1-8B, FP32) · 1K→0.25GB · 10K→2.5GB · 128K→32GB · 1M→250GB.

**Q&A:**
- *Why FP32 and not FP16/INT8?* Slide uses FP32 baseline for clarity; in production almost everyone quantizes KV cache to FP16 or lower (halves/quarters these numbers), but the scaling problem doesn't go away, it just moves the crossover point.
- *Why H_kv and not H (all heads)?* Because GQA/MQA share K/V across multiple query heads specifically to shrink this cache — worth flagging that this is itself an orthogonal optimization to everything else in the talk.

*Transition: "That's the memory side. But there's a second, separate problem: even before decoding starts, you have to process the prompt — that's prefill, and it's compute-bound, not memory-bound."*

---

## 4. Prefill: The Quadratic Wall
**Say:**
- Define prefill precisely: the model has to ingest the entire input prompt before it can generate the first output token. This requires computing attention scores between all n input tokens simultaneously — O(n²).
- Walk the bar chart / table: 4K tokens → 16M attention score computations per head per layer (baseline, ~instant). 128K → 16.4B, a 1,024× increase. 1M → 1T, a 62,500× increase over the 4K baseline. Emphasize this is *quadratic*, not linear — doubling the sequence quadruples the work.
- The real-world punchline: a 1M-token prompt can take 30+ minutes to prefill on hardware that handles 4K tokens in under a second.
- Key distinction to state explicitly (this is a common exam-style question): prefill is *compute-bound* (GPU is busy doing FLOPs, not waiting on memory) — this is different from decoding, which is *memory-bandwidth-bound* (GPU idles waiting to reload the KV cache for each new token). This distinction resurfaces later when explaining NSA's arithmetic intensity argument, so plant the seed here.

**Numbers:** 16M (4K) → 16.4B (128K, 1,024×) → 1T (1M, 62,500×). 30+ min prefill at 1M tokens vs <1s at 4K.

**Q&A:**
- *Doesn't FlashAttention already solve this?* FlashAttention solves the memory-movement part of attention (avoids materializing the full n×n score matrix), giving big constant-factor speedups, but the FLOP count is still O(n²) — the quadratic wall is algorithmic, not just an implementation inefficiency.

*Transition: "So we have two separate problems — KV cache memory, and quadratic prefill compute. Every technique in this talk attacks one or both. They split into two philosophies."*

---

## 5. Two Paths to Efficient Long-Context Inference
**Say:**
- This is the organizing slide for the whole talk — refer back to it mentally at each section break.
- **Lossless** (blue box): don't reduce the amount of attention computed, *distribute* it across devices instead. Output is mathematically/bit-for-bit identical to full attention. Ring Attention, Striped Attention, and context parallelism in vLLM/SGLang live here. Memory savings scale linearly with device count N — this solves the memory problem, not the FLOP problem (you still do all the compute, just spread across more chips).
- **Lossy** (red box): actually reduce how much computation happens per token, by being selective about what gets attended to. NSA (learned sparsity), SwiftKV (skip layers), StreamingLLM (skip most of the past entirely). Trade a small accuracy hit for large efficiency gains (2×–12× depending on method).
- The framing to state explicitly: lossless answers "how do I fit this in memory across many GPUs," lossy answers "how do I do less work in the first place." Production systems use both simultaneously.

**Q&A:**
- *Which one actually matters more in practice?* Depends on the bottleneck: if you're memory-constrained (OOM), lossless/parallelism is necessary regardless of accuracy tradeoffs; if you're throughput/latency constrained with memory to spare, lossy methods buy you speed for a small quality cost.

*Transition into section break: "Let's start with lossless — how do you split a single attention computation across many GPUs and still get an exact answer?"*

---

## 6. [Section Break] Part I — Lossless Approaches
**Say:** Brief pause slide. State the throughline: "Context Parallelism: exact attention distributed across devices."

---

## 7. Context Parallelism: Split Sequence, Share KV
**Say:**
- Context parallelism (CP) is the general strategy; Ring and Striped Attention are specific *algorithms* for implementing it. Don't conflate them — CP is "shard the sequence dimension across GPUs," Ring/Striped are "how do you actually compute correct attention once it's sharded."
- Show the concrete savings card: Llama-3.1-8B at 128K needs 32GB of KV cache. On 1 GPU that's often infeasible once model weights are also loaded. Split across 4 GPUs → 8GB each, comfortable. Across 8 → 4GB each, leaves room for larger batch sizes (more concurrent requests = more throughput).
- Now the hard part, the reason this isn't trivial: the **global softmax problem**. If GPU 1 holds only Q₁ and K₁, computing genuine global attention requires normalizing by the sum of exponentials over the *entire* sequence — but GPU 1 doesn't have K₂, K₃, K₄. Softmax is a "global competition": you can't know a token's relative weight without seeing every other token's score. This is the crux problem that Ring Attention exists to solve.
- The resolution, stated at a high level here (details next slide): attention between a query block and any set of KV blocks can be computed in *any order*, as long as you correctly merge and rescale the running softmax statistics (max, sum-of-exp, weighted output) as new blocks arrive. This order-independence is what makes a "ring" topology possible.

**Q&A:**
- *Why not just gather all K/V to every GPU first?* That defeats the entire point — it would still require O(n) memory per GPU for the full KV cache, exactly what we're trying to avoid.

*Transition: "That merge-in-any-order trick has a name — online softmax — and it's the engine behind Ring Attention."*

---

## 8. Ring Attention
**Say:**
- State the core idea in one sentence: rotate KV blocks around a ring of GPUs while keeping Q blocks stationary, accumulating softmax statistics online so the final result is exact.
- Walk the 4-step algorithm on the card in order: (1) split sequence across N GPUs, each keeps a fixed Q_i; (2) over N−1 rounds, KV blocks rotate — GPU i sends to i+1, receives from i−1, *simultaneously*; (3) each GPU computes blockwise attention on whatever KV block it currently holds, accumulating running softmax stats; (4) after N−1 rounds every GPU has seen every KV block — output is bit-for-bit exact, not approximate.
- Use the ring diagram interactively if live: click "Step" to rotate the KV labels around the ring, showing that Q stays fixed while KV block identity moves. This is the clearest way to convey it non-mathematically before the math slide.
- State the fundamental limitation clearly, because it directly motivates Striped Attention two slides later: this creates a **strict sequential dependency**. No GPU can finalize its output until all N−1 KV blocks have rotated all the way through. You cannot skip rounds. This is the latency cost of exactness.

**Q&A:**
- *Why is it called "blockwise" and not just token-by-token?* Blocks (not individual tokens) are the unit of both compute and communication — this is what allows overlapping transfer with computation (next-next slide) and keeps FlashAttention-style memory-efficient kernels usable within each block.

*Transition: "I said the merge is exact — let's see the actual math, because 'exact despite an incremental global max' is the elegant part."*

---

## 9. Ring Attention: Online Softmax
**Say:**
- This is the mathematical heart of why Ring Attention has zero approximation error. Go slow here — it's the most technically dense slide.
- Standard softmax requires knowing the global max logit m* up front (for numerical stability) and the global sum of exponentials — both are properties of the *entire* row, not a chunk of it. The trick: maintain *running* versions of these statistics and correct them retroactively when new information (a new KV block) arrives.
- Each device tracks three running values as it processes blocks: m (running max logit), s (running sum of exp, denominator), o (running weighted output).
- When a new block arrives with its own local stats (m′, s′, o′) and it turns out m′ is the new global max: you don't get to just add the new numbers in — you have to *rescale* everything computed so far by exp(m_old − m_new), because all your previous exponentials were computed relative to the old, now-stale max. Point at the two update equations: s_new = e^{m−m′}·s + s′, and o_new is the correspondingly rescaled weighted average.
- The payoff: this is provably bit-for-bit identical to computing full softmax over the whole row at once. The only cost paid is the KV block communication between GPUs (bandwidth), not any accuracy.

**Q&A:**
- *Why do we need the max at all — why not just sum exponentials directly?* Numerical stability: raw logits can be large, and exponentiating them directly overflows floating point. Subtracting the running max before exponentiating keeps values in a safe range. This is the standard "safe softmax" trick, just made incremental/distributed here.

*Transition: "Exact math is great, but distributed systems live or die on whether communication becomes the bottleneck. Does rotating KV blocks around the ring actually cost you time?"*

---

## 10. Ring Attention: Communication Overlap and Memory
**Say:**
- Key insight: communication and computation happen *simultaneously*. While GPU i computes attention against the KV block it currently holds, it is *at the same time* sending that same block onward to GPU i+1 and receiving the next block from GPU i−1. If compute time per block ≥ transfer time per block, communication is fully hidden — free, in wall-clock terms.
- Derive the condition live: compute cost per block is 4dc² FLOPs (two matmuls, each (c×d)·(d×c) costing 2dc²) → time = 4dc²/F. Communication cost is 4cd bytes (send + receive K and V) → time = 4cd/B. Setting compute time ≥ transfer time and solving gives the strikingly simple condition: **c ≥ F/B** — block size just needs to exceed the ratio of your hardware's FLOPs to its bandwidth. Since compute scales quadratically in block size c and communication only linearly, this condition gets easier to satisfy as c grows — practically always satisfiable with reasonably large blocks.
- Memory side: each GPU only ever needs 6 blocks in memory at once — 1 stationary Q block, 2 for the current K/V pair being computed on, 2 buffer slots for the incoming K/V pair (so receiving doesn't stall on compute), and 1 output accumulator. Total = 6bch bytes. The critical point: this is **independent of total sequence length s** — memory scales with block size, not sequence length, which is what enables "zero-overhead" scaling to arbitrarily long context.
- Scaling law takeaway: if 1 GPU supports context length s, N GPUs support N·s, at the *same* FLOP utilization. The paper demonstrates this beyond 100M tokens without any approximation to attention.

**Numbers:** Compute time 4dc²/F · transfer time 4cd/B · overlap condition c ≥ F/B · memory = 6 blocks (6bch bytes), independent of s.

*Transition: "Ring Attention is exact and scales, but it has a hidden inefficiency specifically when you add causal masking — which is basically always, for autoregressive LLMs."*

---

## 11. Striped Attention: Fixing Causal Workload Imbalance
**Say:**
- Set up causal attention first if the audience needs the refresher: CausalAttn(Q,K,V) = softmax(QKᵀ + C)V, where C is an upper-triangular mask of −∞ values — future tokens get masked to exactly 0 after softmax, since e^(−∞) = 0. This halves the theoretical compute (only the lower triangle needs computing) versus bidirectional attention.
- The problem: Ring Attention, as described, cannot actually cash in that 2× savings. Why? Contiguous block assignment interacts badly with the causal mask. On certain rounds, one GPU's assigned Q-block/KV-block pair is *entirely* in the future relative to each other (fully masked → instant, free) while another GPU's pair is *entirely* in the past (fully unmasked → full work, no savings). Because Ring Attention synchronizes every round (everyone must finish before KV blocks rotate again), the slowest GPU sets the pace for the whole system. The idle GPUs gain nothing — the theoretical 2× disappears entirely.
- Point at the two figures (from the paper, Figure 4): (a) is Ring Attention at round 2 — you can see some devices are handling entirely grey (masked, skippable) blocks while others handle entirely colored (unmasked, must-compute) blocks. (b) is Striped Attention at the same round — every device has a roughly half-grey, half-colored pattern, i.e., balanced work. Explain the axes: rows = queries (current tokens), columns = keys (context tokens being attended to), color = which GPU is responsible, grey = masked/skippable by causal mask.

**Q&A:**
- *Is this Ring Attention's fault or an inherent property of causal masking?* It's specifically the interaction between causal masking and *contiguous* block assignment. If you assign tokens differently (next slide), the imbalance disappears — the causal structure itself is fine, the partitioning scheme was naive.

*Transition: "The fix turns out to be embarrassingly simple: change how tokens are assigned to devices."*

---

## 12. Striped Attention: Round-Robin Token Assignment
**Say:**
- The fix: instead of GPU 0 getting a contiguous chunk (tokens 0–999, say), assign tokens by position modulo N — GPU 0 gets tokens 0, 4, 8, 12, ...; GPU 1 gets 1, 5, 9, 13, ...; and so on (walk the four cards on screen). Every device now holds tokens spread *uniformly* across the entire sequence rather than a contiguous block.
- Consequence: on every round, each device's causal mask ends up roughly half-filled (half past, half future) instead of being all-one-or-all-the-other. Point at the workload formula: per-device work becomes c(c±1)/2 (roughly half, consistently) instead of the Ring Attention worst case of c² (some devices doing the full block every round).
- Correctness argument (important, this is the "why is this still exact" question): attention is *permutation-equivariant* in the key/value dimension — permuting which physical GPU owns which token, computing attention, then un-permuting the output at the end gives an identical result to not permuting at all. So this is still lossless, just a smarter data layout.
- Results table: 1.45× speedup over Ring Attention on 8×A100 at 256K tokens; 1.65× on 16×TPUv4 at 786K tokens. Note the trend: speedup *grows* with sequence length and with degree of sequence parallelism — this matters more, not less, as context length increases, which is exactly the direction the field is moving.

**Numbers:** 1.45× (8×A100, 256K) · 1.65× (16×TPUv4, 786K).

*Transition into section break: "That's the lossless side — exact answers, distributed compute. Now let's flip to methods that actually reduce the amount of work by being selective, at the cost of a small, controlled accuracy hit."*

---

## 13. [Section Break] Part II — Lossy Approaches
**Say:** "Reduce computation itself rather than distributing it." Recall the compute-bound/memory-bound distinction planted on the prefill slide — it resurfaces immediately in NSA.

---

## 14. Native Sparse Attention (NSA) Overview
**Say:**
- Headline claim to state clearly: NSA (DeepSeek) is the first sparse attention method deployed in *production at scale* (DeepSeek-V3.2), not just a research prototype.
- Contrast with prior sparse attention approaches, two separate failure modes (map directly to the two cards):
  1. **Train/inference mismatch**: most prior sparse methods take a model pretrained with *full* attention and bolt sparsity on only at inference time. The model never learned to operate sparsely, so it carries a "dense" architectural bias — this caps how much sparsity you can apply before quality degrades. NSA is trained sparse *from scratch*, so the model actually learns which tokens matter.
  2. **Hardware inefficiency**: even sparse methods that reduce FLOPs on paper often don't translate to real wall-clock speedups. The reason is **arithmetic intensity** — the ratio of math operations to memory accesses, which determines whether your workload is compute-bound or memory-bound on the actual GPU. If scattered/random-access sparse patterns kill arithmetic intensity, the GPU spends its time waiting on memory fetches, not doing (less) math, so nothing gets faster despite the FLOP reduction.
- Reconnect to slide 4's distinction explicitly here: prefill/training is naturally compute-bound (high intensity, big batches keep GPU processors busy); decoding is memory-bound (low intensity — generating one token still requires reloading the *entire* KV cache from memory). This is why sparse attention has to be *designed around hardware*, not just algorithmically clever.
- The three-branch architecture (this is the core idea, spend real time here):
  1. **Compression (coarse)**: a small learnable MLP compresses blocks of tokens into single summary representations, with overlapping strides so information doesn't fall through block boundary cracks. Gives the model a cheap panoramic view of the whole context, but loses fine detail — a compressed block can say "this paragraph discusses pricing" but not the exact number.
  2. **Block selection (fine)**: to recover detail, select a subset of *uncompressed* blocks to attend to at full resolution. Crucially, selection is *blockwise*, not per-token — GPUs read memory in contiguous chunks, so scattering to individual tokens would kill arithmetic intensity again. Cleverly, the importance scores used to pick which blocks matter are *reused* from the compression branch's own intermediate attention scores — no separate expensive scoring pass needed.
  3. **Sliding window (local)**: a conventional local window, kept in its *own* branch with its own gate. Why not just let branch 2 pick nearby tokens? Because local context is trivially easy to learn, and if local and long-range signals share one path, gradient during training gets dominated by the easy local signal, starving the compression/selection branches of learning signal. Isolating local attention protects the long-range mechanisms' ability to learn at all.

**Q&A:**
- *Why does GQA/MQA specifically matter for block selection?* When KV cache is shared across a group of query heads (GQA), NSA aggregates importance scores across the whole group so all heads select the *same* blocks — otherwise different heads in the group would want different scattered blocks, destroying the contiguous-read property that makes the kernel fast. This is covered in more depth on the kernel slide.

*Transition: "Here's the actual figure from the paper showing all three branches side by side, and how they're merged."*

---

## 15. NSA: Architecture at a Glance
**Say:**
- Walk the figure directly (Figure 2 from the NSA paper): left side shows the three parallel branches processing the same query against preceding K/V; right side visualizes the resulting attention pattern per branch — green = computed, white = skipped. Point out how sparse the selected and sliding branches are compared to full attention, while compression handles everything but cheaply.
- The outputs of all three branches are combined via a **learned gate** — the model itself learns how much to weight coarse-global vs fine-selected vs local-recent for each query, rather than using a fixed heuristic mixture.
- Emphasize the sentence that ties algorithm to hardware: all heads in a GQA group selecting *identical* blocks isn't just a convenient simplification, it's a hardware prerequisite — it's what keeps KV reads contiguous and lets the custom kernel avoid redundant memory transfers (detailed next slide).

*Transition: "So the algorithm is designed to produce hardware-friendly access patterns. Now the kernel that actually exploits them, and the numbers this buys you."*

---

## 16. NSA: Group-Centric Kernel and Results
**Say:**
- Explain why standard FlashAttention can't just be reused as-is for the selection branch: FlashAttention's whole speed advantage comes from streaming *contiguous* KV blocks through SRAM for a contiguous block of queries — this works because in dense attention every query in that block needs the exact same complete KV set. But in NSA's selection branch, different queries at similar time-positions may have selected *different* KV blocks (dynamic, data-dependent selection) — naively this forces scattered, disjoint fetches, which is precisely the arithmetic-intensity killer discussed two slides ago.
- The fix — group-centric loading (Triton kernel): instead of grouping the workload by timestamp (all queries near the same position), group by **GQA head group** (all query heads sharing one KV-cache). Because the algorithm already guaranteed (via per-group score aggregation) that every head within a group selects *identical* blocks, the kernel can load all heads in the group together and fetch one shared set of contiguous KV blocks for the whole group — zero redundant transfers from global memory to SRAM. The outer query-position loop is offloaded to the GPU's grid scheduler to balance work across streaming multiprocessors.
- State this as the central design insight of the whole paper: the algorithmic constraints (blockwise selection + per-group aggregation) aren't just modeling choices, they are *hardware prerequisites* that make the kernel possible. Neither the clever algorithm nor the clever kernel alone would deliver real speedups — they're co-designed.
- Results table, read the numbers out: at 64K context, NSA achieves roughly 2–4× faster training throughput and 11.6× faster decoding versus full attention — and these are measured wall-clock numbers, not theoretical FLOP savings, specifically because the kernel keeps the GPU's memory subsystem from becoming the bottleneck. On quality: NSA matches or exceeds full attention on general benchmarks, achieves *perfect* retrieval accuracy at 64K tokens (a classic long-context stress test), and even exceeds full attention on chain-of-thought math reasoning — because sparsity was learned natively, the model isn't fighting an architecture mismatch the way post-hoc sparse methods do.

**Numbers:** ~2–4× training throughput, 11.6× decoding speedup at 64K vs full attention. Perfect retrieval at 64K.

**Q&A:**
- *Does NSA generalize past 64K, where most of these numbers are reported?* The paper's headline numbers are at 64K; the architectural argument (arithmetic intensity, hierarchical branches) should scale further, but ask this if pressed — it's a fair open question to flag on the closing/open-problems slide.

*Transition: "NSA changes *how* attention itself is computed. SwiftKV takes a completely different angle — it doesn't touch attention at all, it skips entire *layers* during prefill."*

---

## 17. SwiftKV: Later Layers Add Less
**Say:**
- The empirical observation this is built on: as a token's representation passes through successive transformer layers, it changes less and less — by roughly the halfway point of the model's depth, a token's hidden state is already close to what it will look like at the final layer. Redundant computation in the back half of the model.
- Core mechanic: pick a pivot layer l partway through the model. During **prefill only**, once you've computed hidden state h^(l) at the pivot layer, use it directly to *project* the K and V projections needed for *all remaining layers* (l+1 through L) — skip running attention, MLP, and Q/output projections for prompt tokens in those layers entirely. You still get a KV cache for every layer (needed later during decoding), you just compute it cheaply from an earlier layer's output instead of running the full forward pass.
- Critical caveat to state explicitly: this only applies during **prefill**. Decoding (generating each new output token one at a time) still runs the full model, all layers — because generation quality depends on full-depth reasoning per new token, and you don't have a "batch of easy tokens" the way you do during prefill.
- AcrossKV (bonus, optional feature): normally even skipped layers each still need their own stored KV cache entry — SwiftKV's shortcut saves compute but not necessarily memory. AcrossKV additionally shares a *single* computed KV cache across several consecutive skipped layers, which cuts memory footprint too — useful specifically for long-context serving where memory, not just compute, is the constraint.

**Q&A:**
- *Why does this not just degrade quality proportional to how many layers you skip?* Because it's paired with distillation (next slide) that specifically recovers the accuracy lost from this structural shortcut — stated as its own point next.

*Transition: "Skipping layers changes the model's outputs, so how do you recover quality without a full retrain?"*

---

## 18. SwiftKV: Recovery via Lightweight Distillation
**Say:**
- The architectural change (feeding an earlier layer's output into later layers' K/V projections) does hurt accuracy out of the box — it's a structural mismatch the pretrained weights weren't trained for. Fix: knowledge distillation, but a *cheap* version. Freeze the original (unmodified) model as a frozen "teacher." Only fine-tune the Q/K/V projection weights of the *skipped* layers (less than 10% of total model parameters) so that the modified "student" model's output logits match the teacher's. Because so few parameters are touched and training data is under 1B tokens, this completes in hours, not the weeks a full retrain/distillation would take.
- "Zero infrastructure changes" point, worth emphasizing to an audience thinking about deployability: SwiftKV doesn't invent new attention math or need custom sparse kernels — it only changes *which data feeds into which existing projection matrix*. That means it drops straight into already-optimized serving engines like vLLM and SGLang without needing new kernel engineering, unlike NSA which required a bespoke Triton kernel.
- Results table, read key rows: 25–50% reduction in prefill compute across several model families; up to 2× higher aggregate throughput; 60% lower time-per-output-token; quality drop under 2% on standard benchmarks; tested across a huge model size range, 3B to 405B parameters. Best-fit workload: RAG and summarization — long prompt, short output, where prefill dominates total wall-clock time (the opposite profile of, say, long creative generation where decoding dominates).
- Deployment note: this is genuinely shipped — Arctic Inference plugin for vLLM, from Snowflake, in production.

**Numbers:** 25–50% prefill compute reduction · up to 2× throughput · 60% lower TPOT · <2% quality drop · <10% params fine-tuned · <1B tokens · 3B–405B models tested.

*Transition: "SwiftKV shrinks compute per token by skipping layers. The last lossy method takes the most aggressive approach: what if you just never keep most of the past around at all?"*

---

## 19. StreamingLLM: Why Window Attention Fails
**Say:**
- Motivating problem: streaming/multi-turn applications (long-running chat, agent loops) need to run indefinitely, but standard KV cache grows linearly forever — eventually you run out of memory no matter how big your GPU is. Question: can memory stay *constant* without the model falling apart?
- Two "obvious" fixes, both explained on the card, both fail for different reasons:
  - **Window attention** (keep only the most recent N tokens' KV, evict everything older): gives you genuinely constant memory, but once the sequence exceeds the window size once, perplexity explodes and the model effectively breaks. It's not a graceful degradation — it's a collapse.
  - **Sliding window + recomputation** (rebuild KV states for the recent window from scratch at every single generation step): avoids the collapse, keeps good quality, but pays the full quadratic recomputation cost per step — so slow (up to 22× slower, stated later) that it's impractical for real streaming use.
- The actual root cause, which neither naive fix addresses: **attention sinks**. Softmax forces attention scores across all attended tokens to sum to exactly 1. When a query doesn't have a strong genuine match anywhere in context, the model still has to put that probability mass *somewhere* — empirically, it dumps it onto the first few tokens of the sequence, regardless of what those tokens actually mean semantically. Evicting those initial tokens (which window attention does, since they're the oldest) removes this "dump site" and destabilizes softmax — that's *why* window attention specifically breaks, not just "some vague degradation."

**Q&A:**
- *Isn't the recency window the more "important" part conceptually?* Yes for content, but the sinks aren't there for content — they're a mechanical byproduct of how softmax normalizes. That's the whole surprising finding of this paper, and worth stating plainly.

*Transition: "So what exactly are these sink tokens, and why specifically the first few?"*

---

## 20. StreamingLLM: Attention Sinks and the Fix
**Say:**
- Restate and reinforce: transformers assign disproportionately high attention weight to the first handful of sequence tokens *regardless of content*. This isn't a bug or curiosity — it's structurally load-bearing for how softmax works; something has to absorb "leftover" probability mass, and empirically it's always the earliest tokens.
- The key experiment (this is the most convincing evidence to walk through carefully): the authors replace the first 4 tokens of input with a meaningless repeated newline character "\n". The model *still* piles disproportionate attention onto those positions. This rules out a semantic explanation — it's not that early tokens happen to carry special meaning, it is specifically their **absolute position** that matters. This single experiment is why the paper is confident sinks are positional, not content-driven.
- Direct consequence for design: any windowing scheme that evicts the earliest tokens removes the sink and destabilizes the whole attention distribution — this is the precise mechanism behind window attention's collapse from the previous slide.
- The fix, stated plainly: permanently retain the first 4 tokens in the KV cache no matter how long the stream runs, in addition to a normal rolling window of recent tokens. Eviction only ever happens from the rolling window, never from the sink tokens. Memory stays bounded (sinks + window size = constant), and softmax stays stable because the dump site never disappears.

**Q&A:**
- *Why exactly 4 tokens, not 1 or 10?* Empirically found in the paper to be enough — models are typically trained to use *multiple* initial tokens as sinks (not just one), likely because there's no single canonical "start" token forcing convergence onto one dedicated position.

*Transition: "Keeping the sinks fixes stability, but there's one more piece: positions. If sink and window tokens are far apart in the original sequence, won't relative position embeddings still break?"*

---

## 21. StreamingLLM: Solutions, Position Re-Indexing, and Results
**Say:**
- Two complementary solution variants worth distinguishing (left card, "Two Solutions"): **Zero Sink / SoftMax-off-by-one** modifies softmax itself so scores don't have to strictly sum to 1 (adds an implicit "zero" token with e^0=1 in the denominator) — gives partial relief but the model still leans on other initial tokens as sinks too, so it's not a full fix on its own. **Sink Token** — a dedicated, *trainable* token whose entire job is to absorb excess attention — is described as the most effective single intervention, both stabilizing attention and slightly *improving* overall perplexity versus the baseline.
- **Position re-indexing**, the second key mechanism (right card): assign position embeddings based on a token's *slot in the cache*, not its original position in the real sequence. Practically: the model always believes it's looking at a short document that fits comfortably within its original training window, no matter how many tokens have actually streamed past. It never encounters position indices beyond what it was trained on, so it never hits the well-known failure mode of LLMs generalizing poorly to out-of-distribution position embeddings.
- Walk the figure (Figure 4 from the paper): the KV cache is conceptually split into a small fixed attention-sink region (first 4 tokens, permanent) plus a rolling window of recent tokens; positions are indexed relative to cache slot, not absolute sequence position.
- Results, read them clearly since they're the strongest numbers in the lossy section: enables stable, coherent generation up to **4 million+ tokens** with a *fixed-size* KV cache, a **22.2× speedup** versus the sliding-window-with-recomputation baseline, and all of this works with **zero fine-tuning** across four different model families tested (Llama-2, MPT, Falcon, Pythia) — it's a pure inference-time trick, not a training intervention (aside from the optional dedicated sink token variant, which does require light pretraining inclusion).

**Numbers:** 4M+ tokens with fixed cache · 22.2× speedup vs recompute baseline · 0 fine-tuning required · tested on Llama-2/MPT/Falcon/Pythia.

*Transition into section break: "That's all five core techniques. Now, how do production serving systems actually combine these ideas at scale?"*

---

## 22. [Section Break] Part III — Industry Deployment
**Say:** "How production systems combine these ideas." This section grounds the previous two (mostly research/algorithmic) sections in real serving-system engineering trade-offs.

---

## 23. SGLang: Chunked Pipeline Parallelism
**Say:**
- Set the concrete scenario stated on the slide: serving a 100-layer model across 4 physical servers, given a single 100K-token prompt — how do you keep all four servers actually busy simultaneously rather than one at a time?
- **Pipeline Parallelism (PP)** baseline: split *layers* across servers over the slower inter-server network links — e.g., Server 1 holds layers 1–25, Server 2 holds 26–50, etc. Naive version: Server 1 processes the *entire* 100K-token prompt through its layers before handing off — meanwhile Servers 2, 3, 4 sit completely idle waiting. This wasted idle time is called the "pipeline bubble," and it's proportional to how much of the prompt has to fully clear one stage before the next stage has anything to do.
- **Context Parallelism (CP) within a single server**: this is where Ring/Striped Attention actually gets used in practice — inside Server 1, using the very fast intra-server NVLink interconnect, split the *prompt itself* across the 4 local GPUs (GPU A: tokens 1–25,000, GPU B: 25,001–50,000, etc.), computing each layer simultaneously across those GPUs via the ring/rotation mechanism covered earlier.
- **The fix — Chunked PP**: instead of sending the whole 100K-token prompt through Server 1 before handing off, slice the prompt chronologically into small chunks (e.g., 4K tokens). Server 1 processes chunk 1 through its layers, immediately hands that chunk to Server 2, and immediately starts on chunk 2 itself rather than waiting. By the time chunk 1 has propagated to Server 4, all four servers are simultaneously busy on different chunks — like a factory assembly line. This eliminates the pipeline bubble.
- **Hybrid CP × PP** (the actual production pattern): combine both simultaneously — within each server, GPUs use CP to jointly compute each chunk's layers fast over NVLink; across servers, finished chunks flow via Chunked PP over the slower inter-server network. This matches each parallelism strategy to the interconnect speed it's best suited for (CP for the fast intra-server links, PP for the slower inter-server links) — ensuring GPUs neither run out of memory nor sit idle.

**Q&A:**
- *Why not just use CP across all 4 servers instead of PP at all?* CP requires frequent communication between participating devices (KV blocks rotating every round) — doing that over slow inter-server network links instead of NVLink would make communication, not compute, the bottleneck. PP is specifically suited to crossing the slower network boundary because it only needs to hand off activations once per chunk, not every round.

*Transition: "SGLang leans on Chunked PP as the primary long-context strategy. vLLM, interestingly, makes almost the opposite bet."*

---

## 24. vLLM vs SGLang: The Current Tension
**Say:**
- Frame this as a live, unresolved engineering disagreement between two major open-source serving frameworks, not settled fact.
- **vLLM's bet**: Context Parallelism (Ring/Striped Attention) as the primary long-context serving strategy, enabled by default. Worth mentioning **Helix** (NVIDIA, 2025), a proposed evolution: instead of the sequential ring topology, every GPU sends its partial attention output *and* its log-sum-exp normalization factor to every other GPU simultaneously (all-to-all communication) — trading the ring's sequential dependency for parallel communication, at the cost of more total messages. Also flag: vLLM shipped day-0 support for DeepSeek's NSA, showing the two lossy/lossless tracks aren't mutually exclusive in a real stack.
- **SGLang's bet**: Pipeline Parallelism (via Chunked PP) as the primary strategy, treating CP as a secondary, intra-server-only optimization. This maps better onto cluster topologies where inter-server bandwidth is genuinely constrained relative to blazing-fast intra-server NVLink — i.e., it's a bet that most real clusters look more like "fast islands connected by slower links" than "uniformly fast everywhere."
- **The actual convergence point** (this is the takeaway, not "who's right"): both teams explicitly acknowledge the long-run answer is a hybrid CP × PP approach (same conclusion as the SGLang slide) — the live debate is really about which dimension to prioritize *given today's specific hardware interconnect realities* (NVLink cheapens CP; InfiniBand-class inter-server links favor PP). This is a hardware-topology argument, not a purely algorithmic one.

*Transition: "Let's zoom back out and put all five techniques on one frontier."*

---

## 25. The Quality-Efficiency Frontier (Summary)
**Say:**
- Walk the summary table row by row as a recap, explicitly naming the loss/gain tradeoff for each: Ring Attention (no loss, N× memory savings, best for training/exact inference), Striped Attention (no loss, 1.65× over Ring, best for causal LM training specifically), NSA (minimal loss, 9–12×-ish gains at 64K, best for pretraining + long decode), SwiftKV (<2% loss, 2× throughput, best for RAG/summarization), StreamingLLM (moderate loss — it's the most aggressive, genuinely dropping most of the past — but 22× vs recompute, best for infinite streaming specifically, not general-purpose long-context QA).
- Reinforce the framing from slide 5 one last time: lossless methods solve the *memory* problem (fit the sequence somewhere), lossy methods solve the *compute* problem (do less work per token); real production systems layer both together rather than picking one.
- Open problems list, worth spending a sentence on each since this is a good place to invite questions: optimal CP × PP scheduling (directly from the vLLM/SGLang tension); which sparsity patterns NSA-style methods should learn across different task types (retrieval vs reasoning vs generation may want different sparsity shapes); the optimal SwiftKV layer cutoff per model architecture (is there a principled way to pick the pivot layer, or is it purely empirical per-model?); Helix's all-to-all vs Ring Attention's ring topology as fundamentally different communication-pattern tradeoffs; how attention sinks behave under non-standard training regimes (e.g., models not trained with a dedicated sink token from scratch).
- Closing framing: "Has the goal — long context as cheap as short context — been achieved?" Answer given on slide: for most practical workloads, largely yes — a 128K request that used to mean 30+ minutes of prefill and 32GB dedicated to KV cache now runs at practical latency on standard clusters. The 1M-token frontier is still actively being pushed, both in research and in the vLLM/SGLang production tension just discussed.

*Transition: "References for anyone who wants to dig into the primary sources, then happy to take questions."*

---

## 26. References
**Say:** No need to read every citation aloud — just note that all five core papers are linked (Ring Attention ICLR 2024, Striped Attention 2023, NSA/DeepSeek 2025, SwiftKV 2024, StreamingLLM ICLR 2024), plus the industry sources (LMSYS chunked pipeline parallelism blog, vLLM context-parallel docs, and the specific GitHub issue threads #22693 vs #34018 where the Ring-vs-all-to-all CP debate is actually happening in the open, for anyone who wants to read real engineers arguing about this in real time).

---

## 27. Thank You / Questions
**Say:** Open the floor. Likely question areas to be ready for, based on what's *not* on the slides:
- Quantization interaction with KV cache (slide 3 used FP32 for clarity; real deployments use FP16/INT8/FP8 KV cache, which shifts all the GB numbers down but not the scaling shape).
- Whether these five techniques can compose (yes — e.g., NSA + context parallelism, or StreamingLLM's sink+window idea inside a context-parallel shard).
- Training-time vs inference-time applicability: Ring/Striped originated in a training context (near-infinite context *training*) and are reused at inference; NSA is trained-in from pretraining; SwiftKV and StreamingLLM are inference-time-only interventions on already-trained models (with SwiftKV needing a light distillation pass).
