# Context parallelism / Lossless approach
Context Parallelism is about how to parallelize the sequence length across multiple GPUs. It chops up the sequence, dramatically lowering the memory burden per GPU. This is the magic that allows modern models to process entire books or massive codebases in a single prompt without crashing with Out Of Memory (OOM) errors.

By splitting the sequence, each GPU can calculate attention on its local chunk. When you gather and combine these local calculations correctly, the final output is not just an approximation—it is mathematically identical to standard full attention, but you've saved GPU memory by a massive factor!

But there is one major mathematical caveat: Suppose we split our queries, keys, and values across two GPUs: $Q = [Q_1, Q_2]$, $K = [K_1, K_2]$, and $V = [V_1, V_2]$. GPU 1 can easily calculate its local raw scores: $Q_1 K_1^T$. But to get true global attention, it needs to apply the softmax function. The softmax formula requires dividing by the sum of exponentials across the entire sequence dimension. Because softmax is a global competition, you cannot know a token's true rank or percentage without knowing the scores of every other token in the entire sequence. How can GPU 1 calculate a true global softmax when it only holds $K_1$, and doesn't have access to $K_2$ to finish the math?

## Ring Attention with Blockwise Transformers for Near-Infinite Context
Ring Attention with Blockwise Transformers (Ring Attention), which leverages blockwise computation of self-attention and feedforward to distribute long sequences across multiple devices while fully overlapping the communication of key-value blocks with the computation of blockwise attention

self-attention between a query block and a group of key-value blocks can be computed in any order, as long as the statistics of each block are combined correctly for rescaling.

As we compute blockwise attention and feedforward, each host efficiently coordinates by concurrently sending key-value blocks being used for attention computation to the next host while receiving key-value blocks from the preceding host, effectively overlapping transferring of blocks with blockwise computation. 


The text points out a major flaw in the inner loop (key-value interactions).If we look back at the equation, to calculate the final attention for block 1, $Q_1$ doesn't just need $K_1$ and $V_1$. Because attention requires every word to look at every other word, $Q_1$ must be multiplied by all Keys and Values in the entire sequence:$$\text{Output}_1 \approx \text{softmax}\left( Q_1 [K_1^T, K_2^T, K_3^T, \dots] \right) [V_1, V_2, V_3, \dots]$$Because Host 1 only owns $K_1$ and $V_1$, it has to fetch $K_2, V_2$ from Host 2, and $K_3, V_3$ from Host 3 over the network.

Since each host possesses only one key-value block, the naive approach of fetching blocks from other hosts results in two significant issues. Firstly, it introduces a computation delay as the system waits to receive the necessary key-value blocks.Secondly, the accumulation of key-value blocks leads to increased memory usage, which defeats the purpose of reducing memory cost.

the self-attention between a query block and a group of key-value blocks can be computed in any order, as long as the statistics of each block are combined correctly for rescaling.

Concretely, for any host-i, during the computation of attention between its query block and a key-value block, it concurrently sends key-value blocks to the next host-(i + 1) while receiving key-value blocks from the preceding host-(i − 1). If the computation time exceeds the time required for transferring key-value blocks, this results in no additional communication cost. This overlapping mechanism applies to both forward and backward passes of our approach since the same operations and techniques can be used.

utilizing blockwise parallel transformers to substantially reduce memory costs. As we show in the next section, this enables zero-overhead scaling of context size during both training and inference and allows arbitrarily large context size.

Arithmetic Intensity Between Hosts. In order to determine the minimal required block size to overlap transferring with computation, assume that each host has F FLOPS and that the bandwidthbetween hosts is denoted as B. It’s worth noting that our approach involves interactions only with the immediately previous and next hosts in a circular configuration, thus our analysis applies to both GPU all-to-all topology and TPU torus topology. Let’s consider the variables: block size denoted as c and hidden size as d. When computing blockwise self-attention, we require 2dc2 FLOPs for calculating attention scores using queries and keys, and an additional 2dc2 FLOPs for multiplying these attention scores by values. In total, the computation demands amount to 4dc2 FLOPs. We exclude the projection of queries, keys, and values, as well as blockwise feedforward operations, since they only add compute complexity without any communication costs between hosts. This simplification leads to more stringent condition and does not compromise the validity of our approach. On the communication front, both key and value blocks require a total of 2cd bytes. Thus, the combined communication demand is 4cd bytes. To achieve an overlap between communication and computation, the following condition must hold: 4dc2/F ≥ 4cd/B. This implies that the block size, denoted as c, should be greater than or equal to F/B. Effectively, this means that the block size needs to be larger than the ratio of FLOPs over bandwidth.


simpler explanation: When GPU 1 finishes processing its local chunk, it needs to receive KK K and VV V blocks from GPU 2 to compute the cross-attention. While waiting for that data to arrive over the network, the GPU is idle — wasted time. **The trick**: if we make the blocks big enough, we can compute on the *current* block while *simultaneously* receiving the next one. Computation hides the communication. The question is: how big must each block be for this to work?

The two timers running in parallel: For a block of c tokens with hidden dimension d:
Compute timer — the GPU does two matrix multiplications per block: Q \cdot K^T (scores) and text{scores} \cdot V (output). Each is a multiply of shape (c \times d) \cdot (d \times c), costing 2dc^2 FLOPs. Total: 4dc^2FLOPs. Time = \frac{4dc^2}{F}

Communication timer — you need to receive the next block's K and V, each of size (c \times d). That's 2cd bytes to send,
2cd to receive. Total: 4cd bytes. Time = \frac{4cd}{B}

The overlap condition
For communication to be fully hidden:
\underbrace{\frac{4dc^2}{F}}_{\text{compute time}} \;\geq\; \underbrace{\frac{4cd}{B}}_{\text{comm time}}
c \geq \frac{F}{B}
That's it. The minimum block size is just the ratio of compute speed to network speed.



Memory Requirement. A host needs to store multiple blocks, including one block size to store the current query block, two block sizes for the current key and value blocks, and two block sizes for receiving key and value blocks. Furthermore, storing the output of blockwise attention and feedforward necessitates one block size, as the output retains the shape of the query block. Therefore, a total of six blocks are required, which translates to 6bch bytes of memory. our method exhibits the advantage of linear memory scaling with respect to the block size c, and is independent of the input sequence length s.

Our analysis shows that the model needs to have a sequence length of s = 6c, which is six times the minimal block size. These requirements are easy to meet with parallelism such as data and tensor parallelism and memory efficient blockwise attention and feedforward.

Algorithm 1 Reducing Transformers Memory Cost with Ring Attention.
Required: Input sequence x. Number of hosts Nh.
Initialize
Split input sequence into Nh blocks that each host has one input block.
Compute query, key, and value for its input block on each host.
for Each transformer layer do
for count = 1 to Nh − 1 do
for For each host concurrently. do
Compute memory efficient attention incrementally using local query, key, value blocks.
Send key and value blocks to next host and receive key and value blocks from previous
host.
end for
end for
for For each host concurrently. do
Compute memory efficient feedforward using local attention output.
end for
end for

We utilize FSDP along with Ring Attention to extend the sequence length to nl m and m sequences. This means that the total batch size in tokens remains the same, but Ring Attention enables a significantly larger context size. 

If a model can be trained with context size s on n GPUs using the blockwise attention and feedforward, with our Ring Attention approach, it becomes possible to train a model with a context size of ns.

Ring Attention trains much longer context sizes for self-attention, resulting in higher self-attention FLOPs compared to baseline models. Since self-attention has a lower MFU (model flops utilization) than feedforward, Ring Attention is expected to have a lower MFU than the baseline models. Our method offers a clear advantage in terms of maintaining MFU while enabling training with significantly longer context lengths. when comparing our approach to prior state-of-the-arts, it is evident that we can train very large context models without compromising MFU or throughput.

Through extensive experiments on language modeling and reinforcement learning, we demonstrate its effectiveness, enabling training sequences that are up to device count times longer than those of prior memory-efficient Transformers, exceeding a context length of 100 million without making approximations to attention.


### Stripped Attention
special case of causal transformer models, and identify a key workload imbalance due to triangular structure of causal attention computations. We propose a simple extension to Ring Attention, which we call Striped Attention to fix this imbalance. Instead of devices having contiguous subsequences, each device has a subset of tokens distributed uniformly throughout the sequence, which we demonstrate leads to more even workloads.

<What is casual self-attention?>
1. Causal attention forces the AI to read like a human: it can only look at the current word and the past words, never the future words.Mathematically, it does this using the equation:$$\text{CausalAttn}(Q, K, V) = \text{Softmax}(QK^\top + C)V$$The $C$ is a causal mask—a matrix grid where all the "future" word interactions are replaced with $-\infty$. When you run $-\infty$ through a Softmax function, it turns into exactly $0$. This guarantees the AI completely ignores future words.

2. The Shortcut: 50% Less MathBecause half of the attention grid represents the "future," half of the grid will just end up being zeros. The text points out two obvious shortcuts:Skip the first step: Don't bother calculating the upper-right half of the $QK^\top$ grid because the mask $C$ is just going to overwrite it with $-\infty$ anyway.Skip the second step: Don't bother multiplying those resulting $0$s by the Values ($V$).By skipping the future entirely, you theoretically cut the amount of math (and therefore the computing time) in half.

Because of the mask, the math isn't spread out evenly:

Chip A might get assigned a chunk of the matrix that is entirely "future" words. Thanks to the mask, its math is entirely skipped, and it finishes almost instantly.

Chip B might get a chunk that is entirely "past" words. It has no zeros to skip and has to grind through 100% of the math.

Because the chips are working together in a sequence (a ring), Chip A just ends up sitting idle waiting for Chip B to finish its heavy workload. Even though you are skipping a lot of math overall, the "busy" chips still take the normal amount of time to finish, meaning the total run time of the system doesn't speed up at all.



It is well known that causal self-attention can be computed more cheaply than general bidirectional self-attention; in causal self-attention, each query only interacts with keys which appear earlier than it in the sequence, reducing the total number of operations required by roughly a factor of 2×.

Unlike existing single-device attention kernels, we observe that Ring Attention cannot make effective use of the structure of causal attention to improve its throughput on a per device basis. The reason for this limitation is a workload imbalance: on all but the first iteration of the Ring Attention algorithm, the workload of some devices is entirely necessary (unmasked), while the workload of others is entirely unnecessary (masked) for the final output. The latency of RingAttention is determined by the maximum latency of any participating device per iteration. As a result, regardless of per device optimizations, the latency per iteration would be the same as the time taken to compute a fully unmasked workload. As a result, RingAttention will run as fast as a workload with no attention masking, despite in principle needing to compute only half the operations. To successfully exploit the structure of causal self-attention, we need to change how Ring Attention partitions work among devices. We propose Striped Attention, a variant of Ring Attention which permutes the input sequence in a way which almost entirely eliminates the workload imbalance present in the original Ring Attention algorithm. In particular, we permute the sequence so that every device always operates on a discontiguous subset of tokens distributed approximately uniformly throughout the original sequence. This ensures that approximately half of the query/key interactions on each device will be inhibited by the causal mask. Like Ring Attention, Striped Attention is an exact attention algorithm; we exploit the permutation equivariance of the core attention computation to ensure that our decision to internally permute the input sequence does not affect the final output of the model. Unlike Ring Attention, Striped Attention is able to take advantage of the structure of causal attention to save time on every iteration.

We empirically observe end-to-end speedups of up to 1.45× when training billion-scale causal language models on sequences with hundreds of thousands of tokens using a server with 8 A100 80GB GPUs. We see evern greater speedups on TPUs, achieving 1.65× speedups on problems with over half a million tokens on 16 TPUv4 chips. 

"Ring Attention then computes the output of attention by executing N rounds of computation and communication, keeping the Q blocks stationary on their respective devices while passing the K and V blocks from neighbor to neighbor in a circular fashion. On each iteration, each device computes the interactions between the Q block it owns, and the K and V blocks it is currently holding. Each device owns a stationary block of rows of the output matrix corresponding to the same sequence positions as its Q block. Devices accumulate partial outputs across iterations using the lazy softmax strategy introduced by Rabe and Staats "



Striped Attention reduces the workload imbalance in Ring Attention for causal transformers by partitioning the sequence in a novel way. Rather than partitioning the tokens into contiguous blocks, we partition them into sets of evenly spaced stripes based on their residues modulo the device count N. For example, given a device count of 4, device 0 would own tokens 0, 4, 8, . . ., device 1 would own tokens 1, 5, 9, . . ., and so on. Inpractice, we can achieve this partitioning scheme by permuting the input tokens before the model’s first embedding layer, and then partitioning the permuted sequence into contiguous blocks as in Ring Attention. 

After partitioning, our algorithm proceeds almost identically to Ring Attention. in contrast to Ring Attention, Striped Attention ensures that every device’s causal mask is upper-triangular on every iteration.

<figure 4 from paper probably> Figure 4. Workload distribution on rounds 0 and 2 of Ring Attention and Striped Attention. The square matrices represent the set of all possible pairwise query/key interactions; row indices correspond to queries, and column indices correspond to keys. All cells above the diagonal are masked out by the causal mask and can be skipped. The colors indicate which devices are responsible for which parts of the computation. On round 2, we can see that some devices in Ring Attention are responsible for workloads which are entirely masked out, whereas Striped Attention maintains a balanced workload across all devices.

Ring Attention: At least one device gets stuck doing a full block of math, resulting in a maximum workload of $c^2$ operations (where $c$ is the block size). Striped Attention: Every device has roughly half of its work masked out, meaning its workload drops to approximately: $$\text{Work}(i, j) = \frac{c(c \pm 1)}{2}$$

By dealing the data out in stripes instead of solid blocks, every computer chip gets an exactly equal mix of "past" words (which require math) and "future" words (which are skipped). No chip sits idle, no chip gets overworked, and the entire system **can run up to** 2x faster.

The most clear trend is that the benefit of our method scales with sequence length in multiple ways. First, given a particular mesh configuration, the speedup increases with the block size. As the number of tiles per block increases, a greater percentage of computation is skipped, and this leads to real speedups. Second, given a particular block size, increased sequence parallelism leads to greater speedup as long as the block has at least two tiles in the query and key dimension. This is most clear on the A100 plot. At a given block size, all models with similar degrees of sequence parallelism have similar speedups regardless of the degree of model parallelism. However, models with high degrees of sequence parallelism have greater speedups.

In experiments running Striped Attention on A100 GPUs and TPUv4s, we are able to achieve up to 1.45× end-to-end throughput improvements over the original Ring Attention algorithm on causal transformer training at a sequence length of 256k. Furthermore, on 16 TPUv4 chips, we were able to achieve 1.65× speedups at sequence lengths of 786k.


#### Summary?
**Ring Attention** solves the memory bottleneck of long-sequence attention by splitting the sequence across $N$ GPUs arranged in a ring. Each GPU keeps its query block $Q_i$ stationary and passes key-value blocks around the ring over $N{-}1$ rounds. In each round, a GPU computes blockwise attention against the current $K$, $V$ block it holds while simultaneously sending that block to the next GPU and receiving a new one from the previous GPU. Because self-attention between a query block and any group of key-value blocks can be computed in any order — as long as the softmax statistics (running max and sum of exponentials) are combined correctly for rescaling — the final output is mathematically identical to full attention. The communication is fully hidden when compute time exceeds transfer time: $4dc^2/F \geq 4cd/B$, which simplifies to $c \geq F/B$. Since compute scales quadratically with block size $c$ and communication scales linearly, this condition is easy to satisfy. Each GPU only ever stores 6 blocks (query, current and incoming $K$/$V$ pairs, output accumulator), so memory is $O(c)$ and independent of total sequence length $s$.

**Striped Attention** addresses a specific inefficiency when Ring Attention is used with causal (autoregressive) models. In causal attention, each token only attends to earlier tokens, theoretically halving the work. But Ring Attention assigns contiguous blocks to devices, so during certain rounds some devices get entirely masked (all-future) interactions and finish instantly, while others get entirely unmasked (all-past) interactions and do full work. Since devices synchronize each round, the slowest device sets the pace and the 2× savings is lost. Striped Attention fixes this by assigning tokens in round-robin order — device $k$ gets tokens $k, N{+}k, 2N{+}k, \dots$ — so every device holds tokens uniformly distributed across the sequence. This guarantees each device's causal mask is roughly half-filled on every round, giving a consistent per-device workload of approximately $c(c \pm 1)/2$ instead of the worst-case $c^2$. The result is exact (attention is permutation-equivariant), with empirical speedups of up to 1.45× on 8 A100 GPUs at 256k tokens and 1.65× on 16 TPUv4 chips at 786k tokens.



# Lossy Approaches
Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities.


## Deepseek's Native Sparse Attention
We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimization~s for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. Experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle.

To achieve more effective and efficient sparse attention, we present NSA, a Natively trainable Sparse Attention architecture that integrates hierarchical token modeling. NSA reduces per-query computation by organizing keys and values into temporal blocks and processing them through three attention paths: compressed coarse-grained tokens, selectively retained fine-grained tokens, and sliding windows for local contextual information. Then we implement specialized kernels to maximize its practical efficiency. NSA introduces two core innovations corresponding to the key requirements above: (1) Hardware-aligned system: Optimize blockwise sparse attention for Tensor Core utilization and memory access, ensuring balanced arithmetic intensity. (2) Training-aware design: Enable stable end-to-end training through efficient algorithms and backward operators. This optimization enables NSA to support both efficient deployment and end-to-end training.


Modern sparse attention methods have made significant strides in reducing the theoretical computational complexity of transformer models. However, most approaches predominantly apply sparsity during inference while retaining a pretrained Full Attention backbone, potentially introducing architectural bias that limits their ability to fully exploit sparse attention’s advantages. many existing sparse attention methods focus on KV-cache reduction or theoretical computation reduction, but struggle to achieve significant latency reduction in advanced frameworks or backends. Non-Trainable Components and Inefficient Back-propagation

Arithmetic Intensity—the ratio of mathematical operations to memory accesses—determines whether a task is compute-bound (limited by GPU processing speed) or memory-bound (limited by data transfer speeds). If an algorithm's intensity falls below a GPU's specific hardware threshold, it gets bottlenecked by memory bandwidth. In causal self-attention, the training and prefilling phases process large batches of data simultaneously, resulting in high arithmetic intensity that maxes out the GPU's compute capabilities. Conversely, auto-regressive decoding generates only one token at a time but must continuously reload the entire key-value (KV) cache; this drastically lowers the arithmetic intensity, making the generation phase fundamentally memory-bound.

<figure 2 from paper https://arxiv.org/pdf/2502.11089> Figure 2 | Overview of NSA’s architecture. Left: The framework processes input sequences through three parallel attention branches: For a given query, preceding keys and values are processed into compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context. Right: Visualization of different attention patterns produced by each branch. Green areas indicate regions where attention scores need to be computed, while white areas represent regions that can be skipped.

Algorithm Design
Here is a summary of the algorithm design, followed by an explanation of why the authors architected it this way.

**1. Token Compression for Broad Context**
The authors introduce three distinct strategies to efficiently manage how the model pays attention to keys and values: Token Compression, Token Selection, and a Sliding Window. First, **Token Compression** reduces the overall computational load by aggregating sequential blocks of tokens into single, compressed representations using a learnable neural network (MLP). By overlapping these blocks slightly (using a stride smaller than the block length), the model captures broad, high-level semantic information across the entire sequence without suffering from severe information fragmentation.

**2. Blockwise Token Selection for Fine Details**
Because compression inevitably loses fine-grained, specific details, the model pairs it with a **Token Selection** mechanism to preserve crucial exact tokens. However, instead of picking individual, scattered tokens from the sequence, the algorithm selects entire continuous *blocks* of tokens. This blockwise approach leverages the natural tendency of attention scores to cluster (neighboring words usually share similar importance) and avoids the scattered memory access that typically slows down hardware.

**3. Efficient Importance Scoring**
To figure out which blocks to select without adding massive computational overhead, the algorithm cleverly reuses data. It takes the intermediate attention scores generated during the *compression* step and maps them to evaluate the importance of the uncompressed selection blocks. For models that share key-value caches across multiple attention heads (like GQA or MQA), these scores are aggregated to ensure the entire group selects the exact same blocks. Ultimately, the algorithm simply retains the top-$n$ highest-scoring blocks for detailed processing.

**4. Sliding Window for Local Stability**
Finally, a dedicated **Sliding Window** mechanism is introduced to exclusively process the most recent, local tokens. Because AI models tend to adapt to immediate local context very quickly, combining local and long-range data into a single attention calculation often results in "shortcut learning," where the model simply ignores the complex compression and selection features. By isolating local tokens into their own independent branch and later merging the results via a gating mechanism, the model stably learns both short-term and long-term patterns without interference.

The authors designed Native Sparse Attention (NSA) this way to solve the critical tension between **theoretical memory efficiency**, **actual GPU hardware limitations**, and **stable model training**.

If they only compressed tokens, the AI would lose the precise details necessary for high accuracy. If they tried to solve that by plucking out exact, individual tokens, GPU memory fetching would slow to a crawl, ruining the speed benefits of FlashAttention (which requires continuous block reads). By forcing the selection process to be *blockwise*, they maintain hardware speed. Furthermore, by using the cheap compression step to generate the "importance scores" for the selection step, they avoid paying the high computational tax of evaluating the whole sequence twice. Finally, isolating the local tokens into a sliding window ensures that the easily learned local patterns don't drown out the gradient signals needed to train the more complex, long-range compression and selection networks.


**The Kernel Design**
To achieve FlashAttention-level speeds during training and prefilling, the authors built custom, hardware-aligned sparse attention kernels using Triton, specifically optimized for architectures with shared key-value (KV) caches like Grouped Query Attention (GQA). Standard FlashAttention kernels load continuous blocks of *queries* into the GPU's ultra-fast SRAM, but doing this for sparse attention would force the GPU to fetch scattered, disjoint KV blocks for each query. To fix this, the new kernel flips the data-loading strategy: it uses a "Group-Centric" approach. For any given position in the sequence, the kernel loads all the query heads within a GQA group simultaneously, sequentially fetches the continuous KV blocks that those specific heads share, and offloads the query loops to the GPU's grid scheduler to balance the computational workload.

**The Rationale**
This custom architecture was designed specifically to maximize "arithmetic intensity" (the ratio of math operations to memory fetches) and prevent the GPU from becoming memory-bound. If the authors had stuck to standard FlashAttention, the GPU would waste massive amounts of time fetching fragmented KV data because different queries within a time-block might need completely different selected tokens. By grouping the queries by their GQA heads instead of their timestamp, the kernel ensures that all currently loaded queries require the exact same KV blocks. This entirely eliminates redundant data transfers from the GPU's slow global memory to its fast SRAM, keeping the compute cores constantly fed with data rather than sitting idle waiting for scattered memory reads.

**The Integration**
This kernel design is the final puzzle piece that makes the previously discussed algorithm physically viable on hardware. While the compression and sliding window branches use continuous data that already works fine with standard FlashAttention, the dynamic "Token Selection" branch threatened to ruin hardware efficiency. However, because the algorithm was specifically designed to select *continuous blocks* rather than individual tokens, and because it aggregates importance scores so that all heads in a GQA group *agree* on the same selected blocks, this custom kernel can fetch those exact blocks in large, contiguous chunks. The algorithm guarantees the data is grouped perfectly, and this kernel exploits that grouping to process sparse attention without sacrificing hardware speed.

In summary, Native Sparse Attention (NSA) delivers a massive upgrade in both model performance and hardware efficiency. By intelligently filtering out irrelevant noise, NSA **outperforms Full Attention on general reasoning tasks** and achieves **perfect retrieval accuracy** in massive 64k-token documents. Furthermore, because its sparse patterns are learned natively during pretraining, it natively excels at complex **chain-of-thought mathematical reasoning** without losing context over long sequences. Beyond raw accuracy, NSA's hardware-aligned, blockwise design yields extraordinary computational speedups: it achieves up to **9x faster training** and **11.6x faster decoding** at 64k contexts by drastically slashing the amount of memory the GPU is forced to fetch and load.



## SwiftKV
We present SwiftKV, a novel model transformation and distillation procedure targeted at reducing the prefill compute (in FLOPs) of prompt tokens while preserving high generation quality. First, SwiftKV prefills later layers’ KV cache using an earlier layer’s output, allowing prompt tokens to skip those later layers. Second, SwiftKV employs a lightweight knowledge-preserving distillation procedure that can adapt existing LLMs with minimal accuracy impact. Third, SwiftKV can naturally incorporate KV cache compression to improve inference performance in low-memory scenarios. Our comprehensive experiments show that SwiftKV can effectively reduce prefill computation by 25–50% across several LLM families while incurring minimum quality degradation. In the end-to-end inference serving, SwiftKV realizes up to 2× higher aggregate throughput and 60% lower time per output token. It can achieve a staggering 560 TFlops/GPU of normalized inference throughput, which translates to 16K tokens/s for Llama-3.1-70B.


