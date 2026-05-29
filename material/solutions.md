# Context parallelism
Context Parallelism is about how to parallelize the sequence length across multiple GPUs. It chops up the sequence, dramatically lowering the memory burden per GPU. This is the magic that allows modern models to process entire books or massive codebases in a single prompt without crashing with Out Of Memory (OOM) errors.

By splitting the sequence, each GPU can calculate attention on its local chunk. When you gather and combine these local calculations correctly, the final output is not just an approximation—it is mathematically identical to standard full attention, but you've saved GPU memory by a massive factor!

But there is one major mathematical caveat: Suppose we split our queries, keys, and values across two GPUs: $Q = [Q_1, Q_2]$, $K = [K_1, K_2]$, and $V = [V_1, V_2]$. GPU 1 can easily calculate its local raw scores: $Q_1 K_1^T$. But to get true global attention, it needs to apply the softmax function. The softmax formula requires dividing by the sum of exponentials across the entire sequence dimension. Because softmax is a global competition, you cannot know a token's true rank or percentage without knowing the scores of every other token in the entire sequence. How can GPU 1 calculate a true global softmax when it only holds $K_1$, and doesn't have access to $K_2$ to finish the math?

# Ring Attention with Blockwise Transformers for Near-Infinite Context
Ring Attention with Blockwise Transformers (Ring Attention), which leverages blockwise computation of self-attention and feedforward to distribute long sequences across multiple devices while fully overlapping the communication of key-value blocks with the computation of blockwise attention

self-attention between a query block and a group of key-value blocks can be computed in any order, as long as the statistics of each block are combined correctly for rescaling.

As we compute blockwise attention and feedforward, each host efficiently coordinates by concurrently sending key-value blocks being used for attention computation to the next host while receiving key-value blocks from the preceding host, effectively overlapping transferring of blocks with blockwise computation. 





## Stripped Attention



