Presentor: Kavyanjali Agnihotri 

# Main Problem 
Serving models at long context lengths (128K to 1M+ tokens) introduces challenges across memory, compute, and communication.
- why is long context needed
- why and how is this a problem
- how does the future look

## Quick KV Caching recap
The KV cache for a single 1M-token request can consume tens of gigabytes, and the quadratic cost of attention makes prefill latency explode. (what is the prefill latency and why does it explode)

# some proposed solutions
This talk should cover the systems and algorithmic techniques that make million-token inference practical today.

Context parallelism addresses this by sharding the sequence across GPUs
- what does this mean 
- how is the sharding done
- how does it improve / maintain the inference 

Ring Attention forms devices into a logical ring and Striped Attention improves on this by distributing tokens in an interleaved pattern. 
- explain how
- explain why
- overcomes what
- has limitations where
You could discuss how Ring Attention achieves exact results via the online softmax merge: each device maintains running statistics (maximum logit, sum of exponentials, weighted output) and when a new KV block arrives, all previous partial results are rescaled by exp(m_old - m_new) to correct for the shifting maximum. This is mathematically exact but imposes a strict sequential dependency as no query’s output can be finalized until all KV blocks have rotated through the ring.

On the model side, DeepSeek’s Native Sparse Attention (NSA/DSA) is the first sparse attention method deployed in production at scale.
- what is this
- why is it cool
- how

Snowflake’s SwiftKV takes a different approach: it computes later layers’ KV cache from an earlier layer’s output, letting prompt tokens skip much of the model during prefill. 
- what
- why
- how

## interesting (must) add
You could also contrast “lossless” approaches (Ring Attention, context parallelism) with “lossy” ones (StreamingLLM’s attention sinks + sliding window, sparse attention) and discuss where the quality-efficiency frontier lies.



# Problem Explanation

Context length has grown from over 512 tokens (gpt-1, 2018) and 2k/8k tokens (qwen-1) to 128k (gpt-4 series) and over 1 million tokens (llama 4, gemini 3, pro, qwen2.5, etc) 

# Need for Long Context i.e. 128K to 1M tokens+
LLMs started with a fairly smaller token windowns and grew rapidly because of the different possible use cases from summarising docs, code analysis to conversation AI, multimodel uses, agentic workflow etc.


# What is the problem?
KV Cache and prefill latency

Attention = softmax((Q K^T) / sqrt(d)) V

$Q$ (Queries): What a specific word is looking for.

$K$ (Keys): What every other word in the text contains.

$V$ (Values): The actual meaning of those words.

example: "a fox jumped over the fence"


What is the KV cache?
During autoregressive generation, each transformer layer computes key and value vectors from the input. Rather than recomputing these for all previous tokens at every step, they are cached in GPU memory. This turns the per-step complexity from O(n²) to O(n) during decoding — but at the cost of storing all those tensors.

we know that 1 token roughly equals 4 tokens or 0.75 word. 128K token equal to about 96K english words (standard full lenght novel)

KV_cache_per_token is a fixed constant per model architecture and depends on num_layers, num_kv_heads, head_dim, bytes_per_element

Let's take https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, for this model with FP32 data type the KV_per_token = 0.2 MB, then,

Total cache = KV_per_token × sequence_length × batch_size.

so for 1000 tokens, KV cache becomes ~245 MB
then 10k token, KV cache = 2.45 GB
then 128k token, KV Cache = 31.25 GB

Prefill is the initial phase where the model ingests the entire input prompt. It must compute attention between all n input tokens — an O(n²) operation. Nothing can be generated until this completes.

FLOPS??

The explosion is straightforward math:
At 4K tokens: 4K × 4K = 16M attention score computations per head per layer
At 128K tokens: 128K × 128K = 16.4B — a 1,024× increase
At 1M tokens: 1M × 1M = 1T — a 62,500× increase over 4K

This is why a 1M-token prompt takes 30 minutes on hardware that handles 4K tokens in under a second. The prefill phase is compute-bound at long contexts (unlike decoding, which is memory-bandwidth-bound), and the quadratic scaling makes it the dominant bottleneck.


# goal: make long-context inference as cheap as short-context
motivation and ideas: <>
next file for solutions 