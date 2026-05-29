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