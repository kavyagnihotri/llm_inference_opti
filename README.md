# LLM Inference Optimization
Seminar, Summer 2026, University of Tübingen

## Overview
Large Language Models are quite large. Just to run the trained model and generate text, i.e. to run ‘inference’, one would think it would cost the companies a lot more money to run them in practice than they seem to have. So what do they do?

This seminar introduces the key technologies in machine learning engineering that make the generation of text from large-scale language models economically feasible at scale. Modern LLM serving systems such as vLLM and SGLang combine a range of systems-level optimizations that together enable the deployment of models with up to trillions of parameters. Understanding what happens under the hood not only helps you extend inference engines with new ideas, but also how to best use these engines in practice (and what all of the funny settings mean).

We will go through a list of key topics in this seminar. Each topic is presented by one of the participants at the weekly meetings of the seminar (this could be you!). If you do not want to prepare a seminar talk, you can still take part passively, but will not receive a grade. The topic assignment will be finalized in the first week of class.

## Format
This is a Seminar: each partiicipant selects one of the topics below and prepares a presentation for one of the weekly sessions, followed by questions and discussion. There is no separate written report required - but your presentation should be good! You will have about an hour to really go in-depth into your assigned topic. This may include detailed descriptions of the systems architecture, profiling runs or simulations that you have run, ablations or just a very clear explanation of a crunchy piece of systems engineering. Please don’t present just a vague collection of high-level ideas that you got from prompting GPT-5.4 “what’s going on in PagedAttention”.

- Week 1: Introduction, Basics, Topic Assignment
- Week 2: Office Hour and Q&A - I would suggest starting to read up on your topic a bit, so you can ask questions here.
- Weeks 4–15: Student presentations (one topic per week, excluding holidays).
See the full schedule below.

### Prerequisites
Prior knowledge of machine learning and an understanding of language modeling with transformers are potentially required - but only insofar as this will not be part of the seminar. At a basic level, inference for LLMs is quite learnable, so you can also quickly catch up if you’re willing to do so. Prior ML engineering or advanced systems knowledge is not required and will (hopefully) be obtained as part of the class.

### Grading
The grade is based on the quality of the presentation, including clarity of explanation, depth of understanding, and ability to answer questions.

## Seminar Topics
The following topics are available for presentations. Each topic covers a core technique used in production LLM serving systems. References include both academic papers and open-source implementations. As presenters you are expected to engage with the actual code in systems like [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang), using the academic papers as a basis. The reference suggestions provided for each topic are not exhaustive, feel free to extend them with your own research, but I hope they give some indication of the planned direction (so we make sure not to repeat concepts in different presentations).

### 9. Long-Context Inference
Serving models at long context lengths (128K to 1M+ tokens) introduces challenges across memory, compute, and communication. The KV cache for a single 1M-token request can consume tens of gigabytes, and the quadratic cost of attention makes prefill latency explode. Context parallelism addresses this by sharding the sequence across GPUs: Ring Attention forms devices into a logical ring and Striped Attention improves on this by distributing tokens in an interleaved pattern. On the model side, DeepSeek’s Native Sparse Attention (NSA/DSA) is the first sparse attention method deployed in production at scale. Snowflake’s SwiftKV takes a different approach: it computes later layers’ KV cache from an earlier layer’s output, letting prompt tokens skip much of the model during prefill. This talk should cover the systems and algorithmic techniques that make million-token inference practical today.

You could discuss how Ring Attention achieves exact results via the online softmax merge: each device maintains running statistics (maximum logit, sum of exponentials, weighted output) and when a new KV block arrives, all previous partial results are rescaled by exp(m_old - m_new) to correct for the shifting maximum. This is mathematically exact but imposes a strict sequential dependency as no query’s output can be finalized until all KV blocks have rotated through the ring. You could also contrast “lossless” approaches (Ring Attention, context parallelism) with “lossy” ones (StreamingLLM’s attention sinks + sliding window, sparse attention) and discuss where the quality-efficiency frontier lies.

#### Reference Suggestions:

- Liu et al., [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (ICLR 2024), [code](https://github.com/haoliuhl/ringattention)
- Brandon et al., [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431) (2023)
- Yuan et al., [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) (DeepSeek, 2025), deployed in DeepSeek-V3.2 with vLLM Day 0 support
- Qiao et al., [SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation](https://arxiv.org/abs/2410.03960), [Arctic Inference plugin for vLLM](https://github.com/snowflakedb/ArcticInference)
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](Efficient Streaming Language Models with Attention Sinks) (ICLR 2024)
- [LMSYS Blog: Chunked Pipeline Parallelism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)
- [vLLM context parallel deployment docs](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
- Engineering discussion: [vLLM #22693](https://github.com/vllm-project/vllm/issues/22693) vs [vLLM #34018](https://github.com/vllm-project/vllm/issues/34018)
