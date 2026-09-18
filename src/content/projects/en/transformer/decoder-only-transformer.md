---

title: "Dissecting the Decoder-Only Transformer from a Kernel Perspective — Training, Prefill, Decode, and KV Cache"

description: "An analysis of the structure and operations of a Decoder-only Transformer, with a focus on the GPU operations and kernels that execute them."

pubDatetime: 2026-09-18

order: 1

translationKey: decoder_only_transformer_overview

tags:

  - transformer

  - decoder-only

  - GPU kernel

  - KVCache

  - decode

  - prefill

draft: false

---

In this post, I analyze the structure and operations of a pre-norm Decoder-only Transformer from a kernel perspective and explain how this structure is used during training and inference. In the next post, I will break down the kernel runtime of the prefill and decode stages to identify the bottlenecks in each stage.

# Decoder-only Transformer (Pre-norm)

Let us begin with the overall structure of a Decoder-only Transformer. Decoder-only Transformers can use either post-norm or pre-norm; this post focuses on the pre-norm architecture. The structure is simpler than it may first appear. Here, a token can be thought of as a unit of meaning that a Transformer can understand.

![Decoder-only Transformer](/image/transformer/01_overview.png)

# Token Embedding and Positional Embedding

First, we need to convert human language into a representation that a machine can understand. This process is called embedding. As mentioned above, a token is a unit of meaning understood by the Transformer. A Transformer has its own vocabulary: the mapping between tokens and IDs is the vocabulary, while the mapping from each ID to a $d_{\text{model}}$-dimensional vector is stored in the embedding table.

Suppose four token IDs are provided as input. The Transformer looks up each ID in its embedding table and retrieves the vector associated with that ID. You can think of it as: “What was the ID for *apple* again? ID 1. Then which vector corresponds to ID 1?” Although this operation can be written as matrix multiplication, the actual kernel performs an indexed lookup into an array. The four input tokens are therefore converted into four vectors.

![Token embedding](/image/transformer/02_token_embedding.png)

Now compare the sentences “The dog chased the cat” and “The cat chased the dog.” They contain the same words, but their meanings differ because the word order is different. The Transformer therefore needs information about each token's position. This is the role of positional embedding.

There are several ways to implement positional embeddings. Here, I use the simplest one: a table stores a vector for each position, and the model retrieves the vector corresponding to a token's position just as it retrieves a token embedding by ID. One drawback is that the maximum sequence length must be fixed in advance because the available positions are predefined. Other methods, such as RoPE, avoid this limitation and will be covered in a later post. Adding the token embedding and positional embedding gives us a vector representation for each token.

![Positional embedding](/image/transformer/03_positional_embedding.png)

# LayerNorm

At this point, another problem arises. We have mapped each word to numerical values, but their magnitudes can differ substantially. Large differences in scale can make later comparisons and computations unstable. Our goal is to obtain useful vector representations of the words, not to let their element values vary without control. LayerNorm therefore normalizes the elements of each token vector to have mean 0 and standard deviation 1. It then applies learned $\gamma$ and $\beta$ parameters to scale and shift the result. Normalization controls the scale, while the learned parameters allow the model to choose a useful distribution. LayerNorm is applied not just once, but at the beginning of every layer in a pre-norm Transformer.

![LayerNorm](/image/transformer/04_layernorm.png)

Computing this normalization requires the mean and standard deviation of the original values. A reduction is needed to aggregate the elements in each row representing a token, followed by element-wise normalization. Because the loaded data is not reused, LayerNorm has low arithmetic intensity and is generally sensitive to memory bandwidth.

# What Is Self-Attention?

Consider the sentence: “I like acorn jelly. It is delicious.” What does “it” refer to? It refers to “acorn jelly.” To reach that conclusion, you probably looked at the other words in the sentence. But did you consider every word equally? Is “I” as important as “acorn jelly” when determining what “it” means? Try replacing “I” with another name—the referent of “it” does not change.

What we want, then, is for each word to consult the other words in proportion to their importance and use that information to determine its meaning in context. A word carries more than its dictionary definition; it acquires additional meaning from the surrounding context. By expressing “importance” as different weights, we can combine information from many words at once using matrix multiplication.

That is what attention does: it retrieves and mixes information from other tokens according to how relevant each token is.

## Query, Key, and Value Projections

Attention has four sets of learned weights. Three of them produce the Query, Key, and Value. A Query asks, “Do you have the information I am looking for?” A Key is an index describing what information each token has, and a Value is the actual content associated with that token. A library search provides a useful analogy: the search request is the Query, the catalog entry representing each book is the Key, and the book's contents are the Value.

We now derive Query, Key, and Value from every token. This is done with GEMM, or general matrix multiplication. A GEMM has the form $(M \times K) \times (K \times N) \rightarrow (M \times N)$. In a QKV projection, $M$ is the number of tokens processed at once, $K$ is $d_{\text{model}}$, and $N$ is the output dimension.

For now, consider a single attention head with a batch size of 1.

Let us start with the Query. In the following figure, the input matrix $X$ contains four tokens, each represented by an eight-dimensional vector. Multiplying $X$ by the Query weight matrix $W_Q$ produces the Query matrix.

![Query projection with dimensions](/image/transformer/05_query_projection_with_size.png)

The Key and Value are obtained in exactly the same way, using $W_K$ and $W_V$ instead. We now have the yellow Query, green Key, and blue Value matrices.

The figure shows $W_Q$, $W_K$, and $W_V$ separately, but nanoGPT declares a single layer: `c_attn = nn.Linear(n_embd, 3 * n_embd)`. In other words, one $(M \times 768) \times (768 \times 2304)$ GEMM kernel generates Q, K, and V together. The subsequent `split` only creates tensor views, so it does not launch another kernel.

![QKV projection](/image/transformer/06_projection.png)

## Attention Scores and Logits

Now let each token ask a question and receive responses from the others. The Query asks the question, while each Key indicates how well its token matches that question. But how can we measure the match? We want relevant tokens to receive larger weights and irrelevant tokens to receive smaller ones before their information is mixed. For this, we use a dot product. The more closely two vectors point in the same direction, the larger their dot product tends to be. Let the resulting matrix be $S$.

![Meaning of the attention-score matrix S](/image/transformer/07_matrixS_with_meaning.png)

Matrix $S$ now contains larger values for better Query–Key matches. However, the variance of the dot product grows in proportion to $d_{\text{head}}$, which can be fairly large. The concept of a head will be explained in the next section. With one head, $d_{\text{head}} = d_{\text{model}}$; in general, $d_{\text{head}} = d_{\text{model}} / h$, where $h$ is the number of heads. We therefore divide the scores by $\sqrt{d_{\text{head}}}$. Softmax then converts each row into a probability distribution whose values sum to 1. Let the resulting matrix be $P$.

![Softmax over attention scores](/image/transformer/08_softmax.png)

Each element of $P$ represents how much attention one Query assigns to one token. Because each row sums to 1, these values can be interpreted as probabilities. We now multiply $P$ by the Value matrix. Each token thereby receives a weighted mixture of the other tokens' information in response to its Query. The result is called a context vector.

![Context vector](/image/transformer/09_context_vector.png)

Finally, when multiple heads are used, their context vectors must exchange information. Although this example still uses one head, the next section will make the role of multiple heads clear. Multiplying the context vectors by $W_O$ mixes information across heads and produces the final output matrix $O$.

![Output projection](/image/transformer/10_output_projection.png)

## Multi-head Attention

Let us now examine multi-head attention. The purpose of attention is to ask each word, “Do you contain the kind of information I am looking for?” and then retrieve more information from the tokens that match well. Multi-head attention extends this idea by asking several kinds of questions at the same time—several different “perspectives”—while keeping the total amount of computation almost unchanged.

The first step of attention is the projection that produces Query, Key, and Value. We divide the corresponding weight matrices into several parts—one per head. For simplicity, the following example uses two heads. In the figure, the yellow $W_Q$ matrix is split into two parts, producing two portions of the Query matrix. The same procedure is applied to the Key and Value.

![Query projection in multi-head attention](/image/transformer/11_query_projection_in_multihead_attention.png)

![QKV projections in multi-head attention](/image/transformer/12_projection_in_multihead_attention.png)

We then multiply each head's Query by its Key to obtain a separate score matrix $S$ for every head. Dividing the work into heads does not change the total number of arithmetic operations. It does, however, change the layout of the intermediate results, and the storage required for the score matrices increases with the number of heads.

![Score matrices in multi-head attention](/image/transformer/13_matrixS_multihead.png)

As before, each score matrix passes through Softmax, producing one probability matrix $P$ per head.

![Softmax in multi-head attention](/image/transformer/14_softmax_multihead.png)

Each probability matrix is multiplied by the Value matrix for the corresponding head. The resulting context vectors are then concatenated.

![Context vectors in multi-head attention](/image/transformer/15_context_vector_multihead.png)

At this point, the per-head results are merely placed next to one another. Multiplying them by $W_O$ mixes information across heads and produces the final attention output matrix $O$.

![Output projection in multi-head attention](/image/transformer/16_output_projection_multihead.png)

Splitting into heads does not copy the data. A tensor of shape `[B, S, 768]` is reshaped into `[B, S, 12, 64]` and transposed to `[B, 12, S, 64]`. Reshape and transpose only change the tensor's shape and strides, so they usually remain view operations and launch no kernel. If a later operation requires contiguous memory, however, an actual copy kernel may be launched at that point.

## Batched Multi-head Attention

Next, let us add a batch dimension. A batch allows multiple independent inputs to be processed at once. When the batch size increases, the QKV projection is performed as shown below. Here, the batch size is 2, so the amount of MMA (matrix multiply-accumulate) work doubles. The storage required for the intermediate values also scales with `batch_size * num_heads`.

![Batched QKV projection](/image/transformer/17_QKV_projection.png)

The Query, Key, and Value tensors for the different batch elements are handled independently through the $PV$ computation.

![Batched attention](/image/transformer/18_attention.png)

The per-head $PV$ results belonging to the same batch element are concatenated and multiplied by $W_O$, mixing information across heads. This produces two output matrices $O$, one for each batch element.

![Batched attention output](/image/transformer/19_attention_output.png)

# FFN (Feed-Forward Network)

So far, attention has mixed information across tokens. The feed-forward network, by contrast, processes each token independently. Its computation is straightforward. First, a GEMM expands every token from $d_{\text{model}}$ dimensions to $d_{\text{ff}}$ dimensions by multiplying it by a $d_{\text{model}} \times d_{\text{ff}}$ weight matrix, where $d_{\text{ff}} > d_{\text{model}}$. An activation function such as GELU then filters the resulting features element-wise. Finally, another GEMM multiplies by a $d_{\text{ff}} \times d_{\text{model}}$ weight matrix to reduce the representation back to its original dimension. Both dense matrices are learned weights. Recent models may instead use variants such as SwiGLU, which I will discuss in the next post.

![Feed-forward network](/image/transformer/20_FFN.png)

By expanding the dimension, applying a nonlinear filter, and reducing the dimension again, the FFN acts as a kind of pattern detector for each token.

# LM Head

The LM head converts the model's final internal representations back into a form that humans can interpret, which is why it is also called unembedding. After the hidden vectors have passed through many Transformer layers, the LM head compares each vector against all tokens in the vocabulary to determine which token is the best match. Vector similarity is computed with dot products, and because many output vectors must be compared with all vocabulary tokens at once, this operation is implemented as a GEMM.

![LM head](/image/transformer/21_lm_head.png)

The result contains a score for every vocabulary token at every output position. Softmax can convert these scores into probabilities, after which a sampling strategy selects the output token for each row. Greedy decoding simply selects the largest score; when greedy decoding is used, explicitly computing Softmax is unnecessary.

# Revisiting the Decoder-only Transformer (Pre-norm)

We have now covered the operations needed to understand the Decoder-only Transformer. In the diagram below, `Self-Attention(LayerNorm())` and `FFN(LayerNorm())` together form one Transformer layer. A real model stacks many such layers, each with its own learned weights.

Along the left side of the diagram is a stream of information that is repeatedly added to rather than overwritten. This is the residual stream. The attention and FFN results computed in each layer are added back into it.

![Decoder-only Transformer overview](/image/transformer/01_overview.png)

The equations below summarize the complete Transformer structure. The blue $x_0$ in the result represents the residual stream. A post-norm architecture still has residual connections, but because the stream passes through LayerNorm after each sublayer, $x_0$ does not remain unchanged in the same way.

![Residual stream](/image/transformer/22_residual_stream.png)

# Training (+ Causal Masking, Part 1)

Now let us examine what it means to “train” a Transformer. Training means finding appropriate values for the weights—the $W$ matrices introduced throughout the previous sections. Before doing so, however, we need to define what makes a set of weights appropriate.

Our objective can be summarized as “predict the next token.” Mathematically, this means minimizing the following loss. Here, $x_i$ is the correct token at position $i$, and $P(x_i \mid x_{<i})$ is the probability that the model assigns to that correct token after observing the preceding context. The context contains only the tokens that appear before the target position; future tokens cannot be used because they do not yet exist during actual generation.

$$

\mathcal{L} = -\frac{1}{S}\sum_{i=1}^{S} \log P(x_i \mid x_{<i})

$$

Consider the sentence “The acornjelly is good.” It provides several training examples at once: after “The,” the correct next token is “acornjelly”; after “The acornjelly,” the next token is “is”; and after “The acornjelly is,” the next token is “good.” In other words, we want the model to increase the probability of “acornjelly” after seeing “The,” of “is” after seeing “The acornjelly,” and of “good” after seeing “The acornjelly is.” Let us express this objective using the Transformer architecture.

First, the tokens of the complete sentence are converted into IDs and then into vectors through token embedding, positional embedding, and LayerNorm. These vectors are passed into the attention block.

In the figure below, consider the striped cell outlined in red. It represents the Key associated with “is” as seen by the Query from “acornjelly.” But “is” appears after “acornjelly,” so the model must not use it when predicting from that position.

![Causal masking](/image/transformer/23_causal_masking.png)

Those cells are therefore masked with $-\infty$. After Softmax, their probabilities become 0. This is causal masking.

![Softmax with causal masking](/image/transformer/24_softmax_with_causal_masking.png)

The remaining computation follows the attention and Transformer operations described earlier. At the LM head, we obtain the result shown below. Each yellow cell contains the probability assigned to the known correct next token at that position. Because causal masking prevents the model from seeing later tokens, each of these values is precisely the probability assigned to the correct token based only on the preceding context. Taking the logarithm turns products into sums, changing the sign converts maximization into minimization, and dividing by the sequence length gives the mean negative log-likelihood. Gradient-based optimization can then minimize this loss. We negate the log-probability because gradient descent is formulated as a minimization procedure, whereas our original goal is to maximize the probability of the correct token.

![LM head with the loss function](/image/transformer/25_lm_head_with_loss_function.png)

This allows the model to make predictions at all $S$ sequence positions simultaneously. The positions can be processed in parallel because the model's prediction at one position is not used to construct the input for the next position. Regardless of whether it correctly predicts “acornjelly” from “The,” it still receives the ground-truth prefix “The acornjelly” when predicting “is.” This is teacher forcing. Thanks to teacher forcing, all positions can be computed in parallel during training. Everything up to this point is the forward pass.

The backward pass uses the forward result to compute the gradient of the loss with respect to every parameter through the chain rule and vector–Jacobian products (VJPs). An optimizer then uses those gradients to update the weights. The underlying idea is gradient descent: repeatedly move in the direction opposite the local slope to approach a local minimum. Imagine trying to walk down a mountain at night. You cannot see your surroundings and can only feel the local slope beneath your feet. The best available strategy is to step downhill.

In the left side of the following figure, the ground rises to the right, so the downhill direction is left. In the right side, the opposite is true. Mathematically, we move in the direction opposite the gradient. To obtain that gradient, we would like to differentiate the objective with respect to each parameter. Explicitly constructing and storing the full derivative is impractical, so frameworks use the chain rule to connect the local derivative of each operation.

![Gradient descent](/image/transformer/26_gradient_descent.png)

Transformers use a finite set of operation types, so the required derivative rules can be derived in advance and implemented as kernels. Instead of constructing a full Jacobian, each rule directly computes the product of the upstream gradient vector and the local Jacobian. This is a VJP. Representative VJP operations are summarized in the following table.

![Backward-pass VJPs](/image/transformer/27_backward_VJP.png)

Frameworks store these per-operation rules, while deriving and implementing a VJP for an entire block as a fused kernel becomes an optimization technique in its own right—for example, in FlashAttention. The red entries in the table also show that many values computed during the forward pass are reused during the backward pass. Those activations must therefore remain stored, which is one of the main sources of memory consumption during training.

Once the gradients are available, the optimizer determines exactly how to update the weights. Optimizer design is beyond the scope of this post.

# Inference

Now let us see how a Transformer performs inference. When we send a prompt to GPT, how does this architecture produce an answer? It first reads the prompt and then generates a new sequence. Broadly speaking, this process has two stages: processing the prompt and generating the first output token, followed by generating the remaining tokens one by one. The first stage is called **prefill**, and the second is called **decode**.

## Prefill (+ Causal Masking, Part 2)

Suppose the input prompt is “Please recommend some traditional Korean food.” The prompt is first tokenized and embedded to form a matrix representation of the sequence. This matrix is then passed through the attention block.

The attention block first computes Query, Key, and Value through the QKV projection. It then multiplies Query by Key to form the score matrix $S$. Causal masking is still required: tokens in the past must not be allowed to see tokens in the future.

You might wonder whether we can keep only the final row, since only the last generated token will ultimately be used. This is possible in the final Transformer layer, but not in the earlier and intermediate layers because the representations that later tokens need to attend to would disappear. Let us examine this directly with matrices.

![Attention with causal masking](/image/transformer/28_attention_with_causal_masking.png)

The following figure shows what goes wrong if prefill computes only the Query for the final token. The yellow Query is generated only for the last token, while the green Keys and blue Values are generated for all tokens. This may initially seem reasonable because the last Query still needs to access the Keys and Values of the earlier tokens. After one attention layer, we obtain an output for the final token. The subsequent FFN introduces no problem because the FFN does not mix information across tokens.

The problem appears in the next layer. Feed the output at the bottom of the figure into the next layer as its input. That layer needs an input representation for every token to project its Keys and Values, but only the final token's representation remains. The other tokens no longer exist as attention targets. Consequently, later layers cannot attend to them. During prefill, every layer must therefore process all prompt tokens while using causal masking to hide only future tokens.

![First layer of an incorrect prefill implementation](/image/transformer/29_first_layer_of_wrong_prefill.png)

The per-head $PV$ results are then concatenated within each batch element and multiplied by $W_O$ to mix information across heads. In the FFN, a GEMM expands the dimension, an activation function filters the features, and another GEMM reduces the dimension. After this process repeats across all layers, the LM head slices out only the final position and selects a token. The first generated token is now complete.

![LM head during prefill](/image/transformer/30_lm_head_prefill.png)

## Decode (+ KV Cache)

Prefill processes the prompt and generates the first output token. Decode then generates the tokens that follow. Conceptually, each step receives every token from the beginning through the most recently generated token and predicts the next one. This reveals the key difference between inference and training. During training, the model receives the ground-truth sequence rather than its own previous predictions, so it can compute predictions for all sequence positions simultaneously. During inference, however, the token generated in one step becomes an input to the next. This data dependency prevents token-generation steps from being processed in parallel.

Does this mean that every decode step must recompute logits for all $S$ positions and then keep only the last row? Fortunately, no. We only need to generate the next token.

During prefill, keeping only the final token would have left later layers without the inputs required to project Keys and Values for the earlier tokens. Decode is different because those values have already been computed. The **KV cache** stores the per-layer Keys and Values produced during prefill and previous decode steps so they can be reused. With a KV cache, each decode step computes only the Query, Key, and Value of the newest token, while retrieving the earlier Keys and Values required by each layer from the cache.

Let us walk through the process. First, the newest token passes through the QKV projection to produce its Query, Key, and Value.

![QKV projection during decode](/image/transformer/31_QKV_projection_decode.png)

The model then retrieves the Keys and Values of all previous tokens from the KV cache. We do not need the previous Queries, because we are not regenerating the already-produced tokens. We do, however, still need their information, which is represented by their Keys and Values. Instead of recomputing these repeatedly, we store them in the KV cache. The following figure shows the attention block while generating token $N$. The blue outlines indicate values loaded from the KV cache, while the red outlines indicate the Query, Key, and Value computed for the current token.

![Attention while generating token N](/image/transformer/32_attention_nth.png)

When generating token $N+1$, the model also needs the information from token $N$, so the contents of the KV cache continue to grow. The following figure shows the attention block for token $N+1$. Because the cache contains no Keys or Values for future tokens, a triangular causal mask is unnecessary at this stage.

![Attention while generating token N+1](/image/transformer/33_attention_n+1.png)

## Prefill vs. Decode

Let us compare prefill and decode. Prefill processes $S$ tokens at once, while decode processes one new token at a time. Consequently, the QKV projection, FFN, and LM head appear as GEMMs during prefill but are effectively GEMVs during decode.

Next, consider how often a weight loaded from DRAM can be reused. During prefill, the hidden states of $S$ tokens are multiplied by the same weights, so each loaded weight value can be reused across $S$ tokens within the GEMM. During decode, the hidden state of only one token is multiplied by those weights, so a loaded weight value is used only once. Prefill computes all Keys and Values for the prompt and stores them in the KV cache. Each decode step then reads the cached values, computes one new Key and Value, and appends them to the cache.

![Prefill versus decode](/image/transformer/34_prefill_vs_decode.png)

# Summary

![Notation](/image/transformer/35_notation.png)

| # | Block | Input shape | Weight shape | Output shape | Operation |
| --- | --- | --- | --- | --- | --- |
| **Embedding** |  |  |  |  |  |
| 0-1 | Token embedding | IDs `[B, M]` | `wte [50304, 768]` | `[B, M, 768]` | gather |
| 0-2 | Position embedding | positions `[M]` | `wpe [1024, 768]` → slice `[p : p+M]` | `[B, M, 768]` | gather + add |
| **Repeated × L (=12)** |  |  |  |  |  |
| 1 | LN1 (pre-norm) | `[B, M, 768]` | $\gamma$ `[768]`, $\beta$ `[768]` | `[B, M, 768]` | row reduction + element-wise |
| 2 | QKV projection (`c_attn`) | `[B·M, 768]` | $W$ `[768, 2304]`, $b$ `[2304]` | `[B, M, 2304]` | **GEMM** |
| 3 | Split + reshape | `[B, M, 2304]` | — | Q, K, V: `[B, 12, M, 64]` each | layout transform |
| 4 | KV-cache write | K, V `[B, 12, M, 64]` | buffers `[B, 12, S_max, 64]` ×2 | in-place | copy |
| 5 | Scores $QK^T$ | Q `[B,12,M,64]`, K `[B,12,S,64]` | — | `[B, 12, M, S]` | **batched GEMM** (`batch = B·h = 12`) |
| 6 | Scale | `[B, 12, M, S]` | scalar $1/\sqrt{64}=1/8$ | same | element-wise |
| 7 | Causal mask | `[B, 12, M, S]` | mask `[M, S]` | same | element-wise |
| 8 | Softmax | `[B, 12, M, S]` | — | `[B, 12, M, S]` | row-wise reduction |
| 9 | $P·V$ | P `[B,12,M,S]`, V `[B,12,S,64]` | — | `[B, 12, M, 64]` | **batched GEMM** |
| 10 | Merge heads | `[B, 12, M, 64]` | — | `[B, M, 768]` | layout transform |
| 11 | Output projection (`c_proj`) | `[B·M, 768]` | $W$ `[768, 768]`, $b$ `[768]` | `[B, M, 768]` | **GEMM** |
| 12 | Residual add | `[B, M, 768]` | — | `[B, M, 768]` | element-wise |
| 13 | LN2 (pre-norm) | `[B, M, 768]` | $\gamma$ `[768]`, $\beta$ `[768]` | `[B, M, 768]` | row reduction + element-wise |
| 14 | FFN up-projection (`c_fc`) | `[B·M, 768]` | $W$ `[768, 3072]`, $b$ `[3072]` | `[B, M, 3072]` | **GEMM** |
| 15 | GELU | `[B, M, 3072]` | — | `[B, M, 3072]` | element-wise |
| 16 | FFN down-projection (`c_proj`) | `[B·M, 3072]` | $W$ `[3072, 768]`, $b$ `[768]` | `[B, M, 768]` | **GEMM** |
| 17 | Residual add | `[B, M, 768]` | — | `[B, M, 768]` | element-wise |
| **Output** |  |  |  |  |  |
| 18 | Final LN | `[B, M, 768]` | $\gamma$ `[768]`, $\beta$ `[768]` | `[B, M, 768]` | row reduction + element-wise |
| 19 | Last-token slice (decode) | `[B, M, 768]` | — | `[B, 1, 768]` | slice |
| 20 | LM head | `[B·M, 768]` | `wteᵀ [768, 50304]` (tied) | `[B, M, 50304]` | **GEMM/GEMV** |
