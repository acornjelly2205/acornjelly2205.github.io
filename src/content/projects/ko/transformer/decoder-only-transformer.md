---

title: "Decoder-only Transformer를 커널 관점에서 해부하기 — Training, Prefill, Decode, KV Cache"

description: "Decoder-only Transformer의 구조와 연산 과정을 살펴보고, GPU에서 어떤 연산과 커널로 실행되는지 분석한다"

pubDatetime: 2026-09-18

order: 1

translationKey: decoder_only_transformer_overview

tags:

  - transformer

  - decoder-only

  - GPU kernel

  - KVCache

  - decode

  - prefil

draft: false

---

이번 포스트에서는 Decoder-only transformer(pre-norm기준)의 구조 및 연산에 대해 커널 관점에서 분석하고 training과 inference단계에서 이 구조를 어떻게 사용하는지 알아본다. 그리고 이후 포스트에서는  inference의 prefill과 decode 단계의 커널 런타임 breakdown을 통해 각 단계의 병목이 무엇인지 파악해본다.

# Decoder-only transformer (pre-norm)

시작은 decoder-only transformer의 구조를 보여주는 것으로 열도록 하겠다. decoder-only transformer구조는 post-norm과 pre-norm으로 나눌 수 있는데 이 글에서는 pre-norm을 기준으로 한다. 구조는 생각보다 간단하다. 여기서 토큰은 transformer가 이해할 수 있는 의미 단위라고 생각하면 된다.

![01. decoder-only transformer.png](/image/transformer/01_overview.png)

# Token Embedding & Positional Embedding

일단 사람들이 사용하는 자연어를 기계가 이해할 수 있는 형태로 바꿔주자. 이런 과정을 embedding이라고 한다. 그리고 앞서 말했듯이 token은 transformer가 이해할 수 있는 의미 단위다. 일단  transformer는 자기만의 단어 사전을 가지고 있다. 이때 token과 ID사이 매핑을 vocabulary라고 하고, ID와 d_model-dimensional vector 사이 매핑, 즉, 사전을 embedding table이라고 한다.

그럼 일단 input으로 4개의 token이 들어왔다고 하자. 각 token은 자신의 ID로 표현되어 있다. 그럼 이제 transformer가 가지고 있는 사전에서 이 ID를 찾는다. 그리고 그 ID가 의미하는 vector로 표현해주면 된다. 예를 들어보자면 “사과… 사과가 몇 번이더라? 1번! 그럼 1번 vector가 뭐야?” 이런 식이다. 이 과정을 수식으로 표현하면 행렬 곱이지만, 실제 커널에서는 array에서 해당 index를 찾는 lookup과정으로 해결할 수 있다. 결과적으로 4개의 vector로 표현되었다.

![02. token embedding.png](/image/transformer/02_token_embedding.png)

자 여기서 “The dog chased the cat”(개가 고양이를 쫓았다.)과 “The cat chased the dog”(고양이가 개를 쫓았다.) 라는 문장을 비교해보자. 두 문장에 사용된 단어는 같지만, 순서에 따라서 의미가 달라진다. 그렇다면 transformer에게도 이 위치 정보를 전달해야 할 것이다. 이 과정이 바로 positional embedding이다. 

Positional embedding에는 여러 방법이 있지만 여기서는 가장 간단한 방법을 선택하겠다. 바로 각 위치별로 사전에 vector를 매핑해놓고, 사전에서 ID를 찾듯, 자신의 위치에 해당되는 vector를 찾는 것이다. 이 방법은 받을 수 있는 최장 token 길이가 정해진다는 단점이 있다. (사전에 정해 놓았으니까!) 반면, 그렇지 않은 방법도 있기는 하다. (RoPE 등, 포스팅 예정)아무튼! 이렇게 자신의 token embedding과 positional embedding을 더해주면 우리가 처리할 token이 vector 형태가 된다.

![03. positional embedding.png](/image/transformer/03_positional_embedding.png)

# LayerNorm

근데 여기서 문제! 발생! 각 단어마다 숫자를 매핑하는 것 까지는 알겠다, 근데 크기 차이가 너무 난다는 문제가 생긴다. 숫자 크기 차이가 너무 많이 생기면 이후 단계에서 값을 비교할 때 문제가 생긴다. 우리가 여기까지에서 하고 싶은 것은 각 단어를 적절한 위치 vector로 바꿔주는 것이지만, 그 크기가 너무 너무 많이 차이가 나기를 원하는 것은 아니다. 따라서 LayerNorm을 통해서 각 token을 나타내는 element 값이 평균이 0이 되고, 표준편차가 1 인 분포를 따르도록 바꿔주자! 이후 학습된 γ와 β로 다시 scale·shift 한다. 정규화로 크기는 통제하되, 어떤 분포가 좋을지는 모델이 학습하게 두는 것이다. 이 과정이 바로 LayerNorm이다. 이 LayerNorm은 한 번이 아니라 각 layer 시작마다 반복된다

![04. LayerNorm.png](/image/transformer/04_layernorm.png)

이러한 정규화를 하기 위해서는 기존 값들의 평균과 표준편차를 알아야 한다. 이 과정에서 각 token을 의미하는 row의 정보를 파악하기 위해 reduction이 필요하고, 이후 각 element는 element-wise하게 계산하는 과정을 거치게 된다. 그리고 이 연산은 읽어온 데이터를 재사용하지 않는다. 따라서 매번 새로 데이터를 로드해야 하고 arithmetic intensity가 낮아 일반적으로 memory bandwidth의 영향을 크게 받는 연산이다.

# Self-Attention 이란?

“나는 도토리묵을 좋아해. 이것은 너무 맛있으니까!“ 라는 문장을 보자. 문장 속 “이것”은 무엇일까? 바로 “도토리묵”이다. 어떻게 답을 찾았는지 생각해보자. 아마도 다른 단어를 참고했을 것이다. 모든 단어를 똑같이 참고했을까? 문장 속 “나”와 “도토리묵”이 “이것”이 무엇인지 파악하는데 똑같이 중요한 정보일까? 혹시 그렇다고 생각하면 “나”를 다른 이름으로 바꿔봐라. 

아무튼 우리가 하고 싶은 것은 다른 단어를 각자 중요한 만큼 참고해서 이 단어가 무엇일지 파악하는 것이다. 문장 속 단어는 사전적 의미만 가지고 있는 것이 아니기 때문이다. 단어는 문맥 속에서 여러가지 의미를 내포하게 된다. 이때, “중요한 만큼”을 “서로 다른 가중치”로 바꾸고, 정보를 섞는다는 행위 여러 개를 한번에 행렬 곱셈으로 표현할 수 있다. 

attention이 하는 일이 바로 이것이다. 중요한 만큼 단어의 정보를 가져와서 섞는 것. 

## Query, Key, Value projection

attention은 4가지 가중치를 가지고 있다. 그 중 3가지가 Query, Key, Value를 위한 것이다. Query는 “너 이 정보 가지고 있니?” 하고 물어보는 것이다. “Key”는 나 “이런 정보 가지고 있어”라는 각 단어의 색인이고, “Value” 는 각 단어가 “가지고 있는 값”을 나타낸다.  책으로 비유를 해보자. 도서관에서 검색을 한다. “너 이런 책 이니?”는 Query에 해당한다. 그리고 검색할 때 보는 각 책을 대표하는 색인은 “Key”, 각 책의 내용이 “Value”에 해당한다.

그럼 이제 각 단어로부터 Query, Key, Value를 뽑아내보자.  이 과정은 GEMM이라는 연산으로 이루어진다. GEMM은 행렬 곱셈을 말한다. GEMM은 (M × K) × (K × N) → (M × N)이다. QKV projection에서 M은 한 번에 처리하는 토큰 수, K는 d_model, N은 출력 차원이다.

일단 지금은 단일 head, batch=1로 살펴보자. 

우선 Query부터 해보자. N은 Sequence의 길이, d_model은 각 토큰이 몇 차원의 vector로 표현되는지를 나타낸다. 아래 이미지를 보면, 입력으로 들어온 행렬 X는 4개의 token으로 이루어져 있고, 각 토큰은 8차원의 vector로 표현된다. 이 입력 행렬 X를 Query를 위한 가중치인 W_Q와 곱해보자. 그렇게 나온 결과 행렬이 바로 Query가 된다.

![05_query_projection_with_size.png](/image/transformer/05_query_projection_with_size.png)

Key와 Value도 Query와 같은 방식으로 얻어진다. 가중치 행렬만 W_K, W_V로 바뀔 뿐이다. 그럼 이제 노란색 Query와 연두색 key, 파란색 Value가 만들어졌다!

그림에서는 W_Q, W_K, W_V를 따로 그렸지만, nanoGPT는 `c_attn = nn.Linear(n_embd, 3 * n_embd)` 하나로 선언한다. 즉 (M × 768) × (768 × 2304) GEMM 커널 1개로 Q, K, V를 동시에 뽑는다. 그리고  이후 `split`으로 나누는 건 텐서 view라 커널이 뜨지 않는다.

![06. projection.png](/image/transformer/06_projection.png)

## attention score ~ logits

이제 각 단어한테 물어보고 답을 들어보자. Query로 물어보고 Key로 대답을 들으면 된다. 근데 내가 찾고 있는 것이랑 Key가 얼마나 잘 맞는지 어떻게 알 수 있을까? 우리의 목표는 잘 맞는 토큰에 더 높은 가중치를 주고 아닌 토큰에는 낮은 가중치를 주고 정보를 섞는 것이다. 이때 우리는 내적을 사용한다. 두 벡터를 내적 한 값은 두 벡터가 같은 방향을 향하고 있을수록 값이 커진다. 그렇게 내적 한 행렬을 matrix_S라고 하자. 

![07. matrix S with meaning.png](/image/transformer/07_matrixS_with_meaning.png)

이제 두 벡터가 잘 맞을수록 더 큰 요소 값을 갖는 matrix_S가 완성 되었다! 그런데 여기서 문제가 있다. d_model의 값은 모델마다 다르고, 꽤 큰 값을 갖는다. 또한, 두 벡터의 내적은 같은 위치에 있는 두 값의 곱을 모두 합한 값으로, 분산이 d_head에 비례해서 커진다.  head가 뭔지는 다음 section에 나온다. 여기서 head가 한 개이고, d_head = d_model/head개수이므로, d_head = d_model/head가 된다. 따라서 $\sqrt{d_{\text{head}}}$로 나눠준다. 그리고 여기서 Softmax()를 통과시켜서 각 질문에 대한 토큰들의 대답의 합이 1이 나오도록, 확률화 시켜준다. 이렇게 나온 결과를 matrix_P라고 하자. 

![08. softmax.png](/image/transformer/08_softmax.png)

matrix_P의 각 요소가 어떤 것을 의미하는가? 바로 각 Query에 대한 각 단어들의 대답을 나타낸다. 그리고 각 row는 합이 1이 되었으니, 이 대답을 확률로 해석할 수 있게 된 것이다! 자 그럼 이제 각 질문에 대한 답을 얻었다. 각 단어가 얼마나 중요한지 알게 되었으니  이 값과 Value 행렬을 곱해보자. 그러면 각 token은 이제 자신이 던진 질문(=query)에 대한 다른 token들의 대답을 얻게 되었다! 이것을 Context Vector라고 해보자.

![09. context vector.png](/image/transformer/09_context_vector.png)

마지막으로, 어떠한 context vector가 있을 때, 서로 다른 head간 내용을 섞어주자. 지금 단계에서는 head 개수가 1이지만 이후 단계를 보면 head가 뭔지 알 수 있을 것이다. context vector에 W_O를 곱해주면 head간 내용이 서로 섞여서 잘 반영된 matrix O가 나온다.

![10. output projection.png](/image/transformer/10_output_projection.png)

## Multi-head attention

자 이번에는 Multi-head attention이 뭔지 알아보자.  attention에서 우리의 목적은 각 단어한테 “너 내가 찾고 있는 이런 단어 맞니?”라고 물어보고 적합한 단어 정보를 더 높은 비중으로 가져와서 참고하는 것이었다. 이 행동을 조금 발전시켜보자. 기존에는 하나의 관점에서 질문 했었다. 근데 이번에는 여러 관점의 질문을 동시에 해서, 여러 “관점”에 대한 정보를 한번에 알아보는 것이다. 단, 전체 연산 횟수는 거의 그대로 하면서!

일단 해보자. attention의 첫 단계는 projection을 거쳐서 Query, Key, Value를 얻어내는 것이었다. 이 과정에서 우리는 해당되는 weight들을 2개로 쪼갤 것이다. 정확히는 head개수 만큼인데, 일단 2로 두자.  아래 이미지는 Query를 얻는 과정이다. 이미지를 보면, 노란색 W_Q가 둘로 나눠져 있고, 노란색 그라데이션 query가 두 동강난 것을 볼 수 있다. 나머지 Key와 Value도 똑같이 적용해주자.

![11. query projection in multihead attention.png](/image/transformer/11_query_projection_in_multihead_attention.png)

![12. projection in multihead attention.png](/image/transformer/12_projection_in_multihead_attention.png)

이제 이렇게 얻은 Query와 Key를 곱해서 matrix S를 얻어보자. Head 별로 나눈 Query 와  Key를 곱해주면 된다.  이 과정을 거치면 Head별로 matrix S를 얻게 된다. 이때, 연산 과정을 살펴보면, 이 과정에서 전체 연산량은 변하지 않는다. 다만, 중간 값이 저장되는 형태로, matrix S를 위한 저장 공간은 Head의 개수 배로 늘어나게 된다.

![13. matrixS multihead.png](/image/transformer/13_matrixS_multihead.png)

이제 앞 단계와 마찬가지로 matrix S를 Softmax함수에 통과시키자. 그럼 matrix S와 같은 개수의 matrix P가 나오게 된다.

![14. softmax multiehad.png](/image/transformer/14_softmax_multihead.png)

이렇게 Softmax를 거쳐서  확률화 시킨 matrix P를 head별로 나눠 놓은 Value와 곱해준 후 서로 연결(Concat) 해준다. 

![15. context vector multihead.png](/image/transformer/15_context_vector_multihead.png)

지금은 아직 각 head내용을 가진 PV가 연속하게 붙어있는 형태에 불과하다. 따라서 W_O를 곱해서 두 head의 내용이 잘 섞이도록 반영해주자. 그러면 우리가 원하는 attention의 최종 결과  matrix O가 나왔다!

![16. output projection multihead.png](/image/transformer/16_output_projection_multihead.png)

head로 나눈다는 건 데이터를 복사하는 게 아니다. `[B, S, 768]` 텐서를 `[B, S, 12, 64]`로 reshape하고 `[B, 12, S, 64]`로 transpose하는 것뿐이다. reshape과 transpose는 stride만 바뀌는 view라 커널이 뜨지 않는다. 다만 이후 연산이 contiguous를 요구하면 그때 실제 복사 커널이 하나 생긴다.

## Batched Multi-head attention

multi-head에 이어서 이번에는 batch 개수를 늘려보겠다. batch는 한번에 여러 데이터를 처리해보자! 이다. 이때 두 데이터는 서로 상관이 없다. 따라서 batch가 늘어나면, QKV projection은 다음 그림과 같이 이루어진다. 지금은 batch가 2개인 상황이고, 따라서  MMA(Matrix Multiply-Accumulate)연산의 양이 2배로 늘었다! 필요한 저장 공간도 batch * num_head 배로 늘어난다.

![17. QKV prjection.png](/image/transformer/17_QKV_projection.png)

이렇게 얻은 Query, Key 그리고 Value는 독립적으로 다루어지면서 PV까지 계산하게 된다. 

![18. attention.png](/image/transformer/18_attention.png)

그리고 동일한 batch에 속한 head들의 PV를 Concat 해준 후 W_O를 곱해서 head간 내용을 반영해주면 matrix O 2개를 얻을 수 있다!

![19. attention output.png](/image/transformer/19_attention_output.png)

# FFN (Feed-Forward Network)

여태까지는 각 토큰끼리 정보를 섞는 과정이었다. 지금부터는 feed-forward network 단계로, 각 토큰의 정보가 독립적으로 처리된다. 이 과정의 연산은 단순하다. 일단 각 토큰의 차원을 확장해준다! 차원 확장은 GEMM으로 나타낼 수 있다. d_model 차원으로 표현된 각 토큰에 d_model x d_ff 행렬을 곱해주면 된다 (d_ff > d_model). 이후 각 element를 GELU와 같은 Activation함수에 넣어서 패턴을 필터링한다. 그리고 다시 차원을 축소하여 원래 사이즈 크기로 돌려놓는다. 이 과정 역시 GEMM으로 이번에는 확장 과정과 반대로 d_ff x d_model 행렬을 곱해준다. 이때, 차원을 확장하고 축소하는 데 사용하는 dense 행렬은 weight중 하나로 학습되는 값이다. 최신 모델에서는 SwiGLU방식을 사용하기도 한다 (다음 포스트에서 다룰 예정이다.). 

![20. FFN.png](/image/transformer/20_FFN.png)

FFN은 차원을 확장하고 필터링하고 다시 축소해줌으로써 일종의 “패턴 탐지기” 역할을 한다.

# LM_Head

LM_Head 는 가장 마지막에 지금까지의 결과물을 다시 사람이 알 수 있는 형태로 바꿔주는 과정이다. 그래서 unembedding이라고도 부른다. 지금까지 여러 레이어를 거쳐서 각 vector를 얻었을 것이다. 이제 이 모델이 가진 vocabulary의 token들 중 어떤 token과 가장 유사한지 살펴보고 가장 적합한 token을 뽑는 것이다! 이 때 두 vector의 유사도는 내적으로 계산하는데, 여러 결과 vector를 한번에, 모든 토큰들에 대해 검사해야 하니 GEMM으로 수행한다.  

![21. LM head.png](/image/transformer/21_lm_head.png)

이렇게 하면 각 vector가 각 토큰들에 대해 어떤 점수를 갖는지를 얻을 수 있다. 이제 Softmax를 거쳐서 각 점수를 확률화 해주고, 각 row에서 값을 뽑는 sampling과정을 거치면 끝! 각 vector에 해당하는 토큰이 결정되었다! (그냥 가장 큰 값을 뽑는 것을 greedy decoding이라고 하는데, 이 방식으로 토큰을 선택할거면 사실 softmax도 필요 없다.)

# RE: Decoder-only Transformer (pre-norm)

지금까지 Decoder-only Transformer의 구조를 이해하는데 필요한 연산이 무엇인지 살펴보았다. 이제 아래 그림을 보면 구조가 이해될 것이다. 이때 Self_attention(Layer_Norm())과 FFN(Layer_Norm())을 합쳐서 하나의 Layer라고 한다. 실제 모델에서는 이 Layer가 여러 겹 쌓이게 되고, 각 layer에 필요한 weight를 학습해야 한다.

그리고 그림의 왼쪽을 보면 덮어써지지 않고 더해지기만 하는 한 줄기의 정보가 보이는데 이것이 바로 residual stream이다. 각 layer에서 계산된 attention 결과와 FFN 결과는 이 residual stream에 더해진다.  

![01. overview.png](/image/transformer/01_overview.png)

transformer전체 구조를 식으로 써보면 아래와 같다. 결과 부분에 있는 파란 x0가 바로 residual stream이다. (참고로 Post-norm 형태에서는 residual connection은 있지만, 매번 LN을 통과해서, x0가 그대로 살아남지는 않는다.)

![22. residual stream.png](/image/transformer/22_residual_stream.png)

# Training (+ causal masking-1)

이제는 그래서 이 Transformer를 어떻게 “학습”시킨다는 건지 알아볼 것이다. 그렇다면 “학습”이 의미하는 것은 뭘까? 그건 바로 적절한 weight 값을 찾는 것이다. 앞서 transformer의 각 연산 블럭을 공부하면서 나온 W행렬들이 바로 weight이다. 그런데 이 과정을 수행하기 위해서는 우선 “적절한 weight”가 뭔지부터 정의해야 한다. 적절한 weight를 어떻게 정의할 수 있을까?

일단 우리의 목적은 한마디로 “다음에 올 토큰 맞추기”로 정의할 수 있다. 이 말을 식으로 바꾸면, 아래 식을 최소화 하는 것이다. 식을 조금 더 풀어보자면, 여기서 $x_i$는 i번째 정답 토큰을 말하고,  $P(x_i \mid x_{<i})$ 은 모델이 앞 문맥을 보고 “정답 토큰에 부여한 확률”이다. 여기서 앞 문맥이란, 지금까지 나온 토큰들이다. 지금 맞추려는 위치의 토큰 앞에 나온 토큰만 보고, 그 뒷 부분은 보지 않는다. 왜냐하면 실제 상황에서 만들어지지도 않은 미래의 토큰은 알 수 없으니까! 

$$

\mathcal{L} = -\frac{1}{S}\sum_{i=1}^{S} \log P(x_i \mid x_{<i})

$$

그럼 이제 학습을 시켜보자. “The acornjelly is good”이라는 문장이 있다. 그럼 이 문장으로 어떻게 학습을 진행할까? 이 한 문장을 나눠보면, 여러 개의 학습 데이터가 나온다. “The” 다음 단어는 “acornjelly”, “acornjelly” 다음은 “is”, “is” 다음은 “good” 이런 식이다. 그럼 이제 우리가 하고 싶은 것을 정리해보면, “The”를 보고는 “acornjelly”, “The acornjelly”를 보고는 “is”, “The acornjelly is”를 보고는 “good”을 맞출 확률을 높이는 것이다. 이것을 transformer의 구조를 사용해서 표현해보자.

 일단 문장 전체의 각 token을 ID로 표현하고, token embedding, positional embedding, 그리고 LayerNorm을 거쳐서 벡터로 표현하고 attention 블럭에 넣어보자. 

아래 그림을 보면, 빨간색 상자 부분을 포함해서 빗금이 쳐져 있는 칸이 보일 것이다. 이 칸은 “acornjelly”가 각 토큰에게 던진 query에 대한 “is”의 답(색인)이다. 그런데 “is”는 사실 “acornjelly” 다음에 나오는 단어라서 참고할 수 없다. 

![23. causal masking.png](/image/transformer/23_causal_masking.png)

따라서 해당 칸들은 $-\infty$ 로 masking해준다. 그리고 이 칸들이 softmax를 지나면 0이 된다. 이것이 바로 causal masking이다.

![24. softmax with causal masking.png](/image/transformer/24_softmax_with_causal_masking.png)

이후 과정은 앞서 설명한 연산 블럭 내용을 따른다. 그렇게 LM_head까지 가면… 아래 그림과 같은 결과를 얻을 수 있다. 여기서 노란 색으로 표시된 각 칸은 우리가 알고 있는 정답 토큰(문장에서 다음에 오던 바로 그 토큰)의 확률 값을 담고 있다. 그런데 앞서 causal masking과정에서 뒤에 토큰은 모두 가렸으니, “앞 문맥을 보고 정답 토큰에 부여한 확률”이 된다. 그리고 곱셈 연산을 덧셈으로 바꿀 수 있게 로그를 씌우고, -로 부호를 바꾼 후  Sequence 길이로 나눠서 평균을 내면! 우리가 원하던 “앞 문맥을 반영한 정답 토큰”의 평균이 되고, 이제 우리는 이 값을 경사하강법으로 최소화 해주면 된다. (우리가 원하는 것은 확률의 최대화인데, 경사하강법은 기본적으로 최소화를 하기 때문에 앞에서 -로 방향을 바꿔줬다.)

![25. lm_head with Loss function.png](/image/transformer/25_lm_head_with_loss_function.png)

이렇게 하면 지금 모델의 상태에서 S(sequence 길이)개의 token을 한번에 예측해볼 수 있다. 병렬적으로 한번에 처리가 가능한 이유는 앞서 추측한 결과가 그 다음단어를 예상하는데 사용되지 않기 때문이다. (”The”로 “acornjelly”를 예측했던 아니던, “is”를 예측할때는 “The acornjelly”를 본다.) 이것이 바로 teacher forcing이다. teacher forcing 덕분에 Training은 모든 위치가 동시에 계산된다. 여기까지가 바로 forward pass이다. 

그리고 이렇게 얻은 값들을 바탕으로 목적함수 값을 최소화 하는 과정을 통해서 정답이 예측될 확률이 높은 모델의 상태를 만들어 간다. 이 backward pass에서는 chain rule과 VJP를 이용해 loss에 대한 각 parameter의 gradient를 계산한다. 이후 optimizer가 이 gradient를 이용해 weight를 update한다. 기본적으로 경사하강법을 바탕으로 하는데, 경사하강법은 그 위치의 기울기 반대방향으로 걸어가서 극소 지점에 다가가는 방법이다. 기울기 반대방향으로 걸어가는 것을 반복하면 극소 지점에 다가갈 수 있나? 하는 의문이 들 것이다. 간단하게 원리를 설명하자면 이렇다. 한밤중에 산을 내려가야 한다고 생각해보자. 주변에는 아무것도 안 보이고, 간신히 땅을 짚어서 현재 위치의 기울기만 알 수 있다. 이 상황에서 어떻게 내려가야 할까? 최선의 방법은 내리막을 따라서 내려가는 것이다. 

아래 그림을 보자. 왼쪽 그림을 보면,  내 오른쪽이 오르막인 경우는 내리막 방향이 왼쪽이다. 오른쪽 그림은 그 반대다. 이 말을 수학적으로 표현하면, “기울기 반대 방향으로 걸어간다”가 된다. 그래서 우리는 현재 지점에서 기울기를 구해야 한다. 그 기울기를 구하기 위해서 목적 함수를 각 파라미터로 편미분 하고 싶다. 근데 이건 저장할 수도 없고 사람이 구해 놓을 수도 없다. 그래서 우리는 chain rule를 이용하여 목적함수를 각 파라미터로 직접 미분하는 것이 아니라, 각 연산의 local derivative를 chain rule로 연결한다.

![26.gradient descent.png](/image/transformer/26_gradient_descent.png)

transformer에서 등장하는 연산은 사실 몇 가지로 정해져있고, 필요한 편미분 식의 종류도 정해져있다. 따라서 식을 미리 손으로 유도해놓고, 그 식을 커널로 구현하여 호출한다. 이렇게 upstream gradient 벡터와 Jacobian의 곱만 직접 계산하는 규칙을 연산별로 미리 유도해 커널로 넣어둔다. 이게 VJP다. VJP의 대표적인 연산은 아래와 같이 분류되어 저장되어 있다. 정리하면 아래 테이블과 같다. 

![27. backward VJP.png](/image/transformer/27_backward_VJP.png)

프레임 워크에서는 이렇게 연산별로 분류되어 규칙이 미리 저장되어 있고, 블럭 단위 VJP를 직접 유도하여 커널로 구현하는 것이 곧 최적화가 된다.(e.g., FlashAttention) 그리고 표에 빨간색으로 표시된 부분을 보면, forward pass에서 계산한 값들이 다시 사용되는 것을 볼 수 있다. 그래서 이때까지는 계산한 값들(activation)을 저장해두어야 한다. 여기서 메모리가 많이 필요하다.

그리고 이렇게 얻은 기울기로 weight를 어떻게 update할지가 optimizer 문제로 이어지는데, 이 문제는 여기서 다루지 않도록 하겠다! 

# Inference

자, 이제는 transformer로 추론을 어떻게 하는지 알아볼 것이다. GPT에게 뭔가를 물어볼때, 이 transformer로 어떤 것들을 해서 답을 주는 걸까? 일단 내가 입력한 프롬프트를 읽은 후 새로운 Sequence를 만들어 낼 것이다. 맞다. 이 과정은 크게 프롬프트를 읽어서 첫 토큰을 생성하는 단계 + 새로운 토큰을 줄줄이 이어서 생성하는 단계로 나누어 지고, 전자는 prefill, 후자는 decode라고 부른다. 이제 이 두 단계를 어떻게 하는지 알아보겠다. 

## prefill (+ causal masking-2)

“Please recommend some traditional Korean food”라는 입력이 들어온 상황을 생각해보자. 어떻게 하면 대답을 생성할 수 있을까? 일단, 이 프롬프트를 토큰으로 바꾼 후 임베딩을 거쳐서 이 시퀀스를 행렬로 표현해야 할 것이다. 그리고 이 행렬이 attention 블록을 통과한다. 

attention 블록에서는 우선 QKV projection을 통해 Query, Key, Value를 얻어낼 것이다. 그리고 나서 Query와 Key를 곱해서(GEMM) matrix S를 만든다. 이때도 상삼각행렬을 masking하는 causal masking을 해줘야 한다. 과거 토큰은 미래를 못보니까!

 가장 마지막에 생성한 토큰만 사용할 것이니 맨 아래 행만 남기면 안되나? 라고 생각할 수도 있다. 그리고 마지막 Layer에서는 그렇게 구현하는 것도 가능하다. 하지만 처음~중간 Layer에서는 그럴 수 없다. 중간 Layer에서 참고 당할 대상이 사라지기 때문이다. 이게 무슨 말인지, 행렬을 통해서 직접 살펴보자. 

![28.attention with causual masking.png](/image/transformer/28_attention_with_causal_masking.png)

아래 그림을 통해서, prefill 단계부터 가장 마지막 토큰의 query만 만들어 처리하면 어떤 문제가 발생하는지 알아보자. 아래 그림에서는 가장 마지막 토큰의 Query만 생성(노란색)하고, Key(초록색)와 Value(파란색)는 모든 토큰에 대해 생성하였다. 마지막 query도 다른 토큰의 색인과 값은 참고해야 하니 여기까지는 그럴 법 하다. 이렇게 attention의 한 layer를 통과하면, 가장 마지막 토큰에 대한 값이 나온다. 그리고 attention이후 FFN 까지 가도 FFN은 토큰끼리 상호작용이 전혀 없으니까 문제는 발생하지 않는다. 

문제는 그 다음이다. 가장 아래 그림의 출력을 다시 가장 위 그림의 인풋으로 넣어보자. 다음 layer에 이전 layer의 output을 이번 layer의 input 으로 삼아서 각 토큰에 대한 Key, Value projection을 하려는데, 마지막 토큰을 제외한 위치에 들어갈 값이 없다! 그럼 attention layer를 거치는 동안 다른 토큰의 값을 참고하는 것이 불가능하다. 참고할 대상이 존재하지 않기 때문이다. 이러한 이유로 prefill 단계에서는 미래의 토큰만 가리고(causal masking) attention 블럭을 처리한다.

![29. first layer of wrong prefill.png](/image/transformer/29_first_layer_of_wrong_prefill.png)

이제 PV를 같은 batch의 head끼리 concat해준 후 W_O와 곱해서 head간 정보를 섞어준다. 그리고 FFN에서 GEMM으로 차원을 확장했다가 activation을 거쳐서 패턴을 탐지하고 다시 GEMM으로 차원을 축소해준다. 이 Layer를 여러 겹 거친 후 마지막 lm_head 단계에서 가장 마지막으로 생성한 토큰만 slicing한 후 토큰을 정해주면 첫 번째 토큰 생성이 완료되었다!

![30.lm_head_prefill.png](/image/transformer/30_lm_head_prefill.png)

## decode (+ KV cache)

prefill 단계에서는 prompt를 바탕으로 첫 번째 토큰을 생성했다. 이제 이어서 다음 토큰을 생성해보자. 첫 번째 토큰부터 방금 생성한 토큰까지를 인풋으로 받아서 다음 토큰을 생성한다. 그리고 또 처음부터 직전 토큰까지…. 의 반복이다. 여기서 inference와 training의 가장 큰 차이점이 드러난다. training 단계에서는 내가 예측한 결과가 아니라 정답 시퀀스를 바탕으로 모델을 학습시켰다. 그래서 Sequence 길이 만큼의 토큰을 한번에 생성하는 것이 가능했다. 그런데 추론은 다르다. 방금 생성한 토큰이 새로운 인풋이 된다. 추론에서는 생성한 토큰과 다음에 생성할 토큰 사이에 data dependency가 생기고, 그래서 병렬적으로 처리가 안된다.

그렇다면 prefill 단계처럼 계속 S개의 logits을 생성하고 마지막 행만 slicing해서 사용하는 과정을 반복해야 할까? 다행히 그렇지 않다. 우리는 마지막 token만을 생성할 것이다. 

어? 조금 전 prefill에서는 그렇게 하면 중간 Layer의 Key와 Value를 projection할 input이 비어서 문제가 발생했었다. 하지만 이번에는 다르다. 우리는 prefill에서 계산해놓은 데이터가 있다. 이렇게 prefill단계와 이전 토큰의 decode 과정에서 계산해놓은 Layer별 Key와 Value값을 저장해놓고 다음 단계에서 다시 사용하는 것을 KVCache 라고 한다. KVCache가 있어서  우리는 매번 마지막 Query, Key, Value만 생성하고 Layer 마다 필요한 Key 값과 Value값은 다시 사용할 수 있다.

어떻게 하는지는 그림으로 살펴보자. 일단 직전 토큰에 대해 QKV projection을 거쳐 Query, Key, Value값을 뽑아준다.

![31.QKV projection docode.png](/image/transformer/31_QKV_projection_decode.png)

그리고 나서 이전 토큰들의 정보를 KV Cache에서 가져온다. 여기서 정보라고 하면 Key값과 Value값을 말한다. 나는 이미 생성된 토큰을 다시 생성하기 위한 Query(=이전에 사용한 Query)에는 관심이 없다. 다만, 여태나온 토큰들의 정보에는 관심이 있다. 따라서 이전 토큰들의 Key값과 Value값은 여전히 필요하고, 그래서 반복적으로 필요한 값들을 매번 계산하지 않고 KV Cache에 저장 해놓는 것이다. 아래 그림은 N번째 토큰 생성의 attention block을 표현한 그림이다. 파란 테두리 부분은 KV Cache로부터 가져온 값이고, 빨간 테두리 그림은 이번에 QKV projection을 통해 계산한 값이다.

![32. attention nth.png](/image/transformer/32_attention_nth.png)

물론 N+1번째 토큰을 생성할 때는 또 N번째 토큰의 정보가 필요하다. 그래서 KV Cache에 저장된 내용은 계속 커진다. 아래 그림은 N+1번째 토큰 생성의 attention block을 표현한 그림이다. 이 단계에서 cache에는 미래 token의 K/V가 존재하지 않으므로, triangular causal mask는 필요하지 않다.

![33. attention n+1.png](/image/transformer/33_attention_n+1.png)

## prefill vs. decode

이번에는 prefill단계와 decode 단계를 비교해보자. 한 번에 처리되는 토큰의 개수는 Prefill이 S개, Decode가 1개이다. 따라서 prefill에서는 QKV projection, FFN, LM_head 연산이 GEMM 형태로 나타나는 반면, Decode에서는 GEMV 형태로 나타난다. 다음으로는 한번 DRAM에서 로드된 weight가 몇 번 재사용 되는지 살펴보자. prefill에서는 S개 토큰의 hidden state와 weight가 곱해진다. 이 GEMM 연산에서 한번 load된 weight 값은 S번 사용된다. 반면 Decode에서는 1개 토큰의 hidden state와 weight가 곱해지므로 한번 load된 weight 값은 한 번 사용된다. 또한, 앞서 언급했듯이 prefill단계에서는 Key값과 Value값 모두 projection을 통해 값을 계산하고, KV cache에 저장해놓고, decode에서는 KV cache에 저장된 값과 새로 계산한 하나의 Key, Value값을 추가로 KV cache에 추가하며 사용한다.

![34. prefill vs. decode.png](/image/transformer/34_prefill_vs_decode.png)

# Summary

![35. notation.png](/image/transformer/35_notation.png)

| # | 블록 | 입력 shape | 가중치 shape | 출력 shape | 연산 종류 |
| --- | --- | --- | --- | --- | --- |
| **임베딩부** |  |  |  |  |  |
| 0-1 | Token embedding | ids `[B, M]` | `wte [50304, 768]` | `[B, M, 768]` | gather |
| 0-2 | Position embedding | pos `[M]` | `wpe [1024, 768]` → 슬라이스 `[p : p+M]` | `[B, M, 768]` | gather + add |
| **× L (=12) 반복** |  |  |  |  |  |
| 1 | LN1 (pre-norm) | `[B, M, 768]` | γ `[768]`, β `[768]` | `[B, M, 768]` | row reduction + elementwise |
| 2 | QKV proj (`c_attn`) | `[B·M, 768]` | W `[768, 2304]`, b `[2304]` | `[B, M, 2304]` | **GEMM** |
| 3 | split + reshape | `[B, M, 2304]` | — | Q, K, V 각 `[B, 12, M, 64]` | layout transform |
| 4 | KV cache write | K,V `[B, 12, M, 64]` | 버퍼 `[B, 12, S_max, 64]` ×2 | (in-place) | copy |
| 5 | scores QKᵀ | Q `[B,12,M,64]`, K `[B,12,S,64]` | — | `[B, 12, M, S]` | **batched GEMM** (batch=B·h=12) |
| 6 | scale | `[B, 12, M, S]` | 스칼라 1/√64 = 1/8 | 동일 | elementwise |
| 7 | causal mask | `[B, 12, M, S]` | mask `[M, S]` | 동일 | elementwise |
| 8 | softmax | `[B, 12, M, S]` | — | `[B, 12, M, S]` | row-wise reduction |
| 9 | P·V | P `[B,12,M,S]`, V `[B,12,S,64]` | — | `[B, 12, M, 64]` | **batched GEMM** |
| 10 | merge heads | `[B, 12, M, 64]` | — | `[B, M, 768]` | layout transform |
| 11 | out proj (`c_proj`) | `[B·M, 768]` | W `[768, 768]`, b `[768]` | `[B, M, 768]` | **GEMM** |
| 12 | residual add | `[B, M, 768]` | — | `[B, M, 768]` | elementwise |
| 13 | LN2 (pre-norm) | `[B, M, 768]` | γ `[768]`, β `[768]` | `[B, M, 768]` | row reduction + elementwise |
| 14 | FFN up (`c_fc`) | `[B·M, 768]` | W `[768, 3072]`, b `[3072]` | `[B, M, 3072]` | **GEMM** |
| 15 | GELU | `[B, M, 3072]` | — | `[B, M, 3072]` | elementwise |
| 16 | FFN down (`c_proj`) | `[B·M, 3072]` | W `[3072, 768]`, b `[768]` | `[B, M, 768]` | **GEMM** |
| 17 | residual add | `[B, M, 768]` | — | `[B, M, 768]` | elementwise |
| **출력부** |  |  |  |  |  |
| 18 | Final LN | `[B, M, 768]` | γ `[768]`, β `[768]` | `[B, M, 768]` | row reduction + elementwise |
| 19 | (decode) last-token slice | `[B, M, 768]` | — | `[B, 1, 768]` | slice |
| 20 | lm_head | `[B·M, 768]` | `wteᵀ [768, 50304]` (tied) | `[B, M, 50304]` | **GEMM/GEMV** |
