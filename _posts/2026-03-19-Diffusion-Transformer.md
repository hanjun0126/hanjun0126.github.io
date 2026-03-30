---
layout: post
title: "Scalable Diffusion Models with Transformers"
meta: "Springfield"
modified_date: "not complete"
paper: "https://arxiv.org/pdf/2212.09748"
tags: [paper]
---

기존 Diffusion 모델이 U-Net을 backbone으로 사용해서 노이즈를 예측했다면 Diffusion Transformer(DiTs)는 ViT를 이용해서 노이즈를 예측하는 모델이다.

<img src="../assets/post/post3/fig1.png"/>

## Introduction

>   With this work, we aim to demystify the significance of architectural choices in diffusion models<br> and offer empirical baselines for future generative modeling research.

Transformer 모델이 여러 도메인에서 사용되고 있음에 따라, convolutional U-Net 구조를 backbone으로 사용하는 diffusion model 에서의 transformer 적용을 시도한다. DiT 이전에도 Transformer는 diffusion에 사용되었지만, 항상 보조 역할이었고, DiT가 처음으로 diffusion의 backbone을 Transformer로 완전히 대체했다. 이는 LDM이 입력 차원을 줄여서 self-attention 비용을 크게 낮췄기에 Transformer를 diffusion backbone으로 사용하는 것이 현실적으로 가능해졌다.

>   We show that the U-Net inductive bias is not crucial to the performance of diffusion models,<br> and they can be readily replaced with standard designs such as transformers.

또한 U-Net의 inductive bias가 diffusion 모델의 성능에 필수적이지 않으며, 이를 transformer와 같은 표준적인 구조로 쉽게 대체할 수 있음을 보여준다. U-Net의 inductive bias는 이미지의 지역성, 계층 구조, 디테일 보존을 가정하는 설계적 편향이고, DiT는 이걸 제거해도 성능이 유지된다는 걸 보여준 것이다.

## Diffusion Transformers

### DDPM

Diffusion 모델은 원본 데이터 $x_0$ 에 점진적으로 노이즈를 가하는 forward noising process를 가정한다. 이는 $x_0$ 를 한 번에 망가뜨리는 것이 아니라, 시간 단계 $t$ 에 따라 조금씩 노이즈를 섞어서 $x_t$ 를 만드는 과정이다. 

$$q(x_t \mid x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\mathbf{I})$$

$q(x_t \mid x_0)$ 는 원본 $x_0$ 에서 $t$ 번째 noisy 상태 $x_t$ 가 만들어질 조건부 분포이고, $\mathcal{N}(\cdot; \mu, \Sigma)$ 는 평균이 $\mu=\sqrt{\bar{\alpha}_t}x_0$, 공분산이 $\Sigma=(1-\bar{\alpha}_t)\mathbf{I}$ 인 gaussian 분포이다. 즉, $x_t$ 는 원본 이미지의 노이즈가 주입된 버전 $\sqrt{\bar\alpha_t}x_0$ 를 중심으로 하고, 그 주변에 분산 $(1-\bar\alpha_t)$ 만큼의 gaussian 노이즈가 퍼져 있는 형태다. 다음으로 $\tilde\alpha_t$ 는 시간 $t$에서 원본 데이터를 얼마나 남아 있는지를 조절하는 하이퍼파라미터이다. 0과 1사이의 값이며, $t$ 가 커질수록 점점 작아진다. $t$ 가 커질수록 $\sqrt{\bar\alpha_t}$  는 작아져 원본 데이터의 비중이 줄고, $\sqrt{1-\bar\alpha_t}$ 는 커져 노이즈의 비중이 커진다. 첫 번째 수식에서 재매개변수화 기법(reparameterization trick)을 적용하면, 아래의 식을 얻는다.

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,\mathbf{I})$$

Gaussian 분포에서 실제 샘플을 뽑는 방법이다. gaussian 분포 $z \sim \mathcal{N}(\mu,\Sigma)$ 는 표준정규분포 $ \epsilon \sim \mathcal{N}(0,I)$ 를 이용하면 $z=\mu + A\epsilon$ 형태로 쓸 수 있다. $AA^\top=\Sigma$ 이어야 한다. 현재 상황에서는 공분산이 $\Sigma=(1-\bar{\alpha}_t)I$ 이므로, $A=\sqrt{1-\bar{\alpha}_t}\,I$ 이다.

$\bar\alpha_t$ 는 한 단계의 노이즈 비율이 아니라, 0단계부터 $t$ 단계까지 누적된 노이즈이다. DDPM에서는 각 step마다 $\beta_t$ 라는 작은 노이즈 스케줄을 정하고, $\alpha_t = 1-\beta_t$ 라고 놓은 뒤, $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ 로 정의한다. 즉, $\bar\alpha_t$ 는 여러 step을 거치며 원본 신호가 얼마나 살아남았는지를 나타내는 누적 보존율이다. 그래서 한 번에 $x_0\rightarrow x_t$ 를 샘플링할 수 있다.

>   Diffusion models are trained to learn the reverse process that inverts forward process corruptions: $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t), \Sigma_\theta(x_t))$, where neural networks are used to predict the statistics of $p_\theta$.

Diffusion 모델은 forward 과정에서 생긴 노이즈를 되돌리는 reverse 과정을 학습한다. Reverse 과정은 평균 $\mu_\theta(x_t)$, 공분산 $\Sigma_\theta(x_t)$ 인 gaussian 분포로 모델링된다.

>   both $q^*$ and $p_\theta$ are Gaussian, so $D_{KL}$ can be evaluated with the mean and covariance.

$$\mathcal{L}(\theta) = -\log p(x_0 \mid x_1) + \sum_t D_{KL}(q^*(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$$

Negative ELBO 수식으로 Loss가 정의되고, 학습에 영향을 주지 않는 상수 항은 제거됐다.

1.   $-\log p(x_0 \mid x_1)$ 는 Reconstruction term
2.   $D_{KL}(q^* \parallel p_\theta)$ 는 KL term





평균을 직접 예측하지 않고 노이즈 $\epsilon$ 를 예측하는 형태로 바꾸면 $\mathcal L_{\text{simple}} = \Vert\epsilon_\theta(x_t) - \epsilon_t\Vert^2$, 학습은 단순한 MSE 문제로 바뀐다. 공분산까지 학습하려면 단순 MSE가 아니라 전체 KL loss를 써야한다.

-   $\epsilon_\theta$ 는 $\mathcal L_\text{Simple}$ MSE로 학습
-   $\Sigma_\theta$ 는 full $\mathcal L$ 로 학습

학습 후에는 노이즈에서 시작해서 reverse 과정을 반복하며 이미지 생성



### Classifier-free guidance

Classifier-free guidance는 conditional diffusion 모델에서 별도의 classifier 없이도 원하는 조건을 더 강하게 반영하도록 샘플링을 조정하는 방법이다. 기본적인 conditional diffusion 모델은 클래스 라벨이나 텍스트와 같은 조건 $c$ 를 입력으로 받아 $p_\theta(x_{t-1}\mid x_t, c)$ 를 모델링한다. 즉, reverse 과정 자체가 조건에 의존하도록 구성된다. 이때 모델은 노이즈 예측 형태로 $\epsilon_\theta(x_t,c)$를 출력하여, 이는 조건 $c$ 를 만적하는 방향으로의 score, 즉 $\nabla_x\log p(x\mid c)$를 간접적으로 나타낸다.

이 방법은 "좋은 샘플"을 $p(c\mid x)$ 가 높은 샘플로 정의하는 데서 출발한다. 베이즈 정리를 적용하면 

$$\log p(c \mid x) = \log p(x \mid c) - \log p(x) + \log p(c)$$

로 쓸 수 있고, 여기서 $\log p(c)$ 는 $x$ 에 대해 상수이므로 무시할 수 있다. 따라서 샘플링을 유도하는 방향은 $\nabla_x \log p(x \mid c)$ $ - \nabla_x \log p(x)$ 로 표현된다. 이 식은 "조건을 만족하는 방향"에서 "자연스러운 데이터 분포 방향"을 뺸 것으로, 결국 조건을 더 강하게 만족하도록 샘플을 이동시키는 방향을 의미한다.

Diffusion 모델에서는 $\epsilon_\theta(x_t, c)$가 $\nabla_x \log p(x \mid c)$, 그리고 $\epsilon_\theta(x_t, \emptyset)$가 $\epsilon_\theta(x_t, \emptyset)$에 대응한다고 볼 수 있다. 여기서 $\emptyset$ 는 조건을 제거한 경우, 즉 unconditional 모델을 의미한다. 따라서 두 출력을 이용하면 원하는 방향을 직접 구성할 수 있다. 이를 바탕으로 실제 샘플링에서 사용하는 식은

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

로 주어진다. 이 식은 기본적인 방향에 조건 방향의 차이를 $s$배로 더해주는 형태이며, $s>1$ 일수록 조건을 더 강하게 반영하게 된다. $s=1$ 이면 일반적인 조건부 diffusion과 동일하고, $s=0$ 이면 unconditional 모델이 된다.

Classifier-free guidance의 중요한 특징은 별도의 classifer를 필요로 하지 않는다는 점이다. 대신 하나의 diffusion 모델을 학습할 때, 일정 확률로 조건 $c$ 를 제거하고 이를 "null embedding" $\emptyset$ 으로 대체하여 학습한다. 이 방식으로 모델은 동시에 $p(x\mid c)$와 $p(x)$를 모두 학습하게 된다. 결과적으로 두 경우의 출력을 이용해 위와 같은 guidance를 구성할 수 있으며, 추가적인 모델 없이도 조건을 강화할 수 있다.



### Latent diffusion models

Latent Diffusion Models(LDMs)의 아이디어는 고해상도 이미지 공간에서 직접 diffusion을 수행하는 대신, 먼저 이미지를 더 작은 latent 표현으로 압축한 뒤 그 공간에서 diffusion을 수행하는 것이다. 일반적인 diffusion 모델은 입력이 픽셀 공간 $x\in\mathbb R^{H\times W\times3}$ 에 있기 때문에 해상도가 높아질수록 연산량과 메모리 사용량이 급격히 증가한다. 이는 특히 학습과 샘플링 모두에서 큰 병목으로 작용한다.

LDM은 이를 해결하기 위해 두 단계 구조를 사용한다. 첫 번째 단계에서는 autoencoder를 학습하여 이미지 $x$ 를 더 작은 표현 $z=E(x)$ 로 압축한다. 여기서 인코더는 공간 해상도를 크게 줄이면서도 이미지의 중요한 semantic 정보를 유지하도록 학습된다. 두 번째 단계에서는 원본 이미지가 아니라 이 latent 표현 $z$를 대상으로 diffusion 모델을 학습한다. 이때 인코더는 고정된 상태로 유지되며, diffusion 모델은 오직 latent 공간에서의 확률 분포를 학습한다.

생성 과정에서는 먼저 latent 공간에서 noise로부터 시작한다. 즉, $z_T\sim\mathcal N(0,I)$ 에서 시작하여 reverse 과정을 통해 점진적으로 $z_0$ 를 복원한다. 이후 이 latent를 디코더 $D$ 에 통과시켜 최종 이미지 $x=D(z_0)$ 를 얻는다. 따라서 전체 생성 파이프라인은 "노이즈 &rarr; 잠재공간 latent &rarr; 이미지"의 구조를 갖는다.

이 방식의 가장 큰 장점은 계산 효율성이다. Latent 공간은 픽셀 공간보다 훨씬 낮은 차원을 가지기 때문에, diffusion 과정에서 필요한 연산량이 크게 감소한다. 동시에, 단순히 정보를 버리는 압축이 아니라 의미정보를 유지하는 표현을 사용하기 때문에 생성 품질은 크게 손상되지 않는다. 오히려 latent 공간에서는 저수준 픽셀 노이즈가 줄어들고 구조적인 정보가 강조되기 때문에 diffusion 모델이 더 안정적으로 학습되는 경향이 있다.

이러한 latent diffusion 구조 위에 transformer 적용한다. 기존 diffusion 모델들이 주로 U-Net 기반 convolution 구조를 사용했던 것과 달리, 이 논문은 transformer 기반 구조를 latent 공간에 적용한다. 결과적으로 전체 시스템은 convolutional VAE와 transformer 기반 diffusion 모델이 결합된 hybrid 구조를 갖는다.

### Patchify

<img src="../assets/post/post3/fig4.png" style="zoom:50%;"/>

DiT에서 입력으로 사용되는 것은 이미지 자체가 아니라 VAE를 통해 압축된 latent 표현 $z$ 이다. 예를 들어 256x256 크기의 이미지는 32x32x4 형태의 latent로 변환된다. DiT의 첫 번째 단계는 이러한 spatial 형태의 입력을 transformer가 처리할 수 있도록 sequence 형태로 변환하는 patchify 과정이다.

Patchify는 입력을 $p\times p$ 크기의 패치로 나눈 뒤, 각 패치를 벡터로 펼치고 이를 하나의 token으로 변환하는 과정이다. 이로써 전체 입력은 $T$개의 토큰으로 이루어진 시퀀스로 변환되며, 각 토큰은 동일한 차원 $d$ 를 갖는다. 이후 transformer에서 위치 정볼르 활용할 수 있도록 sine-cosine 기반의 위치 임베딩이 모든 토큰에 추가된다.

이때 생성되는 토큰의 개수 $T$ 는 patch 크기 $p$ 에 의해 결정된다, 입력 해상도가 고정된 상황에서 $p$ 가 작아질수록 더 많은 패치가 생성되며, 결과적으로 토큰 수 $T$ 는 $1\over p^2$ 에 비례하여 증가한다. 예를 들어, $p$ 를 절반으로 줄이면 토큰 수는 4배로 증가한다. Transformer의 계산량은 토큰 수에 크게 의존하기 때문에, 이러한 변화는 전체 연산량 Gflops를 최소 4배 이상 증가시키는 결과를 낳는다. 흥미로운 점은 patch 크기 $p$ 는 계산량에는 큰 영향을 주지만 모델의 파라미터 수에는 거의 영향을 주지 않는다는 것이다. 

### DiT architecture

<img src="../assets/post/post3/fig3.png" style="zoom:50%;"/>







## Appendix

### ELBO

Diffusion 모델에서 

1.   Forward: $q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t\mid x_{t-1})$

     원본 $x_0$에서 시작해서 매 단계 노이즈를 더하는 과정

     

2.   Reverse: $p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)$

     Gaussian prior $x_T$ 에서 시작하고, 그다음 reverse transition을 반복해서 $x_0$까지 생성하는 과정

     

이고, 모델의 목표는 $\max_\theta \log p_\theta(x_0)$이다. 여기서 $x_0$는 실제 데이터, 원본 이미지다. 그런데 diffusion 모델은 중간 변수 $x_1, x_2, \dots, x_T$를 거치는 latent variable model이다. 그래서 $x_0$의 확률은 중간 변수들을 적분해서 얻어야 한다.


$$
\begin{aligned}
p_\theta(x_0) &= \int p_\theta(x_{0:T})\,dx_{1:T} \\
&= \int q(x_{1:T}\mid x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T} \\
&= \mathbb E_{q(x_{1:T}\mid x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)} \right]
\end{aligned}
$$




$\log p_\theta(x_0)=\log \int p_\theta(x_{0:T})\,dx_{1:T}$ 를 직접 최적화하기 어렵다. 로그 안에 적분이 있기 때문이다. 중간 단계 전체를 합쳐서 계산해야 해서 다루기 불편하다. 그래서 보조분포 $q(x_{1:T}\mid x_0)$를 도입한다. 양변에 $\log$를 씌워주면, $\log$는 concave 함수이므로 $\log \mathbb E_q[Z] \ge \mathbb E_q[\log Z]$ 성립한다.




$$
\begin{aligned}
\log p_\theta(x_0) &= \log \mathbb E_q \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)} \right] \\
&\ge\mathbb E_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]={\mathcal L_\text{ELBO}}
\end{aligned}
$$




우리는 사실 $\log p_\theta(x_0)$를 키우고 싶다. 그런데 직접 다루기 어려우니, 그보다 작거나 같은 값인 $\mathcal L_{\mathrm{ELBO}}$를 대신 키우는 것이다.

$$\boxed{ \mathcal L_{\mathrm{ELBO}} = \mathbb E_{q(x_{1:T}\mid x_0)} \left[ \log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0) \right] }$$

$\mathcal L_\text{ELBO}$ 에 위에서 정의한 forward와 reverse 과정을 결합분포 형태로 나타내고, forward 과정은 Markov chain이므로,

$$q(x_{1:T}\mid x_0) = q(x_T\mid x_0)\,q(x_{T-1}\mid x_T,x_0)\cdots q(x_1\mid x_{2:T},x_0)$$



$$q(x_{t-1}\mid x_t,x_{t+1:T},x_0)=q(x_{t-1}\mid x_t,x_0)$$



$x_0$가 주어졌을 때 각 단계의 posterior $q(x_{t-1}\mid x_t,x_0)$를 계산할 수 있다. 이 분포는 학습 시점에서는 $x_0$를 알고 있으므로 정답 역할을 한다. 이걸 이용하면 $\mathcal L_\text{ELBO}$를 다음처럼 재구성할 수 있다.


$$
\begin{aligned}
\mathcal L_{\mathrm{ELBO}} &= \mathbb E_q \left[ \log p(x_T) +\sum_{t=1}^{T}\log p_\theta(x_{t-1}\mid x_t) - \sum_{t=1}^{T}\log q(x_t\mid x_{t-1}) \right] \\
&=\mathbb E_q\left[ \log p(x_T)+\{\log p_\theta(x_0\mid x_1) +\sum_{t=2}^{T}\log p_\theta(x_{t-1}\mid x_t)\} - \{\log q(x_T\mid x_0) +\sum_{t=2}^{T}\log q(x_{t-1}\mid x_t,x_0)\} \right] \\
&=\mathbb E_q\left[ \log p(x_T)+ \log p_\theta(x_0\mid x_1)- \log q(x_T\mid x_0)  +\sum_{t=2}^{T}\log p_\theta(x_{t-1}\mid x_t) -\sum_{t=2}^{T}\log q(x_{t-1}\mid x_t,x_0) \right] \\
&=\mathbb E_q\left[\log p_\theta(x_0\mid x_1) + \log {p(x_T)\over q(x_T\mid x_0)} +\sum_{t=2}^{T}\log {p_\theta(x_{t-1}\mid x_t) \over q(x_{t-1}\mid x_t,x_0)} \right] \\
&=\mathbb E_q\left[\log p_\theta(x_0\mid x_1) - \log {q(x_T\mid x_0)\over p(x_T)} -\sum_{t=2}^{T}\log {q(x_{t-1}\mid x_t,x_0) \over p_\theta(x_{t-1}\mid x_t)} \right] \\
&=\mathbb E_q\left[\log p_\theta(x_0\mid x_1) - D_{KL}(q(x_T\mid x_0)\|p(x_T)) -\sum_{t=2}^T D_{KL}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)) \right] \\
\end{aligned}
$$


그리고 loss로 최소화하고 싶으면 양변에 마이너스를 붙이고, $D_{KL}(q(x_T\mid x_0)\|p(x_T))$는 보통 $\theta$와 무관하거나 상수 취급되어 생략되는 경우가 많다.

$$-\mathcal L_{\text{ELBO}} = \mathbb E_q \Big[ -\log p_\theta(x_0\mid x_1) + D_{KL}(q(x_T\mid x_0)\|p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)) \Big]$$

위 식의 각 항은 아래와 같다.

1.   $\log p_\theta(x_0\mid x_1)$

     마지막 단계 복원 항이다. $x_1$에서 원본 $x_0$를 얼마나 잘 복원하는지를 본다.

2.   $D_{KL}(q(x_T\mid x_0)\|p(x_T))$​

     forward가 만든 최종 노이즈 분포가 prior와 얼마나 다른지 측정한다. 보통 $\theta$와 무관해서 상수 취급되는 경우가 많다.

3.   $\sum_{t=2}^T D_{KL}(\cdots)$

     각 timestep에서 모델의 reverse 분포 $p_\theta(x_{t-1}\mid x_t)$ 가 정답 posterior $q(x_{t-1}\mid x_t,x_0)$를 얼마나 잘 따라가는지 측정한다.



### Reparameterization Trick

DDPM forward의 한 스탭은 보통 $q(x_t\mid x_{t-1})=\mathcal N\!\left(\sqrt{\alpha_t}x_{t-1},\,\beta_t I\right),\alpha_t = 1-\beta_t$이다. 이걸 여러 스탭 합치면 closed form이 나온다.

$$q(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\bar\alpha_t}x_0,\,(1-\bar\alpha_t)I\right)$$

$\bar\alpha_t=\prod_{s=1}^t \alpha_s$ 이고, 이 분포에서 샘플링하면

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

$x_t$ 는 원본 $x_0$ 와 표준gaussian 노이즈 $\epsilon$ 의 선형결합이다.

DDPM에서 real posterior는 $q(x_{t-1}\mid x_t,x_0)$ 이다. 이 분포도 gaussian이고, 평균과 분산이 closed form으로 나온다.

$q(x_{t-1}\mid x_t,x_0)=\mathcal N\left(x_{t-1};\tilde\mu_t(x_t,x_0),\,\tilde\beta_t I\right)$

여기서 $\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$, 그리고 $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ 이다.

현재 이 수식은 $\tilde\mu_t(x_t,x_0) = A_t x_0 + B_t x_t$의 형태이다. $$A_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t},B_t=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

그런데 생성 시점에는 $x_0$를 모른다. $x_t$만 안다. 그래서 $x_0$를 직접 넣을 수 없다.

하지만 forward 식



$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$



에서 $x_0$를 풀면



$$x_0=\frac{1}{\sqrt{\bar\alpha_t}} \left(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon\right)$$



이 된다. 즉, $x_0$를 알기 어렵다면 대신 $\epsilon$를 예측해서 $x_0$를 복원할 수 있다.



$$
\tilde\mu_t(x_t,x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t\\
&=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} \cdot \frac{1}{\sqrt{\bar\alpha_t}} \left(x_t-\sqrt{1-\bar\alpha_t}\epsilon\right) + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
\end{aligned}
$$



$\bar\alpha_t=\bar\alpha_{t-1}\alpha_t$이므로 $\frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} = \frac{1}{\sqrt{\alpha_t}}$ 이다. 따라서 첫 항은 $\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)} \left(x_t-\sqrt{1-\bar\alpha_t}\epsilon\right)$ 가 된다.



$$
\begin{aligned}
\tilde\mu_t(x_t,\epsilon) &=\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)}x_t-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\epsilon+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
\\&=\left[ \frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \right]x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\epsilon
\\&=\frac{1}{\sqrt{\alpha_t}} \left( x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon \right)
\end{aligned}
$$




$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t) \right)$$
