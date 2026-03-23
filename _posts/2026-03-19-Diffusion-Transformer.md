---
layout: post
title: "Scalable Diffusion Models with Transformers"
meta: "Springfield"
modified_date: 2026-03-19
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

Gaussian diffusion 모델은 원본 데이터 $x_0$ 에 점진적으로 노이즈를 가하는 forward noising process를 가정한다. 이는 $x_0$ 를 한 번에 망가뜨리는 것이 아니라, 시간 단계 $t$ 에 따라 조금씩 노이즈를 섞어서 $x_t$ 를 만드는 과정이다. 

$$q(x_t \mid x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\mathbf{I})$$

$q(x_t \mid x_0)$ 는 원본 $x_0$ 에서 $t$ 번째 noisy 상태 $x_t$ 가 만들어질 조건부 분포이고, $\mathcal{N}(\cdot; \mu, \Sigma)$ 는 평균이 $\mu=\sqrt{\bar{\alpha}_t}x_0$, 공분산이 $\Sigma=(1-\bar{\alpha}_t)\mathbf{I}$ 인 가우시안 분포이다. 즉, $x_t$ 는 원본 이미지의 노이즈가 주입된 버전 $\sqrt{\bar\alpha_t}x_0$ 를 중심으로 하고, 그 주변에 분산 $(1-\bar\alpha_t)$ 만큼의 가우시안 노이즈가 퍼져 있는 형태다. 다음으로 $\tilde\alpha_t$ 는 시간 $t$에서 원본 데이터를 얼마나 남아 있는지를 조절하는 하이퍼파라미터이다. 0과 1사이의 값이며, $t$ 가 커질수록 점점 작아진다. $t$ 가 커질수록 $\sqrt{\bar\alpha_t}$  는 작아져 원본 데이터의 비중이 줄고, $\sqrt{1-\bar\alpha_t}$ 는 커져 노이즈의 비중이 커진다. 첫 번째 수식에서 재매개변수화 기법(reparameterization trick)을 적용하면, 아래의 식을 얻는다.

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,\mathbf{I})$$

가우시안 분포에서 실제 샘플을 뽑는 방법이다. 가우시안 분포 $z \sim \mathcal{N}(\mu,\Sigma)$ 는 표준정규분포 $ \epsilon \sim \mathcal{N}(0,I)$ 를 이용하면 $z=\mu + A\epsilon$ 형태로 쓸 수 있다. $AA^\top=\Sigma$ 이어야 한다. 현재 상황에서는 공분산이 $\Sigma=(1-\bar{\alpha}_t)I$ 이므로, $A=\sqrt{1-\bar{\alpha}_t}\,I$ 이다.

$\bar\alpha_t$ 는 한 단계의 노이즈 비율이 아니라, 0단계부터 $t$ 단계까지 누적된 노이즈이다. DDPM에서는 각 step마다 $\beta_t$ 라는 작은 노이즈 스케줄을 정하고, $\alpha_t = 1-\beta_t$ 라고 놓은 뒤, $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ 로 정의한다. 즉, $\bar\alpha_t$ 는 여러 step을 거치며 원본 신호가 얼마나 살아남았는지를 나타내는 누적 보존율이다. 그래서 한 번에 $x_0\rightarrow x_t$ 를 샘플링할 수 있다.

>   Diffusion models are trained to learn the reverse process that inverts forward process corruptions: $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t), \Sigma_\theta(x_t))$, where neural networks are used to predict the statistics of $p_\theta$.

Diffusion 모델은 forward 과정에서 생긴 노이즈를 되돌리는 reverse 과정을 학습한다. Reverse 과정은 평균 $\mu_\theta(x_t)$, 공분산 $\Sigma_\theta(x_t)$ 인 가우시안 분포로 모델링된다. 모델은 $x_0$ 의 log-likelihood에 대한 ELBO를 최대화하도록 학습된다.

>   both $q^*$ and $p_\theta$ are Gaussian, so $D_{KL}$ can be evaluated with the mean and covariance.

$$\mathcal{L}(\theta) = -\log p(x_0 \mid x_1) + \sum_t D_{KL}(q^*(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$$

