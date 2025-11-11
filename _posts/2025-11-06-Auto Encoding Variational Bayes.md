---
categories: [paper]
description: VAE 정리
tags: [공부]
math: true
---

논문: [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)

논문에서는 **연속적인 잠재 변수를 사용하여 효율적인 근사 추론을 수행하기 위해, 변분 하한(ELBO)의 새로운 추정량인 SGVB (Stochastic Gradient Variational Bayes)를 소개하였다.** 이는 재매개변수화 방법을 사용해 ELBO를 계산하는 실용적인 방법으로 분산이 낮아서 안정적으로 학습이 가능하다. 이 추정량은 표준적인 확률적 경사 하강법을 사용해 간단히 미분되고 최적화될 수 있다. 독립 항등 분포 데이터셋과 데이터 포인트당 연속적인 잠재 변수가 있는 경우를 위해, SGVB 추정량을 사용하여 근사 추론 모델을 학습하는, 효율적인 추론 및 학습 알고리즘인 AEVB를 소개한다.

---

<img src="../assets/img/VAE/abstract.png" alt="abstract" style="zoom:40%;" />

## **Abstract**

>   How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets?
>
>   "다루기 힘든 사후 분포(intractable posterior distributions)를 가진 연속적인 잠재 변수가 존재하고 데이터셋이 클 때, 방향성 확률 모델(directed probabilistic models)에서 어떻게 효율적인 추론(inference)과 학습(learning)을 수행할 수 있을까?"

논문에 등장하는 첫 문장이다. 다루기 힘든 사후 분포란 무엇일까? 먼저 사후 분포(Posterior)란 데이터 $x$ 가 주어졌을 때 $z$ 의 분포($p(z\vert x)$)를 말한다. 이는 추론 모델에서 인코더의 목표이다. 이런 사후 분포가 다루기 힘든 이유는 다음과 같다.

### **Intractable posterior distributions**

베이즈 정리에 따르면 $p(z\vert x)={p(x\vert z)p(z)\over p(x)}$ 이다. 여기서 분모인 $p(x)$ 는 $p(x)=\int p(x\vert z)p(z)dz$ 로 계산해야 한다. $z$ 는 연속적인 잠재 변수이기 때문에, 이 적분 $\int \dots dz$ 는 $z$ 의 모든 가능한 경우의 수에 대해 수행해야 하는데, $z$ 가 가질수 있는 값이 무한대이기 때문에 이는 계산적으로 불가능하다. 즉, $p(x)$ 를 모르니, $p(z\vert x)$ 역시 정확하게 계산할 수 없다.

1.   왜 $p(x)=\int p(x\vert z)p(z)dz$ 로 계산해야 하는가?

     $p(x)$ 는 $z$ 가 무엇이든 상관없이, 데이터 $x$ 가 관측될 확률이다. VAE의 가정은 어떤 잠재 변수 $z$ 가 먼저 뽑히고 ($p(z)$), 그 $z$ 를 바탕으로 데이터 $x$ 가 생성된다($p(x\vert z)$). $p(x)$ 는 우리가 관측한 이미지 $x$ 가 생성될 확률이고, 이 $x$ 는 수많은 가능한 $z$ 중 하나에 의해 생성되었을 것이다. 즉, $p(x)$ 를 구할려면, $x$ 를 생성할 수 있는 모든 $z$ 의 경우를 고려하여 그 확률을 전부 더해야 한다. 여기서 $z$ 는 독립적인 변수이다.

     <img src="../assets/img/VAE/다루기 힘든 사후 분포 질문 1.heic" alt="fig1" style="zoom:20%;" />

2.   $z$ 가 가질 수 있는 경우의 수가 무한대가 될 때, 모든 $z$ 에 대해 계산하는 것이 불가능한 이유가 무엇인가?

     $z$ 는 연속 변수이므로 경우의 수가 무한대이다. $z$ 가 1차원만 되어도 무한히 많은 실수가 존재하는데, VAE 에서는 보통 $z$ 가 32차원, 64차원 등 고차원 벡터이므로 경우의 수가 "무한대 중의 무한대"가 된다. 위의 수식의 적분은 $z$ 가 가질 수 있는 값이 무한대이므로 컴퓨터로 직접 계산하는 것은 불가능하다.

이러한 상황 속 VAE의 목표는 효율적인 학습과 추론이 가능하도록 하는 것이다. 학습이란 우리에게 주어진 데이터를 가장 잘 생성해낼 수 있는 모델의 파라미터 $\theta$ 를 찾는 과정이다. 즉, 디코더($p(x\vert z)$)가 실제 같은 이미지를 만들도록 훈련하는 것이다. 추론이란 이미 학습된 모델(디코더)이 있을 때, 특정 데이터 $x$ 가 주어지면 이 데이터를 생성했을 잠재 변수 $z$ 는 무엇인지 알아내는 과정이다. 즉, 인코더의 역할인 $p(z\vert x)$ 를 계산한다.

VAE는 이러한 계산 불가능한 $p(z\vert x)$ 대신, 우리가 다루기 쉬운 간단한 분포(예: 정규분포)인 $q(z\vert x)$로 근사한다. 이 $q$ 가 바로 VAE의 인코더이고, 이러한 방법을 변분 추론(Variational Inference)이라고 한다. 이를 위해 $p(z\vert x)$ 와 $q(z\vert x)$ 를 가깝게 만드는 동시에 $p(x)$를 최대화하는 ELBO(Evidence Lower Bound)라는 목적함수를 사용하게 된다.

### **Large Datasets**

다음으로 중요하게 볼 점은 "Large Datasets", 즉 대규모 데이터셋에서 어떻게 학습할 것인가? 데이터가 많은 경우 전통적인 통계적 학습 방법(예: MCMC, EM알고리즘)은 너무 느려서 현실적으로 사용할 수 없다. 따라서 우리는 전체 데이터를 조금씩 나눠서 처리하는 SGD 기반의 최적화를 사용해야 한다. 하지만 $q(z\vert x)$ 에서 $z$ 를 샘플링하는 과정은 "확률적"이라 미분이 불가능하다. 이에 재매개변수화 방법(Reparameterization Trick)을 사용해 이 샘플링 과정을 미분 가능한 연산으로 분리해낸다.

결론적으로, 이 첫 문장은 "우리는 이런 심각한 문제가 있는데, 이 문제를 풀기 위해 Variational Inference(변분 추론)와 Autoencoder(신경망)를 SGD(효율성)로 학습할 수 있는 VAE를 제안한다.





## **Introduction**

Variational bayesian (VB) 방식은 모델이 조금만 복잡해져도 $q(z\vert x)$ 를 최적화하는 공식을 손으로 유도하는 것 자체가 불가능하거나, 데이터 하나하나마다 비싼 최적화 계산을 반복해야해서 대규모 데이터셋에 비효율적이었다. 논문은 이 문제들을 해결하기 위해 AEVB(Auto-Encodding VB)라는 새로운 알고리즘을 제안하며, 이 알고리즘의 핵심 기여는 두 가지이다.

1.   **재매개변수화 방법(Reparameterization Trick)**

     $q(z\vert x)$ 에서 $z$ 를 샘플링하는 확률적인 과정을, 미분 가능한 결정론적 과정으로 분리했다. ($z=\mu +\sigma *\varepsilon$) 이 방법 덕분에 모델 전체가 미분 가능해졌다. 복잡한 공식을 손으로 풀 필요 없이 SGD만으로 모델 전체를 한 번에 학습시킬 수 있게 되었다. 이것을 SGVB 추정량이라고 한다.

2.   **Recognition Model 도입**

     $x$ 가 주어지면 $z$ 의 분포 ($q(z\vert x)$) 를 즉시 출력하는 신경망(인코더)을 사용했다. 데이터 하나마다 $z$ 를 따로 계산하는 비싼 반복 작업을 없애기 위해서이다. 

## **상황**

연속적이거나 이산적인 변수 $x$ 의 $N$ 개 i.i.d(독립 항등 분포) 샘플로 구성된 데이터셋 $X={\{x^{i}\}_{i=1}^N}$ 가 있다. 이 데이터들은 우리가 알기 어려운 연속적인 잠재 변수 $z$ 를 포함하는 어떤 랜덤 프로세스에 의해 생성된다고 가정한다.

1.   사전 분포(prior distribution) $p_{\theta^*}(z)$ 로부터 $z^i$ 가 생성된다.
2.   조건부 분포(conditional distribution) $p_{\theta^*}(x\vert z)$ 로부터 $x^i$ 가 생성된다.

$\theta^\ast$ 는 우리가 모르는, 세상을 완벽하게 설명하는 진짜 파라미터(true parameter)를 의미한다. 진짜 분포 $p_{\theta^\ast}$ 를 모르는 상황속에서 우리의 목표는 신경망 같은 모델 $p_\theta$ 을 사용하여 신경망의 가중치 $\theta$ 를 $\theta^\ast$ 와 비슷하게 학습시키는 것이다. 이러한 가정이 있을 때, VAE/AEVB 는 위에서 언급한 다음 두 가지 조건에서도 잘 작동하는 알고리즘이다.

1.   Intractability
2.   A large dataset

## VAE의 목표

그렇다면 **VAE 가 달성하고자 하는 목표**는 무엇인가?

1.   파라미터 $\theta$ 에 대한 효율적인 근사 최대 가능도 또는 최대 사후 확률 추정이다. 이 파라미터는 숨겨진 랜덤 프로세스를 모방하여 실제 데이터와 유사한 인공적인 데이터를 생성할 수 있게 해준다.
2.   관찰된 값 $x$ 가 주어졌을 때, 잠재 변수 $z$ 에 대한 효율적인 근사 사후 추론이다. $p(z\vert x)$ 는 계산 불가능하므로, VAE의 인코더 $q_\phi(z\vert x)$ 를 사용해 $p(z\vert x)$ 를 근사해야 한다.
3.   변수 $x$ 에 대한 효율적인 근사 주변 추론(marginal inference)이다. 이는 $x$ 에 대한 사전 분포가 요구되는 모든 종류의 추론 작업을 수행할 수 있게 해준다. 컴퓨터 비전의 일반적인 응용 분야로는 이미지 denoising, inpainting, super-resolution 이 있다. 

이러한 목표를 달설하기 위해 다루기 힘든 실제 사후 분포에 대한 근사치를 표현하는 인식 모델(recognition model) $q_\phi(z\vert x)$ 을 사용한다. 이 인식 모델의 파라미터 $\phi$ 는 생성 모델의 파라미터 $\theta$ 와 함께 학습된다. 즉, 인식 모델(인코더)과 생성 모델(디코더)의 파라미터 $\phi$ 와 $\theta$ 를 하나의 목적 함수를 이용해 동시에 최적화하게 된다. 추가적으로 논문에서는 인식 모델을 확률적 인코더라고도 부르는데, 모델이 데이터 포인트 $x$ 가 주어졌을 때, 그 데이터 $x$ 가 생성되었을 법한 $z$ 의 가능한 값들에 대한 분포를 생성하기 때문이다. 비슷한 맥락에서 $z$ 가 주어졌을 때, 그에 해당하는 $x$ 의 가능한 값들에 대한 분포를 생성하는 생성 모델은 확률적 디코더라고 부른다.

## ELBO(Evidence Lower Bound)

전체 데이터의 주변 가능도(marginal likelihood)는 개별 데이터 포인트의 주변 가능도의 합으로 구성된다. 

$$\log p_\theta(x^1,\cdots,x^N) = \sum_{i=1}^N\log p_\theta(x^i)$$

데이터셋 전체가 나타날 확률 $p(X)$ 는 각 데이터가 독립적이라고 가정했기 때문에 $p(x^1)\times p(x^2)\times\cdots$ 이다. 로그를 씌우면 곱셈이 덧셈으로 바뀌게 된다. 우리의 최종 목표는 이 전체 로그 가능도를 최대화하는 것이다.

$$\log p_\theta(x^i) = D_{KL}(q_\phi(z|x^i)||p_\theta(z|x^i))+\mathcal{L}(\theta,\phi;x^i)\quad(1)$$

$$\log p_\theta(x^i) = D_{KL}(\text{인코더}\|\text{실제 사후 분포})+\mathcal{L}(\text{ELBO})$$



1.   $D_{KL}(q_\phi(z\vert x^i)\Vert p_\theta(z\vert x^i))$

     우리의 근사 모델(인코더) $q_\phi$ 와 계산 불가능한 진짜 사후 분포 $p_\theta$ 사이의 거리이다. 이 값은 $q_\phi$ 와 $p_\theta$ 와 얼마나 다른지를 측정하며, 항상 0보다 크거나 같다.

2.   $\mathcal{L}(\theta,\phi;x)$ : ELBO



### $\log p_\theta(x)$ 의 유도 과정

우리는 인코더를 실제 사후 분포와 가깝게 하는 것이 목표이다. KL Divergence에 따르면

$$D_{KL}(q\vert p)=\mathbb{E}_q[\log q(z\vert x) - \log p(z\vert x)]$$

이다. 여기서 $q$ 는 $q_\phi(z\vert x)$ $p$ 는 $p_\theta(z\vert x)$ 이다. 베이즈 정리 $p(z\vert x)={p(z,x)\over p(x)}$ 를 이용해 $\log p_\theta(z\vert x)$ 를 $\log p_\theta(x,z)-\log p_\theta(x)$ 로 바꾼다.

$$D_{KL}(q\Vert p) = \mathbb{E}_{q}[\log q(z\vert x) - (\log p_{\theta}(x, z) - \log p_{\theta}(x))]$$

$$D_{KL}(q\Vert p) = \mathbb{E}_{q}[\log q(z\vert x) - \log p_{\theta}(x,z) + \log p_{\theta}(x)]$$

$\mathbb E_{q}[\cdot]$ 는 $q$ 의 변수인 $z$ 에 대한 기댓값이다. $\log p_{\theta}(x)$ 는 $z$ 와 관계없는 상수이므로 기댓값 밖으로 나올 수 있다.

$$D_{KL}(q\Vert p) = \mathbb{E}_{q}[\log q(z\vert x) - \log p_{\theta}(x,z)] + \log p_{\theta}(x)$$

이제 수식을 $\log p_{\theta}(x)$에 대해 식을 정리하면 아래와 같다.

$$\log p_{\theta}(x) = D_{KL}(q\Vert p) - \mathbb{E}_{q}[\log q(z\vert x) - \log p_{\theta}(x, z)]$$

$$\log p_{\theta}(x) = D_{KL}(q\Vert p) + \mathbb{E}_{q}[\log p_{\theta}(x,z) - \log q(z\vert x)]$$

여기서 두 번째 항, $\mathbb E_{q}[\log p_{\theta}(x, z) - \log q(z\vert x)]$ 를 위에서 우리는 **$\mathcal{L}(\theta, \phi;x)$ (ELBO)**라고 정의하였다. 따라서 $\log p_{\theta}(x) = D_{KL}(q_{\phi}(z\vert x)\Vert p_{\theta}(z\vert x)) + \mathcal{L}(\theta, \phi; x)$ 가 되어 [수식 1]이 완성된다.



### ELBO

$\log p(x) = D_{KL} + \mathcal L$ 에서 $D_{KL}$ 은 0보다 크거나 같다. 따라서 $\log p(x)\geq\mathcal L$ 라는 부등식이 항상 성립한다. 즉, $\mathcal L$ 은 $\log p(x)$ 의 하한선(Lower Bound)이 된다. 이것이 VAE의 핵심이다. 우리는 계산 불가능한 $\log p(x)$ 를 직접 최대화할 수 없고, $D_{KL}$역시 $p(z\vert x)$ 를 포함하므로 계산 불가능하다. 하지만 $\mathcal L$ 은 계산 가능하기에 $\log p(x)$ 대신 그 하한선인 $\mathcal L$ 을 최대화하는 것이다. 우리의 목표는 데이터가 나타날 확률 $p(x)$ 가 높기를 바라기 때문이다. $\mathcal L$ 을 높이면 $\log p(x)$ 도 함께 높아지는 효과(와 $D_{KL}$ 이 0에 가까워지는 효과)를 얻을 수 있다.

$\text{ELBO}=\mathbb E_{q}[\log p_{\theta}(x,z) - \log q(z\vert x)]$ 의 식을 우리는 아래와 같이 재구성할 수 있다.

$$\mathcal L(\theta, \phi; x^i) = -D_{KL}(q_{\phi}(z\vert x^i)\Vert p_{\theta}(z)) + \mathbb E_{q_{\phi}(z\vert x^i)}[\log p_{\theta}(x^i\vert z)]\quad(2)$$

$$\mathcal{L} = -D_{KL}(\text{인코더} || \text{사전 분포}) + \text{기대 재구성 손실}$$

1.   $-D_{KL}(q_{\phi}(z\vert x)\Vert p_{\theta}(z))$ (Regularization Term, 정규화 항):

     인코더 $q_{\phi}(z\vert x)$ ($x$ 로부터 $z$ 를 만든 분포)가 사전 분포 $p_{\theta}(z)$ (우리가 가정한 $z$ 의 분포, 보통 $\mathcal{N}(0, 1)$)와 "비슷해지도록" 강제하는 항이다. ELBO($\mathcal{L}$)를 최대화하려면, $-D_{KL}$을 최대화, 즉 $D_{KL}$은 **최소화**해야 한다.

2.   $\mathbb E_{q_{\phi}(z\vert x)}[\log p_{\theta}(x\vert z)]$ (Reconstruction Term, 재구성 항):

     $q_{\phi}$ 가 $z$ 를 생성($z\sim q_{\phi}(z\vert x)$)하고, 그 $z$ 를 디코더 $p_{\theta}(x\vert z)$에 넣었을 때, 원본 $x$ 가 다시 나올 로그 확률의 기댓값이다. "인코더-디코더" 구조에서 원본 $x$ 를 얼마나 잘 복원하는지를 나타낸다. 이 값을 최대화해야 한다.

---

ELBO의 재구성 과정은 결합 확률을 조건부 확률로 바꾸는 과정에서부터 시작한다. $p(A,B)=p(A\vert B)p(B)$ 이므로 $\log p_\theta(x,z)=\log p_\theta(x\vert z) + \log p_\theta(z)$ 이다. 이것을 기존 ELBO 식에 대입하여 다음과 같이 구성할 수 있다.

$$\mathcal L =\mathbb E_q[-\log q(z\vert x)+(\log p_\theta(x\vert z) + \log p_\theta(z))]$$

$$\mathcal L =\mathbb E_q[\log p_\theta(z)-\log q(z\vert x)+\log p_\theta(x\vert z)]$$

$$\mathcal L =\mathbb E_q[\log p_\theta(z)-\log q(z\vert x)] +\mathbb E_q[\log p_\theta(x\vert z)]$$

$$\mathcal L =-\mathbb E_q[\log q(z\vert x)-\log p_\theta(z)] +\mathbb E_q[\log p_\theta(x\vert z)]$$

여기서 우변의 첫 번째 항은 $-D_{KL}(q\Vert p)$ 이므로, ELBO 식은 수식(2)와 같이 쓸 수 있다.

---

우리는 이 ELBO을 파라미터 인코더($\phi$) 와 디코더($\theta$) 둘 다에 대해 미분하고 최적화하기를 원한다. $\mathcal L$ 은 인코더와 디코더 모두에 의해 결정되므로 $\mathcal L$ 을 $\phi$ 로 미분해서 인코더를 업데이트하고, $\theta$ 로 미분해서 디코더를 업데이트해야 한다. 이 과정에서 사용되는 것이 경사 상승법이다. 다만 $\theta$ 에 대한 미분은 간단하지만 $\phi$ 에 대한 미분은 어렵다. 왜냐하면 미분하려는 파라미터 $\phi$ 가 샘플링($z\sim q_\phi(z\vert x)$) 을 수행하는 분포 자체에 들어있기 때문이다. 이런 문제에 대한 일반적인 몬테카를로 그래디언트 추정량은 다음과 같다.

$$\nabla_{\phi} \mathbb{E}_{q_{\phi}(z)}[f(z)] = \mathbb{E}_{q_{\phi}(z)}[f(z)\nabla_{\phi} \log q_{\phi}(z)] \approx \frac{1}{L} \sum_{l=1}^L f(z)\nabla_{\phi} \log q_{\phi}(z^l)\quad \text{where}\quad z^l \sim q_{\phi}(z\vert x^i)\quad(3)$$



### 몬테카를로 그래디언트 추정량

ELBO의 재구성 항만 보면

$$\mathcal L \text{의 일부}=\mathbb E_{q_\phi(z\vert x)}[\log p_\theta(x\vert z)]$$

$q_\phi(z\vert x)$: 인코더 파라미터 $\phi$ 를 가진다.

$f(z)=\log p_\theta(x\vert z)$: 디코더가 $z$ 로 $x$ 를 얼마나 잘 복원하는지 나타낸다.

우리는 인코더를 학습시키기 위해 재구성 항을 $\phi$ 로 미분해야 한다. $\nabla_\phi\mathbb E_{q_\phi}[f(z)]$ 처럼 미분하게 되면 미분하려는 파라미터 $\phi$ 가 기대값을 계산하는 분포 $q_\phi$ 자체에 들어있다. $\mathbb E_{q_{\phi}}[f(z)]$ 는 수학적으로 $\int q_\phi(z)f(z)dz$ 이다.

즉, $\nabla_{\phi} \mathbb E_{q_{\phi}}[f(z)]$ 는 $\nabla_\phi\int q_\phi(z)f(z)dz$ 가 된다. 미분 기호 $\nabla$ 를 적분 안으로 넣을 수는 있지만, $f(z)$ 안으로 그냥 넣을 수는 없다. 가장 큰 문제는, $z$ 가 $q_\phi$ 에서 샘플링 된다는 것이다. 샘플링 과정은 그 자체로 미분이 불가능해서 그래디언트가 $\phi$ 에서 $z$ 를 거쳐 $f(z)$ 로 흘러 들어갈 수 없는 끊어진 상태가 된다.

이 끊어진 그래디언트를 수학적으로 연결해주는 방법이 바로 Log-Derivative 방법이다.

$$\nabla_\phi\mathbb E_{q_\phi(z)}[f(z)]=\mathbb E_{q_\phi(z)}[f(z)\nabla_\phi\log q_\phi(z)]]$$

$$\nabla_\phi\mathbb E_{q_\phi(z)}[f(z)]=\int q_\phi(z)[f(z)\nabla_\phi\log q_\phi(z)]dz$$

위에서 $\nabla_{\phi} \mathbb E_{q_{\phi}}[f(z)]$ 는 $\nabla_\phi\int q_\phi(z)f(z)dz$ 이라 했다. 미분 기호를 안으로 넣으면 $\int(\nabla_\phi q_\phi(z))f(z)dz$ 이다.

$\nabla_\phi\log q_\phi(z)={1\over q_\phi(z)}\nabla_\phi q_\phi(z)$ 이므로, $\nabla_\phi q_\phi(z)$ 는 $q_\phi(z)\nabla_\phi\log q_\phi(z)$ 이다.

즉, $\int(\nabla_\phi q_\phi(z))f(z)dz$ 는 $\int(q_\phi(z)\nabla_\phi\log q_\phi(z))f(z)dz$ 이고, 이를 기댓값 형태로 변형한다. 이렇게 수식을 유도하면 $\nabla$ 가 $\mathbb E[\cdots]$ 안으로 들어오므로 $\mathbb E_{q_{\phi}}[\text{어떤 값}]$ 을 $q_\phi$ 에서 $z$ 를 샘플링한 뒤, 어떤 값을 계산하여 평균하는 것으로 근사할 수 있게 된다.

이제 우리는 그래디언트를 수식(3)과 같이 근사할 수 있다. 이것이 바로 몬테카를로 추정량이다. $L$ 개의 $z^l$ 을 인코더에서 뽑는다. 각 $z^l$ 에 대해 $f(z^l)$ 과 $\nabla_\phi\log q_{\phi}(z^l)$ 을 계산해서 곱한다. 이 값들을 평균낸 값이 $\phi$ 에 대한 그래디언트 신호로 사용하여 SGD로 인코더를 업데이트한다.

다만 이 추정량은 매우 높은 분산을 보이며 비실용적이다. 수식에서는 $f(z)$ 와 $\nabla_\phi\log q_{\phi}(z^l)$ 를 곱한다. $f(z)$ 는 복원 확률 $\nabla_\phi\log {q_\phi}(z)$ 는 민감도이다. 우리가 $\phi$ 에 대해서 미분을 하게 되면, $\log q_\phi(z)$ 의 값은 크게 바뀔 수 있다. 그렇기에 기울기가 폭발할 수 있다. 이런 문제를 해결하기 위한 것이 재매개변수화 방법이다.



### Reparameterization Trick

우리는 $q_\phi(z\vert x)$ ($x$ 에 의존하는) 형태의 근사 사후 분포를 가정하지만, 이 기술은 $x$ 에만 의존하지 않는 $q_\phi(z)$ 에도 적용될 수 있다. 근사 사후 분포 $q_\phi(z\vert x)$에 대해 우리는 확률 변수 $\tilde z \sim q_\phi(z\vert x)$ 를 노이즈 변수 $\epsilon$ 의 미분 가능한 변환 $g_\phi(\epsilon, x)$ 를 사용하여 재매개변수화 할 수 있다. 

$$\tilde z =g_\phi(\epsilon, x)\quad\text{with}\quad \epsilon\sim p(\epsilon)$$ 

$q_\phi(z\vert x)$ 가 정규분포 $\mathcal N(\mu, \sigma^2)$ 라고 가정해보자. 여기서는 정규 분포를 따른다고 가정했지만, 정규 분포 이외에도 베타 분포, 라플라스 분포 등, 우리가 선택한 그 분포에 대해 재매개변수화가 가능하다면 어떤 분포든 상관없다. 인코더는 $x$ 를 받아 $\mu$ 와 $\sigma$ 를 출력한다. $p(\epsilon)$ 이 표준 정규 분포(표준 정규 분포를 사용하는 이유는 수학적 편리함 때문)를 따를 때,

$$g_\phi(\epsilon, x) = \mu + \sigma\cdot\epsilon$$ 

이고, 최종적으로 $z: \tilde z=\mu_\phi(x)+\sigma_\phi(x)\cdot\epsilon$ 이다. 보시다시피 $z$ 는 $\mu$ 와 $\sigma$ 를 따르는 정규분포가 되지만, $\phi$ 에서 $z$ 로 가는 그래디언트 경로는 $\mu_{\phi}$ 와 $\sigma_{\phi}$ 를 통해 $\epsilon$ 과의 덧셈/곱셈 연산으로 완전히 미분 가능해졌다. 즉, 기존의 정규 분포에서 샘플링하는 과정이 표준 정규 분포를 이용하여 $z$ 를 구하는 과정으로 바뀐 것이다. 따라서 몬테카를로 추정치를 다음과 같이 구성할 수 있다.

$$\mathbb E_{q\phi(z\vert x^i)}[f(z)]=\mathbb E_{p(\epsilon)}[f(g_\phi(\epsilon,x^i))]\simeq {1\over L}\sum_{l=1}^Lf(g_\phi(\epsilon^l,x^i))\quad \text{where}\quad \epsilon^l\sim p(\epsilon)$$

위 식을 보기 편하게 작성하면 아래와 같다.

$$\mathbb E_{\mathcal N(z;\mu,\sigma^2)}[f(z)]=\mathbb E_{\mathcal N(\epsilon;0,1)}[f(\mu+\sigma\times\epsilon)]\simeq{1\over L}\sum_{l=1}^L(f(\mu+\sigma\times\epsilon^l))\quad \text{where}\quad \epsilon^l\sim\mathcal N(0,1)$$

$q_\phi$ 에 대한 기댓값은 $p_\epsilon$ 에 대한 기댓값으로 바뀌고, 이는 $p(\epsilon)$ 에서 $L$ 번 샘플링하여 평균을 내는 것으로 근사할 수 있다. 

우리는 재매개변수화 방법을 ELBO에 적용하여, 우리의 일반적인 SGVB 추정량 $\mathcal {\tilde L}^A \approx \mathcal L$ 을 도출한다.

$$\mathcal{\tilde L}^A (\theta,\phi,x^i)={1\over L}\sum_{l=1}^L(\log p_\theta(x^i, z^{(i,l)})-\log q_\phi(z^{(i,l)}\vert x^i))$$

$$\text{where}\quad z^{(i,l)}=g_\phi(\epsilon^{(i,l)}, x^i)\quad\text{and}\quad \epsilon^l\sim p(\epsilon)$$

ELBO의 KL 항($-D_{KL}(q_\phi\Vert p_\theta)$) 은 해석적으로 적분(analytically integrate)될 수 있으므로, 샘플링을 통한 추정이 필요한 것은 기대 재구성 오차 항($\mathbb E_{q_\phi}[\log p_\theta(x\vert z)]$)뿐이다. 해석적으로 적분 될 수 있다는 것은, 수학 공식으로 한 번에 값을 계산할 수 있다는 의미이다. 인코더는 정규 분포를 따른다고 가정했고, 사전 분포는 표준 정규 분포를 따른다. 그러므로 가우시안 분포 두 개 사이의 KL-Divergence 는 아래 수식과 같다.

$$D_{KL}(\mathcal N(\mu, \sigma^2)\Vert \mathcal N(0,1)) = 0.5\times \sum(\sigma^2+\mu^2-1-\log(\sigma^2))$$ 

인코더가 $\mu$ 와 $\sigma$ 만 출력하면, KL 항은 샘플링 없이 이 공식을 통해 정확한 값을 바로 계산할 수 있다. KL-Divergence 항은 $\phi$ 인코더 파라미터를 정규화하는 것으로 해석될 수 있으며, 근사 사후 분포가 사전 분포에 가까워지도록 하는 것을 목표로 한다. 이는 근사 사후 분포를 데이터셋에 과적합 되는 것을 방지해준다.

따라서 KL 항은 정확한 공식으로 계산하고, 재구성 오차 항만을 샘플링하여 새롭게 추정하도록 한다.

$$\mathcal {\tilde L}^B(\theta,\phi, x^i)=-D_{KL}(q_\phi(z\vert x^i)\Vert p_\theta(z))+{1\over L}\sum_{l=1}^L(\log p_\theta(x^i\vert z^{(i,l))})$$

$$\text{where}\quad z^{(i,l)}=g_\phi(\epsilon^{(i,l)}, x^i)\quad\text{and}\quad \epsilon^l\sim p(\epsilon)\quad(7)$$

확률적인 부분이 절반으로 줄었기 때문에, $\epsilon$ 이 바뀔 때마다 값이 덜 흔들린다. 기본 방식보다 분산이 낮다.

$$\mathcal {\tilde L}^A=\text{[확률적 샘플링 항 1]}-\text{[확률적 샘플링 항 2]}$$

$$\mathcal {\tilde L}^B=-[\text{해석적 KL 항}] + [\text{샘플링된 재구성 항의 평균}]$$

재구성 항의 계산은 다음과 같다.

1.   $L$ 개의 $\epsilon^l$ 을 $\mathcal N(0,1)$ 에서 샘플링한다.
2.   $L$ 개의 $z^{(i,l)}=\mu + \sigma \times \epsilon$ 을 계산한다.
3.   $L$ 개의 $z$ 를 디코더에 넣어, $L$ 개의 $\log p_\theta(x^i\vert z^{(i,j)})$ (복원 확률)을 계산하고 단순 평균한다.

우리는 ELBO를 최대화해야 하므로, 손실함수 $-\mathcal {\tilde L}^B$ 를 최소화해야 한다.



### 미니 배치 훈련

지금까지는 $x^i$ 데이터 1개에 대한 손실 $\mathcal L$ 을 다뤘다. 이제 이걸 미니배치($M$ 개)와 전체 데이터셋($N$ 개)으로 확장한다.

$$\mathcal L(\theta,\phi; X)\simeq\mathcal {\tilde L}^M(\theta,\phi;X^M)={N\over M}\sum_{i=1}^M\mathcal (\theta,\phi; x^i)$$

전체 데이터셋의 ELBO ($\mathcal L(X)$) 는 미니배치의 평균 ELBO (${1\over M}\sum\mathcal {\tilde L}$)에 스케일링($N$)한 것으로 근사할 수 있다. 역서 미니배치 $X^M$ 은 $N$ 개의 데이터포인트가 있는 전체 데이터셋에서 무작위로 추출된 샘플이다. 저자들은 실험을 통해, 미니배치 크기가 충분히 크다면, 데이터 포인트당 샘플 수 $L$ 은 1로 설정해도 된다는 것을 발견했다.

우리가 만든 최종 미니배치 손실 함수 $\mathcal {\tilde L}^M$ 은 미분 가능하다. KL 항은 공식으로 미분하고, 재구성 항은 재매개변수화 트릭으로 미분 가능하다. 따라서 이 손실을 $\phi$ , $\theta$ 로 미분해서 $\nabla$ 를 구하고, 이를 옵티마이저에 넘겨주면 아래의 Algorithm 1처럼 파라미터가 자동으로 업데이토 되도록 할 수 있다.



### Algorithm



<img src="../assets/img/VAE/algorithm.png" alt="algorithm1" style="zoom:40%;" />



### Variational Auto-Encoder

잠재 변수에 대한 사전 분포 $p_\theta(z)$ 를 표준 정규 분포라, 디코더 $p_\theta(x\vert z)$ 는 정규 분포라 가정하자. 디코더 분포의 파라미터는 $z$ 로부터 MLP를 통해 계산된다. 실제 사후 분포 $p_\theta(z\vert x)$ 는 계산 불가하므로, 인코더 $q_\phi(z\vert x)$ 역시 대각 공분산을 갖는 정규 분포라 가정한다. 대각 공분산을 갖는 것은 $z$ 의 각 차원이 서로 독립이라는 의미이다.

$$\log q_\phi(z^i\vert x^i)=\log\mathcal N(z;\mu^i\sigma^{2(i)}\mathbf I)\quad (9)$$

인코더 신경망은 $x^i$ 를 입력받아 $\mu^i$ 벡터와 $\sigma^i$ 벡터를 출력한다. 이전에 설명한 재매개변수화 기법 ($z=\mu+\sigma\odot\epsilon$) 을 사용하여 $z$ 를 샘플링한다. $\odot$ 는 $J$ 개의 각 차원별로 독립적으로 계산(element-wise product)하는 기호이다. 현재 모델의 사전분포와 인코더는 정규 분포이므로, 우리는 $D_{KL}$항을 샘플링 없이 계산할 수 있는 수식(7)의 추정량($\mathcal L^B$)을 사용할 수 있다. 이 덕분에 VAE는 분산이 낮은 손실 함수를 사용할 수 있어 학습이 매우 안정적이다.

$$\mathcal L(\theta,\phi;x^i)\approx{1\over2}\sum_{j=1}^J(1+\log((\sigma_j^i)^2)-(\mu_j^i)^2-(\sigma_j^i)^2)+{1\over L}\sum_{l=1}^L\log p_\theta(x^i\vert z^{(i,l)})$$

$$\text{where}\quad z^{(i,l)}=\mu^i+\sigma^i\odot\epsilon^l\quad\text{and}\quad\epsilon^l\sim\mathcal N(0,\mathbf I)\quad(10)$$

최종적으로 모델에 대한 최종 목적 함수(ELBO 추정량)은 KL 항 공식과 재구성 항 샘플링의 합이다. 이것이 실제 VAE에서 사용되는 최종 손실 함수이다.

1.   첫 번째 항

     $-D_{KL}(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,1))$ 를 수학 공식으로 계산한 결과이다. $J$ 는 $z$ 의 차원수이다. 이 항은 샘플링 $\epsilon$ 이 필요 없으며, 인코더가 출력한 $\mu$, $\sigma$ 만으로 즉시 계산된다.

2.   두 번째 항

     재구성 손실이다. 이 항은 $z$ 를 반드시 $L$ 번 샘플링 해야만 계산할 수 있다. 데이터 종류에 따라 BCE 또는 MSE 손실로 계산된다.