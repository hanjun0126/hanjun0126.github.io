---
categories: [개인공부]
description: diffusion 모델의 reverse process 공부
tags: [공부]
math: true
published: false
---

## Reverse Process

확산 모델의 forward process 는 쉽게 구현할 수 있었다. 하지만 reverse 과정은 쉽지는 않은데, 이는 $x_t$ 노이즈가 낀 이미지를 보고 $x_{t-1}$ 원래 이미지를 맞춰야 하기 때문이다.

### Forward

$$p_\theta(x_{t-1}\vert x_t)=\mathcal N(x_{t-1};\mu_\theta(x_t,t),\sum_\theta(x_t,t))$$

우리의 목표이다.

$$\mu_\theta(x_t,t)={1\over\sqrt\alpha_t}(x_t-{\beta_t\over\sqrt{1-\tilde{\alpha_t}}}\epsilon_\theta(x_t,t))$$

이미지를 복원하는것이 목표가 아닌, 이미지에 껴있는 노이즈를 맞추는게 목표이다.

$$x_t=\sqrt\alpha_t x_{t-1}+\sqrt{1-\alpha_t}\epsilon$$

$$x_{t-1}={1\over\sqrt\alpha_t}(x_t-{\beta_t\over\sqrt{1-\tilde{\alpha_t}}}\epsilon_\theta(x_t,t))+\sigma_t\mathbf z$$

랜덤 노이즈에서 이미지를 생성화는 과정

