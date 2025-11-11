---
categories: [paper, other]
description: Knowledge Distillation 논문 정리
tags: [AI]
math: true
---

Paper: [Distilling the Knowledge in a Neural Network][https://arxiv.org/pdf/1503.02531]

Author: Geoffrey Hinton, Oriol Vinyals, Jeff Dean

>   Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a signel model which is much easier to deploy and we develop this approach further using a different compression technique.

## 요약

>   we call "distillation" to transfer the knowledge from the cumbersome model to a small model that is more suitable for deployment.

"Distilling the knowledge in a Neural Network" 논문에서는 Distillation 이라는 복잡한 모델에서 학습된 정보를 사용하기 쉬운 가벼운 모델에 전이하는 방법을 소개한다. 이 방법은 Rich Caruana 와 그의 동료들이 증명한 큰 모델들을 앙상블해서 얻은 정보는 하나의 작은 모델에 전달할 수 있다는 것을 토대로 제안됐다.

모델의 학습 목적은 정답에 대한 확률값을 최대화 하는 것이지만, 학습을 하면서 모델은 정답이 아닌 클래스에도 매우 작지만 확률값을 갖게 한다. 이런 상대적인 확률은 모델의 일반화 성능을 보여주기도 한다. 가령 모델은 BMW 차량을 트럭으로 착각할 경우가 당근으로 착각할 경우보다 많다.

모델은 최종적으로 사용자가 원하는 일반화 성능을 보여야 하지만 보통 모델은 훈련 데이터셋에 최적화 되기 마련이다. 이에 일반화를 잘할 수 있도록 모델을 학습하는 게 좋지만 일반화를 올바르게 하는 정보는 일반적으로 얻을 수 없다. 그렇지만 큰 모델의 정보를 작은 모델에 distilling 하여  작은 모델을 큰 모델처럼 일반화 성능을 갖도록 학습시킬 수 있다.

일반화 성능을 전달하는 분명한 방법은 "soft target"이라 할 수 있는 큰 모델의 클래스 확률 정보를 사용하는 것이다. Soft target 이 큰 entropy를 가진다면 hard target보다 훈련 케이스마다 더 많은 정보를 제공하고 작은 분산을 가지므로 더 적은 데이터로도 학습이 가능하고 더 큰 학습률을 사용할 수 있다.

MNIST 데이터 셋에서 2라는 이미지를 잘 학습된 모델이라면 2라는 정답을 쉽게 도출할 것이다. 하지만 2라는 이미지는 $10^{-6}$ 확률만큼 3처럼, $10^{-7}$확률만큼 7처럼 보이기도 한다. 이런 정보는 매우 가치가 있다. 하지만 cross-entropy 손실 함수에서 0으로 처리되기에 모델 예측에 아무런 영향을 주지 않는다.

Caruana 와 그의 동료들은 이러한 문제를 피하기 위해 작은 모델을 학습할 때 출력 값으로 확률값이 아닌 소프트맥스 값을 사용한다. 이는 큰 모델과 작은 모델 사이에 만들어지는 로짓에 제곱 차를 최소화 한다. 저자들의 ditillation 이라하는 일반화 방법은 큰 모델에서 만들어진 soft target을 작은 모델에서 사용하는 것이다.



## Distillation

신경망 모델들은 대부분 소프트맥스 함수를 사용하여 출력 계층에서 클래스 확률을 구한다. 기본적인 소프트맥스 함수에는 T(temperature) 인자가 1로 세팅되어 있다. Knowledge distillation 이라는 명칭은 지식을 증류하는 것으로 해석할 수 있는데, 증류 과정에서 사용하는 온도를 해당 식에 사용하였다. 온도를 높이면 클래스들간의 더 soft한 확률 분배를 갖게 된다.
$$
q_i={exp(z_i/T)\over\sum_jexp(z_j/T)}
$$
distilled 모델은 




$$
{\partial C\over\partial z_i}={1\over T}({e^{z_i/T}\over\sum_je^{z_j/T}}-{e^{v_i/T}\over\sum_je^{v_j/T}})
$$
