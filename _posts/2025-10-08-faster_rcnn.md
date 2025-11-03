---
categories: [paper, computer vision]
description: Faster R-CNN 정리
tags: [CV]
math: true
---

Paper: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

>   In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position.

Object Detection 과제는 객체의 위치를 추정하는 region proposal 알고리즘에 의존한다. SPPnet이나 Fast R-CNN 같은 모델 덕분에 탐지 네트워크 자체의 실행 시간은 줄었지만, 이 때문에 오히려 영역을 제안하는 과정이 속도를 저해하는 주된 원인(병목 현상, bottleneck)이 되었다. 논문에서는 이미지 전체의 컨볼루션 특징(feature)을 탐지 네트워크와 공유하는 '영역 제안 네트워크(RPN)'를 소개한다. 이를 통해 거의 공짜로(cost-free) 영역 제안을 할 수 있다. RPN은 모든 위치에서 객체의 경계(bounds, 네모 박스의 위치와 크기)와 객체일 가능성(objectness score)을 동시에 예측하는 완전 컨볼루션 네트워크(Fully Convolutional Network)이다.



## 논문 배경

>   Recent advances in object detection are driven by the success of region proposal methods and region-based convolutional neural networks (R-CNNs).

최근의 Object Detection 기술 발전은 영역 제안(region proposal) 방법들과 영역 기반의 컨볼루션 신경망(R-CNNs) 덕분에 크게 발전했다. 특히 R-CNN의 속도는 컨볼루션 연산을 공유하는 Fast R-CNN의 등장으로 매우 빨라졌다. 그 결과, 이제는 역으로 후보 영역을 찾아내는 과정이 전체 시스템의 속도를 저해하는 주된 원인이 되었다.

두 가지 핵심 요소:

1.   후보 영역 제안 : 이미지에서 객체가 있을 만한 위치를 먼저 찾아내는 단계이다. (e.g. Selective Search)
2.   R-CNN : 찾아낸 후보 영역들 각각에 CNN을 적용하여 무엇인지 분류하는 모델이다.

초기의 R-CNN은 계산 비용이 매우 높았지만, 후보 영역들 간에 컨볼루션 연산을 공유하면서 그 비용이 획기적으로 줄었다. 

초기 R-CNN 은 수천 개의 후보 영역을 각각 독립적으로 CNN에 입력했기 때문에, 똑같은 연산을 수천 번 반복해야 해서 계산 비용이 매우 높았다. 이를 후보 영역들 간에 컨볼류션 연산을 공유하면서 그 비용을 획기적으로 줄였다. 이미지 전체를 CNN에 딱 한 번만 통과시켜 특징 맵을 만들고, 이 특징 맵에서 각 후보 영역에 해당하는 부분을 잘라와 사용한다. 이를 연산 공유라 한다. 가장 최신의 Fast R-CNN은 후보 영역을 찾는 데 걸리는 시간을 제외하면, 매우 깊은 신경망을 사용하고도 거의 실시간에 가까운 속도를 달성했다.

>   Now, proposals are the test-time computational bottleneck in state-of-the-art detection systems.

반대로 후보 영역 제안이 최신 탐지 시스템에서 실행 시간의 병목이 되었다. 이 문제를 해결하기 위해 등장한 것이 Faster R-CNN 이다.

Faster R-CNN에 들어가기 앞서 기존의 영역 제안 방법들에 대해서 살펴보자.

1.   Selective Search 방법은 사람이 설계한 낮은 수준의 특징을 기반으로 슈퍼픽셀(비슷한 픽셀들의 작은 묶음)들을 탐욕적으로 병합해 나간다. 즉, 딥러닝처럼 스스로 학습하는 것이 아니라, 색상, 질감, 크기 등 미리 정해진 단순한 규칙에 따라 이미지 조각들을 합쳐가며 우보 영역을 만드는 방식이다. CPU에서 이미지 한 장당 2초가 걸린다. 이는 탐지 네트워크 자체는 0.2초 만에 끝나는데, 후보 영역을 찾는 데만 2초가 걸리니 매우 비효율적인 상황인 것이다.

2.   EdgeBoxes 방법은 성능과 속도 사이에서 가장 나은 균형을 보여주며, 이미지당 0.2초가 걸려, 위 방법보다는 10배 빠르지만 충분히 빠르다고 볼 수 없다. 영역 제안 단계에서 여전히 탐지 네트워크만큼의 실행시간을 소모한다. 전체 과정에서 절반을 영역 제안에 사용해야 하는 것은 좋지 못하다.

Faster R-CNN은 기존 방식을 사용하는 대신 새로운 방법으로 컨볼루션 레이어를 공유하는 RPN(Regional Proposal Network)을 제안한다. 컨볼루션 레이어를 공유함으로써, 영역을 제안하는 데 드는 추가 비용은 이미지당 0.01초 수준으로 매우 작다.

>   Our observation is that the convolutional feature maps used by region-based detectors, like Fast R-CNN, can also be used for generating region proposals.

객체를 분류하기 위해 추출한 특징이나, 객체의 위치를 찾기 위해 필요한 특징이나 본질적으로 같을 것이라는 관찰에서 시작한다. Fast R-CNN과 같은 영역 기반 탐지기에서 사용하는 컨볼루션 특징 맵이 영역 제안을 생성하는 데에도 사용될 수 있다. 이 컨볼루션 특징 맵 위에, 몇 개의 컨볼루션 레이어를 추가하여 RPN을 구성한다. 이 추가된 레이어들은 일정한 격자의 모든 위치에서 영역의 경계를 회귀하고 객체일 확률을 동시에 계산한다. 따라서 RPN은 일정의 완전 컨볼루션 네트워크이며, 탐지용 후보 영역을 생성하는 직업에 특화되도록 end-to-end 방식으로 훈련 될 수 있다.

기존 모델들이 다양한 크기의 객체를 찾기 위해 이미지 자체의 크기를 바꾸거나, 다양한 크기의 필터를 사용하는 복잡하고 느린 방식을 썼던 것과 달리, RPN은 앵커 박스(Anchor Box)라는 새로운 개념을 도입했다.앵커 박스는 일종의 미리 정해둔 다양한 형태의 기준 박스들로, 이 기준 박스들을 이미지 전체에 적용함으로써 단 한 번의 분석만으로 여러 크기와 비율의 객체를 효율적으로 찾아낼 수 있다.

1.   이미지 피라미드 : 같은 이미지를 작은 버전, 중간 버전, 큰 버전으로 여러 개 만들어서 각각 분석하는 방식이다. 정확하지만 매우 느리다.
2.   필터 피라미드 : 이미지는 그대로 두고, 3x3, 5x5, 7x7 등 다양한 크기의 필터를 적용하는 방식이다.
3.   앵커 박스 : 위 두 가지와 달리, 이미지를 딱 한 번만 분석한다. 대신, 이미지의 각 위치마다 키가 큰 박스, 옆으로 넓은 박스, 정사각형 박스 등 미리 정의된 여러 형태의 기준 박스를 대어보고, 이 기준 박스를 미세 저정하여 객체를 찾는다. 