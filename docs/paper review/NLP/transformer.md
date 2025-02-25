---
layout: default
title: Attention is all you need
nav_order: 1
grand_parent: paper review
parent: NLP
---

# *Attention is all you need*

## *abstract*

기존의 sequence 변환 모델들은 encoder 와 decoder를 사용하는 복잡한 recurrent 또는 cnn 을 기반으로 한다. 가장 성능이 높은 모델 또한 attention 기법으로 encoder 와 decoder 를 연결했다. 논문에서는 attention 메커니즘에만 기반한 transformer 구조를 제안한다.

*WMT  2014 English-to-German* translation task 에서는 28.4 BLEU(기존의 최고 성능보다 2 BLEU 성능 향상), *WMT 2014 English-to-French* translation task 에서는 41.8 BLEU 를 달성했다.

이 두 실험에서 transformer 구조가 성능면에서 뛰어남을 보여준다. 더 병렬화 되고 학습에 사용되는 시간은 훨씬 적다.

***

## *1. Introduction*

**Recurrent** 모델<sup>[1](#footnote_1)</sup>들은 일반적으로 입력과 출력 sequence 의 위치를 따라 계산을 수행한다. 본질적으로 순차적인 계산 특성은 병렬화를 배제한다. 이는 sequence 의 길이가 길어질수록 치명적이다. &rarr; sequential computation 의 제약이 있다.

**attention** 기법은 입력과 출력 sequence 의 거리에 상관없이 종속성을 모델링할 수 있다.

새롭게 제안하는 **Transformer** 구조는 순환 신경망을 배제하고 오로지 attention 만 사용하기에 입력과 출력의 global 의존성을 도출할 수 있다.

<span style="font-size:70%"><a name="footnote_1">1</a>: RNN, LSTM, recurrent gated neural network</span> 

***

## *2. Background*

**Self-attention** 은 sequence 의 representation 을 계산하기 위해 단일 sequencce 의 서로 다른 위치를 연관시키는 기법이다.

**End-to-end memory** 는 **Recurrent attention** 기법을 기반으로 한다.

***

## *3. Model Architecture*

![model](../../../assets/images/transformer_model_architecture.png){: width="50%" height="50%"}



<**center**><**img** src="../../../assets/images/transformer_model_architecture.png" width="50" height="50"></**center**>

