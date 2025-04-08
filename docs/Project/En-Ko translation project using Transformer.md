---
layout: default
title: En-Ko translation project using Transformer
nav_order: 2.1
parent: Project
---

# En-Ko translation project using Transformer

## abstract

트랜스포머를 사용하여 영어 한국어 번역을 진행하는 것이 이 프로젝트의 목표이다. 필자가 이 프로젝트를 통해서 얻고자 하는 바는 스스로 트랜스포머 모델을 만드는 것이다. 파이토치 라이브러리를 이용하면 프로젝트를 구현함에 있어 어려움은 없겠지만, 스스로 구현함에 있어 이점은 있을 것이라 생각한다. 주로 다룰 내용은 데이터 분석과 처리, 토크나이징, 임베딩, 모델이다. 처음 시작하는 프로젝트이기에 부족한 부분이 있겠지만, 

추가적으로 **jupyter lab** 을 사용하여 일련의 과정으로 프로젝트를 진행하겠다.



### Environment

**MacBook Pro: Apple M4 Pro 칩(14코어 CPU, 20코어 GPU, 16코어 Neural Engine)**을 사용하여 시작하는 프로젝트이기에 학습에 제한이 있지만, 좋은 성능의 모델 제작이 아니므로 결과보다는 과정에 중심을 두겠다.



## DataSet

프로젝트는 [허깅페이스][https://huggingface.co]에서 제공하는 데이터셋을 사용한다. 허깅페이스 페이지 상단에 있는 "[Datasets][https://huggingface.co/datasets]"에 들어가면 Task 에 따른 다양한 데이터셋을 이용할 수 있다. 필자는 뉴스와 일상대화를 다루는 말뭉치인 [bongsoo/news_talk_en_ko][https://huggingface.co/datasets/bongsoo/news_talk_en_ko] 데이터셋을 사용하여 프로젝트를 진행하였다. bongsoo/news_talk_en_ko 데이터셋은 "news_talk_en_ko_train_130000.tsv" 라는 이름의 파일로 다운로드 된다. 파일 안에는 130,000 줄로 이루어진 영어와 한국어 번역 내용이 들어있다. 다음의 명령어로 쉽게 파일이 어떻게 구성되는지 알 수 있다.

```python
!head -3 news_talk_en_ko_train_130000.tsv
```

```
Skinner's reward is mostly eye-watering.	스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다.
Even some problems can be predicted.	심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다.
Only God will exactly know why.	오직 하나님만이 그 이유를 제대로 알 수 있을 겁니다.
```

데이터셋을 페이지에서 로컬 환경에 다운로드하면, `datasets` 라이브러리를 사용하여 데이터를 불러올 수 있다.

```python
from datasets import load_dataset

En_Ko = load_dataset("bongsoo/news_talk_en_ko")					# 데이터셋 위치는 프로젝트를 실행하는 폴더에 들어있다.
```

`En_Ko` 라는 데이터 객체를 확인해보면 `DatasetDict` 라는 것을 알 수 있고, 안에 `train` 키만 있는 것을 확인할 수 있다.

```python
DatasetDict({
    train: Dataset({
        features: ["Skinner's reward is mostly eye-watering.", '스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다.'],
        num_rows: 1299999
    })
})
```

`train`키에 `Dataset` 객체가 하나 있는데 `features`가 첫번째 데이터로 되어있고 행수는 1299999개인 것을 보아 데이터 파일에 컬럼명이 적혀있는 헤더라인이 없어서 첫줄을 헤더로 읽은것 같다. 첫줄을 데이터로 다시 집어 넣고 컬럼명은 `en`, `ko`로 설정하기 위해 데이터 셋을 `pandas`로 읽어드린다.