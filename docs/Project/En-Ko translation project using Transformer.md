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

프로젝트는 [허깅페이스][https://huggingface.co]에서 제공하는 데이터셋을 사용한다. 허깅페이스 페이지 상단에 있는 "[Datasets][https://huggingface.co/datasets]"에 들어가면 Task 에 따른 다양한 데이터셋을 이용할 수 있다. 필자는 뉴스와 일상대화를 다루는 말뭉치인 [bongsoo/news_talk_en_ko][https://huggingface.co/datasets/bongsoo/news_talk_en_ko] 데이터셋을 사용하여 프로젝트를 진행하였다. bongsoo/news_talk_en_ko 데이터셋은 "news_talk_en_ko_train_130000.tsv" 라는 이름의 파일로 다운로드 된다. 파일 안에는 130,000 줄로 이루어진 영어와 한국어 번역 내용이 들어있고, 다음의 명령어로 쉽게 파일이 어떻게 구성되는지 알 수 있다.

```python
!head -3 news_talk_en_ko_train_130000.tsv
```

```
Skinner's reward is mostly eye-watering.	스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다.
Even some problems can be predicted.	심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다.
Only God will exactly know why.	오직 하나님만이 그 이유를 제대로 알 수 있을 겁니다.
```

 `pandas` 라이브러리를 사용하면 데이터셋을 따로 다운로드 하지 않고, 데이터를 불러올 수 있다. 데이터셋이 따로 헤더가 설정되어 있지 않고, 데이터가 탭으로 구분되어 있기에 `, header=None, sep="\t"` 인자를 추가해서 불러준다. 그러고나서 헤더를 따로 추가해주면 된다.

```python
import pandas as pd

En_Ko = pd.read_csv("hf://datasets/bongsoo/news_talk_en_ko/news_talk_en_ko_train_130000.tsv", header=None, sep="\t")
En_Ko.columns = ["En", "Ko"]
```

`En_Ko` 는 `pandas.core.frame.DataFrame` 구조를 가지고 있고, 출력하면 다음과 같이 나온다.

---

|         |                                                En |                                                           Ko |
| ------: | ------------------------------------------------: | -----------------------------------------------------------: |
|       0 |          Skinner's reward is mostly eye-watering. |      스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다. |
|       1 |              Even some problems can be predicted. |  심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다. |
|       2 |                   Only God will exactly know why. |          오직 하나님만이 그 이유를 제대로 알 수 있을 겁니다. |
|       3 |   Businesses should not overlook China's dispute. |      중국의 논쟁을 보며 간과해선 안 될 게 기업들의 고충이다. |
|       4 |         Slow-beating songs often float over time. |        박자가 느린 노래는 오랜 시간이 지나 뜨는 경우가 있다. |
|     ... |                                               ... |                                                          ... |
| 1299995 | It says that this is a lost card, I am sorry, ... |  분실된 카드라고 나오는데, 죄송하지만 본인 카드 맞으신 가요? |
| 1299996 | It is my card, I found it after reporting a lo... | 제 카드는 맞는데, 분실신고했다가 다시 찾았는데 그럼 사용이 안 되나요? |
| 1299997 | Well, you will have to check with the bank, I ... | 글쎄요, 은행에 확인해보셔야 할 것 같아요, 저는 잘 모르겠네요. |
| 1299998 |        Then I will pay with cash, how much is it? |                그럼 일단 현금으로 계산할게요, 얼마나 나왔죠? |
| 1299999 |              I booked a room, can I check-in now? |                  숙박 예약을 했는데요, 지금 입실 가능한가요? |

1300000 rows × 2 columns

---

이렇게 데이터 셋을 `DataFrame`으로 만들었다. 이 `En_Ko` 로부터 다시 허깅페이스 데이터 셋을 생성할 것이다.

```python
from datasets import Dataset

dataset = Dataset.from_pandas(En_Ko)
```



```
Dataset({
    features: ['En', 'Ko'],
    num_rows: 1300000
})
```