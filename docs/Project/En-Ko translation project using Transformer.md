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

`pandas.core.frame.DataFrame` 구조의  `En_Ko` 를 허깅페이스 데이터셋(`datasets.arrow_dataset.Dataset`)으로 변경할 것이다.

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

이렇게 만들어진 `Dataset` 을 train, valid, test 데이터셋으로 나누어 줄 것이다. train 데이터는 120,000개, valid 데이터는 90,000개, test 데이터는 10,000개로 나눠주었다. 그리고 이들을  `tsv` 파일로 새롭게 저장해주면, 필요할때마다 이 파일들을 읽어 허깅페이스 데이터셋을 만들 수 있다.

```python
num_train, num_valid, num_test = 120000, 90000, 10000

En_Ko_train = En_Ko.iloc[:num_train]
En_Ko_valid = En_Ko.iloc[num_train:num_train+num_valid]
En_Ko_test = En_Ko.iloc[-num_test:]

En_Ko_train.to_csv("train.tsv", sep="\t", index=False)
En_Ko_valid.to_csv("valid.tsv", sep="\t", index=False)
En_Ko_test.to_csv("test.tsv", sep="\t", index=False)
```

허깅페이스 데이터셋으로 만들려면 아래처럼 정의한 `data_files` 를 `load_dataset` 에 넘기면 된다. 이때 `delimiter="\t"` 으로 지정해야 한다.

```python
from dayasets import load_dataset

data_files = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}
dataset =  load_dataset("csv", data_files=data_files, delimiter="\t")
```

`DatasetDict` 에 `train`, `valid`, `test` key로 각가 120만, 9만, 1만개의 문장이 저장된 것을 확인할 수 있다.

```
DatasetDict({
    train: Dataset({
        features: ['En', 'Ko'],
        num_rows: 120000
    })
    valid: Dataset({
        features: ['En', 'Ko'],
        num_rows: 90000
    })
    test: Dataset({
        features: ['En', 'Ko'],
        num_rows: 10000
    })
})
```

이 데이터셋에서 개별 데이터에 대한 접근은 `[split][feature][row num]` 형태로 가능하다.

```python
print(dataset['train']['en'][:3], dataset['train']['ko'][:3])
```

```
["Skinner's reward is mostly eye-watering.", 'Even some problems can be predicted.', 'Only God will exactly know why.']
['스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다.', '심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다.', '오직 하나님만이 그 이유를 제대로 알 수 있을 겁니다.']
```





## Model: Transformer

시작에 앞서 모델을 구현하는 데 필요한 라이브러리를 먼저 선언한다.

```python
import torch
import torch.nn as nn
import math
from matplotlib import pyplot as plt
```

### Embedding

트랜스포머에 입력 데이터인 단어를 입력하기 위해서는 단어에 대해 두가지 전처리를 해야 하는데 하나는 **Embedding** 이고 다른 하나는 **Positional Encoding **이다. 

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
```



### PositionalEncoding

위치 인코딩은 입력 시퀀스의 순서 정보를 모델에 전달하는 방법이다. 각 단어의 위치 정보를 나타내는 벡터를 더하여 임베딩 벡터에 위치 정보를 반영한다. 위치 인코딩 벡터는 $sin$ 함수와 $cos$ 함수를 사용해 생성되며, 이를 통해 임베딩 벡터와 위치 정보가 결합된 최종 입력 벡터를 생성한다.  **위치 인코딩 벡터를 추가함으로써 모델은 단어의 순서 정보를 학습할 수 있다**.

위치 인코딩은 각 토큰의 위치를 각도로 표현해 $sin$ 함수(`pe[:, 0, 0::2] = torch.sin(position * div_term` )와 $cos$ 함수(`pe[:, 0, 1::2] = torch.cos(position * div_term)`)로 위치 인코딩 벡터를 계산한다. 이러한 계산 방법은 토큰의 위치마다 동일한 임베딩 벡터를 사용하지 않기 때문에 각 토큰의 위치 정보를 모델이 학습할 수 있다. 

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)	# tensor([[0], [1], ... , [max_len-2], [max_len-1]]) 생성
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (1/10000(2i/d_model))

        pe = torch.zeros(max_len, 1, d_model)	# tensor([[[0] * max_len] * d_model])
        pe[:, 0, 0::2] = torch.sin(position * div_term) # 짝수 인덱스
        pe[:, 0, 1::2] = torch.cos(position * div_term) # 홀수 인덱스
        self.register_buffer("pe", pe) # 모델이 매개변수를 갱신하지 않도록 설정

    def forward(self, x):
        x = x + self.pe[: x.size(0)] # 입력 길이에 맞춰 필요한 위치 인코딩만 선택하여 더함.
        return self.dropout(x)
```

**code review**

`PositionalEncoding` 클래스는 입력 **임베딩 차원(d_model)**과 **최대 시퀀스(max_len)**를 입력받는다. 입력 시퀀스의 위치마다 $sin$ 과 $cos$ 함수로 위치 인코딩을 계산한다.
$$
PE_{(pos, 2i)} = \sin \left(\frac{pos}{{10000^{\frac{2i}{d_{\text{model}}}}}} \right), \space
PE_{(pos, 2i+1)} = \cos \left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}} \right)
$$
${\frac{1}{10000^{\frac{2i}{d_{\text{model}}}}}}$ 의 수식은 `torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))` 으로 구현할 수 있다. 코드를 풀어쓰면 $\exp^{(2i*-\frac{\log_e{10^5}}{d_\text{model}})}$이다. 이는 $\exp^{\log_e{10000^{-\frac{2i}{d_{\text{model}}}}}}$ 이라 쓸 수 있고,  $\exp(\log(x)) = x$ 이므로 결과적으로 ${\frac{1}{10000^{\frac{2i}{d_{\text{model}}}}}}$ 으로 쓸 수 있다.

`pe` 를 `pe[:, 0, 0::2]` 와 `pe[:, 0, 1::2]` 로 정의하여, 짝수 인덱스에서는 $sin$ 함수를 홀수 인덱스에서는 $cos$ 함수를 사용하여 

`self.register_buffer("pe", pe)` 는 pe를 모델의 상수 버퍼로 등록한다. 이는 학습되진 않지만 모델 저장/로딩할 때 같이 저장된다.

