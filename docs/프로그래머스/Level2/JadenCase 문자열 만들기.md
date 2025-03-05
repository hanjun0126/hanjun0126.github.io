---
layout: default
title: JadenCase 문자열 만들기
nav_order: 3
grand_parent: 프로그래머스
parent: Level2
---

[JadenCase 문자열 만들기](https://school.programmers.co.kr/learn/courses/30/lessons/12951)

문자열 s 를 JadenCase 문자열로 만드는 문제이다. JadenCase 규칙은 다음과 같다.

1. 단어의 첫 글자는 대문자, 나머지는 소문자로 변환한다.
2. 공백이 여러 개 존재할 수도 있다(공백 유지).
3. 숫자는 그대로 둔다.

split() 은 연속된 공백은 무시하므로 split(" ") 으로 사용한다.

capitalize() 함수는 첫 글자는 대문자, 나머지는 소문자로 변환된다.

```python
def solution(s):
    return " ".join(word.capitalize() for word in s.split(" "))
```

