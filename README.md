
# &#128104;&#127995;&#8205;&#128187; 데이터는 과학EDA

## 과제2 모바일 광고 경매 낙찰 성공/실패 예측 모델
<전체 실행 프로세스, 코드 실행 방법, 개발 환경(OS) 및 라이브러리 설명>


## 1. 팀원 소개 &#128104;&#8205;&#128105;&#8205;&#128103;&#8205;&#128103;
### &#127940;&#127995; 김우진 (~)
역할: -

### &#129464;&#127995; 장민희 (lool_0803@naver.com)
역할: -


## 2. 전체 실행 프로세스
### 함수 정리
- 기타 함수
- 시간 카테고리 편집 함수
- 광고 IAB 카테고리 변수 편집 함수

### 데이터 전처리
- 데이터 불러오기
- 분포 및 결측치 조회
(단순 통계량 확인, 명목형 변수 개수 확인, OS 종류 별 분포 확인 - 플랫폼, ADID 타입과 상관관계가 있을 것이라고 예상)
- 결측치 처리
( 결측치 높은 것, 정의되지 않았고 변동가능성이 있는 변수 제거)
( 광고 응답 소재 카테고리 nan값 'blank' 로 변경)
- 이상치 처리
(범주형이 아닌 연속형 변수 중 이상치 조회 및 삭제)
(p1 : DSP bidprice , p2 : AX bidfloor , p3 : SSP bidfloor 로 예상하여
__x1 = P1 / P2__
__x2 = (1 - (P4 / P1))__
__x3 = (P4 - P3) / (P1 - P3)__
라는 공식을 통해 금액 간 관계를 가정하고 분석을 진행)
(x1, x2, x3 추가)
x2와 x3는 0에서 1사이에 값이 존재하나 x1은 그렇지 않음을 확인
-> 정규화 진행 예정
(x1 분포 확인 및 이상치 제거)
- 명목형, 범주형 변수 범주화 및 재정렬
- x1, x3 정규화
- 데이터 원핫인코딩
(명목형 변수의 데이터 타입을 str로 변경 후 원핫인코딩 진행)
- train, test data 분리

### 데이터 분석
- __RFECV를 통해 변수선택__
(RFECV는 변수 선택 방법으로, 원하는 개수의 변수들이 남을 때까지 학습을 반복하며 유의미하지 않은 변수들을 제거하며 학습마다 Cross Validation을 활용해 성능을 계산
- __min_features_to_select :__ 최소한으로 선택할 변수 개수
- __step :__ 매 단계마다 제거할 변수 개수
- __cv :__ Cross Validation시 사용할 폴드 개수)
(fit을 적용한 후 RFE를 통해 선택된 변수가 무엇인지는 selector.support_를 통해 확인)
(전체 변수에 대해 선택된 경우 True, 선택되지 않은 경우는 False를 반환)
(True에 해당하는 변수 추출)
- train data에 채택한 변수 사용
- 로지스틱 회귀분석 모델 학습
- 예측값 추출 및 정확도 검정



모델 선택 이유: 
~~

#### 사용한 모델 비교:
|사용 모델|장점|단점|정확도 (test)|
|--|--|--|--|
Logistic Regression|1. 구현이 용이 2. 속도와 예측이 빠름|선형관계를 전제로 한 모델이라 예측력이 떨어짐|0.8619|
KoBert|1. 높은 정확도2. 문맥에 대한 정보 저장|호환성의 문제로 사용 불가|0.8902|
LSTM|1. 다른 기술과 높은 호환성 2. 긴 문장 처리에 용이 3. 문맥에 대한 정보 저장|정확도가 모델 중 낮음|0.8602|


## 3. 개발 환경(OS) 및 라이브러리 설명

### 개발 환경(OS)
:

### 라이브러리 설명

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


## 4. 코드 실행 방법 (그런거 없는데영,.,..)
### ~~
~~
- ~~
- ~~
- 
### ~~
- ~~
- 

