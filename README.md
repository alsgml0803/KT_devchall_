
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
- __기타 함수__ (nan 값 변경, 이상치 제거, 정확도 계산)
- __시간 카테고리 편집 함수__ <br/>
  : 하루 24시간을 4시간 씩 6개의 활동시간으로 나눠서 고려했습니다.<br/> 
  /0시-4시 / 4시-8시 / 8시-12시 / 12시-16시 / 16시-20시 / 20시-24시/<br/> 
  시간에 따라 사람들의 활동이 다를 것이고, 그것을 반영하여 RTB도 영향을 받을 것이라 예상했습니다.
- __광고 IAB 카테고리 변수 편집 함수__<br/>
  : 카테고리 변수 분류가 많아 대분류를 기준으로 재정렬하고<br/>
   nan 값은 우선 0으로 대체하였습니다.

### 데이터 전처리
- __데이터 불러오기__
- __분포 및 결측치 조회__<br/>
: 단순 통계량 확인, 명목형 변수 개수 확인, OS 종류 별 분포 확인 <br/>
  -> 플랫폼, ADID 타입과 상관관계가 있을 것이라고 예상하여 진행
- __결측치 처리__<br/>
: 결측치 비율이 높거나, 값이 정의되지 않았고 변동가능성이 있는 변수를 제거<br/>
 광고 응답 소재 카테고리 nan값을 'blank' 로 변경
- __이상치 처리__<br/>
: 범주형이 아닌 연속형 변수 중 이상치 조회 및 삭제<br/>
 p1 : DSP bidprice , p2 : AX bidfloor , p3 : SSP bidfloor 로 예상하여<br/>
__x1 = P1 / P2__<br/>
__x2 = (1 - (P4 / P1))__<br/>
__x3 = (P4 - P3) / (P1 - P3)__<br/>
라는 공식을 통해 금액 간 관계를 가정하고 변수를 추가하여 분석을 진행<br/>
: x2와 x3는 0에서 1사이에 값이 존재하나 x1은 그렇지 않음을 확인<br/>
 x1 분포 확인 및 이상치 제거
- __명목형, 범주형 변수 범주화 및 재정렬__<br/>
: 시각, DSP ID 에 대하여 값 대체
- __x1, x3 정규화__<br/>
: MinMaxScaler를 사용하여 0에서 1사이로 지정하고 진행
- __데이터 원핫인코딩__<br/>
: 명목형 변수의 데이터 타입을 str로 변경 후 원핫인코딩 진행
- __train, test data 분리__<br/>

### 데이터 분석
- __RFECV를 통해 변수선택__<br/>
: RFECV는 변수 선택 방법으로, 원하는 개수의 변수들이 남을 때까지 학습을 반복하며 유의미하지 않은 변수들을 제거하며 학습마다 Cross Validation을 활용해 성능을 계산<br/>
  __min_features_to_select :__ 최소한으로 선택할 변수 개수<br/>
  __step :__ 매 단계마다 제거할 변수 개수<br/>
  __cv :__ Cross Validation시 사용할 폴드 개수<br/>
fit을 적용한 후 RFE를 통해 선택된 변수가 무엇인지는 selector.support_를 통해 확인<br/>
전체 변수에 대해 선택된 경우 True, 선택되지 않은 경우는 False를 반환<br/>
True에 해당하는 변수 추출<br/>
train data에 채택한 변수 사용
- __로지스틱 회귀분석 모델 학습__<br/>
: 종속 변수와 독립 변수간의 관계를 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것<br/>
종속 변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 사용
- __예측값 추출 및 정확도 검정__<br/>
: 로지스틱 회귀모델을 이용하여 예측값을 추출하고 정확도를 검정, 모델을 판단


## 3. 개발 환경(OS) 및 라이브러리 설명

### 개발 환경(OS)
:

### 라이브러리 설명

import seaborn as sns<br/>
import numpy as np<br/>
import pandas as pd<br/>
import matplotlib.pyplot as plt<br/>
import matplotlib as mpl<br/>
from sklearn.preprocessing import MinMaxScaler<br/>
from sklearn.model_selection import train_test_split<br/>
from sklearn.feature_selection import RFECV<br/>
from sklearn.linear_model import LogisticRegression<br/>
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score<br/>
import statsmodels.api as sm<br/>
import warnings<br/>
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

