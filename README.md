# AIFFEL_DLthon 01
DLthon을 준비하는 레포지토리입니다.

# 팀명: BIGMAC
**B**est **I**ntelligent **G**roup for **M**achine learning **A**nd **C**lassification

### 멤버

| 이름   | 깃허브 | 담당 |
|--------|--------|----------------------------|
| 염철헌(팀장) | [깃허브](https://github.com/paranoidandroid2124/) | 모델 설계(textCNN, BiLSTM), 앙상블 스태킹, 데이터 증강 |
| 김유은 | [깃허브](https://github.com/yoo-eun00) | 모델 설계(KoBERT), 하이퍼파라미터 튜닝 |
| 김천지 | [깃허브](https://github.com/CheonjiKim) | 모델 설계(KoBERT), 하이퍼파라미터 튜닝 |
| 손병진 | [깃허브](https://github.com/SonByeongJin) | 모델 설계(KoELECTRA), 합성데이터 준비 및 전처리 |

### 세부 일정

| Day   | 날짜 | 태스크 |
|--------|--------|----------------------------|
| 1 | 25.02.18(화) | 일반 데이터 준비, 전처리 및 각자 모델 구성 후 종합 |
| 2 | 25.02.19(수) | 주요 모델 채택 후 실험 및 성능 평가 |
| 3 | 25.02.20(목) | 하이퍼파라미터 튜닝 및 성능 평가, 발표자료 구성 |
| 4 | 25.02.21(금) | 발표! |

# 주제: DKTC
본 프로젝트의 주제는 DKTC(Dataset of Korean Threatening Conversations)로, 일반텍스트까지 총 5개의 클래스를 분류하는 문제입니다. 자세한 설명은 [원본](https://github.com/tunib-ai/DKTC)에 기재되어 있습니다.

본 프로젝트는 [AIFFEL](https://www.kaggle.com/competitions/aiffel-dl-thon-dktc-online-12) 캐글 페이지에 submission.csv를 제출합니다.

## 핵심 과제
DKTC 훈련 데이터에 합성 데이터를 추가하여,  
협박 대화 = 0,  
갈취 대화 = 1,  
직장 내 괴롭힘 대화 = 2,  
기타 괴롭힘 대화 = 3,  
일반 대화 = 4 의 총 5가지 대화 유형 클래스를 분류하는 모델을 작성합니다.




#### 결과
성능(Accuracy):
1일차: 0.70(KoELECTRA)
2일차: 0.78(textCNN - Homogeneous Ensemble Stacking with augmented data)

#### 모델 구성(성능 순 나열)
1. textCNN
2. KoELECTRA
3. BiLSTM
4. KoBERT

#### 디테일
##### 1차 시도: textCNN
각 모델은 기본적으로 하이퍼파라미터를 조정한 동형의 앙상블을 기반으로 1차 스태킹을 하여 메타모델을 Fully Connected Network를 통해 학습합니다.

메타 모델을 통해 Validation Data를 F1-score, Classification Report, Confusion Matrix 등의 시각화 도구를 사용하여 분석하고, Test Data를 기반으로 Submission 파일을 작성합니다.

##### 2차 시도: textCNN + BiLSTM 스태킹
모델은 모듈화했고, main.py에서 모델을 불러와 각각 학습시킨 뒤 메타모델을 만들어 연결합니다. 스태킹 이후 최종 예측을 수행해 submission.csv 파일에 내놓습니다.

##### 3차 시도: .textCNN + BiLSTM + KoELECTRA 스태킹

##### 일반 데이터 준비
1차로 합성 데이터는 ChatGPT, Gemini 등의 챗봇을 기반으로 생성하였고, 성능을 확보하고자 합성 데이터와 기존의 train.csv의 합병 데이터로 실험을 진행 후 AI-Hub 등 존재하는 오픈 데이터 허브에서 자료를 가져와 2차로 확장하였음.

데이터 처리 과정에서는 무작위 셔플과 단어 추가/삭제를 통한 데이터 증강을 사용하였음.

또, 토크나이저를 커스텀화해서 형태소 분석에 공을 더 들였지만... 시간 관계 상 시도는 못함
