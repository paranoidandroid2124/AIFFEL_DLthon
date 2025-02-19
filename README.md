# AIFFEL_DLthon 01
DLthon을 준비하는 레포지토리입니다.

# 팀명: BIGMAC
#### __B__est __I__ntelligent __G__roup for __M__achine learning __A__nd __C__lassification


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

#### 결과
성능(Accuracy):
1일차: 0.70(KoELECTRA)
2일차: 0.78(textCNN - Homogeneous Ensemble Stacking with augmented data)

#### 모델 구성(성능 순 나열)
1. textCNN
2. KoELECTRA
3. BiLSTM
4. KoBERT
