# YouTube Description

## 제목 추천

반도체 방향성 예측 MVP - 미국 반도체 시장 정보로 SK하이닉스 다음 거래일 예측하기

## 설명란 본문

미국 반도체 시장의 직전 마감 정보를 활용해 SK하이닉스의 다음 거래일 방향성을 예측하는 딥러닝 수업 개인 프로젝트 MVP입니다.

이 프로젝트에서는 한국 시장 개장 전에 확인 가능한 미국 반도체 종목, ETF, 시장 지표, 환율/금리 정보를 이용해 `y_gap`과 `y_daily` 방향성을 예측했습니다.  
모델은 Logistic Regression, Random Forest, LSTM을 비교했고, 최근 3개월 hold-out 구간에서 성능과 간단한 백테스트를 확인했습니다.

영상에서 확인할 내용:

- 프로젝트 개요와 문제 정의
- F1~F5 feature set별 ML baseline 성능 비교
- Logistic Regression / Random Forest / LSTM 결과
- 예측 확률과 실제 방향 비교
- 백테스트 수익 곡선
- EDA, 전처리 프로토콜, 변수 중요도, LSTM ablation 보완 분석
- Streamlit MVP 대시보드 구성

주요 결과:

- `y_gap`에서는 `F1 + Logistic Regression` 조합이 가장 안정적인 성능을 보였습니다.
- `y_daily`에서는 `F5 + Logistic Regression` 조합이 가장 높은 ROC-AUC를 보였습니다.
- Random Forest는 최종 best 모델은 아니지만 변수 중요도 해석용 보조 모델로 활용했습니다.
- 기본 LSTM은 ML baseline보다 낮았지만, L2 regularization, BatchNormalization, Dropout을 강화한 ablation에서 개선 가능성을 확인했습니다.

Links:

- YouTube Demo: https://youtu.be/TiGpb7cMYU8
- Streamlit MVP: [배포 링크 입력]
- GitHub Repository: https://github.com/jjomton/semiconductor_stock_prediction
- GitHub Pages: https://jjomton.github.io/semiconductor_stock_prediction/

사용 기술:

Python, pandas, scikit-learn, TensorFlow/Keras, Streamlit, matplotlib

주의:

본 프로젝트는 딥러닝 수업 개인 프로젝트로, 실제 투자 조언이나 매매 추천이 아닙니다. 금융 데이터의 방향성 예측 가능성을 검증하기 위한 학습 목적의 MVP입니다.

## 태그 추천

`#딥러닝` `#Streamlit` `#머신러닝` `#주가예측` `#반도체` `#SK하이닉스` `#데이터분석` `#Python` `#LSTM` `#LogisticRegression`
