# Semiconductor Stock Direction Prediction MVP

미국 반도체 시장의 직전 마감 정보를 활용해 SK하이닉스의 다음 거래일 방향성을 예측하고, 저장된 결과를 Streamlit 대시보드로 확인하는 딥러닝 수업 개인 프로젝트입니다.

## Links

- Streamlit MVP: 배포 후 Streamlit Cloud 링크를 입력하세요.
- GitHub Pages: `https://jjomton.github.io/semiconductor_stock_prediction/`
- YouTube Demo: `https://youtu.be/TiGpb7cMYU8`
- Repository: `https://github.com/jjomton/semiconductor_stock_prediction`

## Project Summary

한국 시장은 미국 시장보다 먼저 열리기 때문에, 한국 장 시작 전에 미국 반도체 종목과 ETF의 직전 마감 정보를 확인할 수 있습니다. 이 프로젝트는 해당 정보가 SK하이닉스의 다음 거래일 방향성 예측에 도움이 되는지 검증합니다.

주요 예측 타깃은 다음과 같습니다.

- `y_gap`: 다음 거래일 시가가 전일 종가보다 상승했는지 여부
- `y_daily`: 다음 거래일 종가가 전일 종가보다 상승했는지 여부
- `y_intraday`: 당일 종가가 당일 시가보다 상승했는지 여부

Streamlit 앱은 모델을 재학습하지 않고 저장된 산출물만 읽어 결과를 보여줍니다.

## Streamlit Dashboard

로컬 실행:

```powershell
cd C:\project\dlproj\semiconductor_stock_prediction
python -m streamlit run app/streamlit_app.py
```

Streamlit Community Cloud 설정:

- Repository: `jjomton/semiconductor_stock_prediction`
- Branch: `main`
- Main file path: `app/streamlit_app.py`
- Requirements: `requirements.txt`

## Dashboard Tabs

- `Overview`: 프로젝트 개요와 핵심 결론
- `Model Results`: F1~F5 feature set별 Logistic Regression / Random Forest 성능 비교
- `Predictions & Backtest`: 저장된 예측 결과와 백테스트
- `Report Figures`: 보고서용 핵심 차트
- `Supplement Analysis`: EDA, 전처리 프로토콜, 변수 중요도, LSTM ablation 보완 분석

## Data And Features

사용 데이터:

- 한국 타깃 종목: SK하이닉스
- 미국 반도체 ETF: SOXX, SMH
- 미국 반도체 종목: NVDA, AMD, MU, INTC, AVGO, QCOM, AMAT, LRCX, KLAC
- 시장 지표: QQQ, SPY, VIX
- 환율/금리: USD/KRW, USD/JPY, Dollar Index, US 10Y
- 한국 보조 지표: KOSPI, KOSDAQ

Feature set:

| Feature Set | 구성 |
| --- | --- |
| F1 | 미국 반도체 ETF |
| F2 | F1 + 미국 반도체 주요 종목 |
| F3 | F2 + 미국 시장지표/VIX |
| F4 | F3 + 환율/금리 |
| F5 | F4 + 한국 시장 및 타깃 종목 과거 흐름 |

## Models

ML baseline:

- Logistic Regression
- Random Forest

DL model:

- LSTM
- LSTM ablation: L2 regularization, BatchNormalization, Dropout 강화

평가 지표:

- Accuracy
- Balanced Accuracy
- Precision / Recall / F1-score
- ROC-AUC
- Confusion Matrix
- 간단 백테스트 누적 수익률

## Key Results

- `y_gap`: `F1 + Logistic Regression` 조합이 가장 높은 ROC-AUC를 보였습니다.
- `y_daily`: `F5 + Logistic Regression` 조합이 가장 높은 ROC-AUC를 보였습니다.
- Random Forest는 최종 best 모델은 아니지만 변수 중요도 해석을 위한 보조 모델로 사용했습니다.
- 기본 LSTM은 ML baseline보다 낮았고, 규제형 LSTM ablation에서 개선 가능성을 확인했습니다.

## Repository Contents

배포에 필요한 파일만 포함합니다.

```text
app/streamlit_app.py
.streamlit/config.toml
requirements.txt
README.md
docs/index.html
docs/assets/
artifacts/models/
artifacts/predictions/
artifacts/reports/
artifacts/report_figures/
artifacts/report_supplement/
```

원본 데이터 전체, 수업 실습 파일, 발표 PDF/PPT, 가상환경, 캐시 파일은 배포 대상에서 제외합니다.

## GitHub Pages

이 저장소는 루트의 `index.html`을 GitHub Pages 정적 사이트로 사용할 수 있게 구성했습니다.

GitHub 설정:

1. Repository의 `Settings`로 이동
2. `Pages` 선택
3. Source를 `Deploy from a branch`로 설정
4. Branch: `main`
5. Folder: `/root`
6. Save

배포 후 주소:

```text
https://jjomton.github.io/semiconductor_stock_prediction/
```
