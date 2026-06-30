# Semiconductor Stock Direction Prediction MVP

미국 반도체 시장의 직전 마감 정보를 활용해 SK하이닉스의 다음 거래일 방향성을 예측하고, 저장된 결과를 Streamlit 대시보드로 확인하는 딥러닝 수업 개인 프로젝트입니다.

## Streamlit 실행

로컬 실행:

```powershell
cd C:\project\dlproj\semiconductor_stock_prediction
python -m streamlit run app/streamlit_app.py
```

Streamlit Community Cloud 설정:

- Repository: 이 GitHub 저장소
- Branch: `main`
- Main file path: `app/streamlit_app.py`
- Python dependencies: `requirements.txt`

## 대시보드 구성

- `Overview`: 프로젝트 개요와 핵심 결론
- `Model Results`: F1~F5 feature set별 Logistic Regression / Random Forest 성능 비교
- `Predictions & Backtest`: 저장된 예측 결과와 백테스트
- `Report Figures`: 보고서용 핵심 차트
- `Supplement Analysis`: EDA, 전처리 프로토콜, 변수 중요도, LSTM ablation 보완 분석

## 포함된 배포 산출물

Streamlit 앱은 모델을 다시 학습하지 않고 아래 저장 산출물만 읽습니다.

- `artifacts/models/`: 저장된 모델 파일 목록 표시용
- `artifacts/predictions/`: 테스트 구간 예측 결과
- `artifacts/reports/`: ML/DL 성능 지표와 백테스트 결과
- `artifacts/report_figures/`: 보고서용 핵심 그림
- `artifacts/report_supplement/`: EDA 및 보완 분석 그림/표

## 제외한 파일

배포 저장소에는 수업 실습 파일, 원본 데이터 전체, 전처리 중간 대용량 CSV, 발표 PDF/PPT, 가상환경, 캐시 파일을 포함하지 않습니다.

## 핵심 결과

- `y_gap`: `F1 + Logistic Regression` 조합이 가장 높은 ROC-AUC를 보였습니다.
- `y_daily`: `F5 + Logistic Regression` 조합이 가장 높은 ROC-AUC를 보였습니다.
- Random Forest는 최종 best 모델은 아니지만 변수 중요도 해석을 위한 보조 모델로 사용했습니다.
- 기본 LSTM은 ML baseline보다 낮았고, L2 regularization, BatchNormalization, Dropout을 강화한 ablation으로 개선 가능성을 확인했습니다.
