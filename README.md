# Fedstock Backend (Central FL Orchestrator API)

이 디렉토리는 Fedstock 중앙 서버용 FastAPI 애플리케이션입니다.

현재 구현 범위:

- 기준 run 자산 로드
  - 로컬 사전학습 모델 `.pt`
  - 클라이언트별 noisy feature importance
  - 기존 clustering 결과
- 신규 클라이언트 등록
  - 로컬 모델 업로드
  - noisy importance 업로드
- importance 기반 cluster assignment
- 같은 cluster 내부 모델 집계
- 집계된 모델 또는 특정 모델로 CSV 예측

## 실행

```bash
cd Fedstock-Backend
uvicorn app.main:app --reload --port 8100
```

## 주요 환경변수

- `FEDSTOCK_REFERENCE_RUN`
  - 기본값: `../outputs/runs/20260604_115558_467836`
- `FEDSTOCK_BACKEND_STORAGE`
  - 기본값: `./storage`

## 주요 엔드포인트

- `GET /health`
- `POST /bootstrap/reference-run`
- `GET /state/summary`
- `GET /clients`
- `GET /clusters`
- `GET /clients/{client_id}/fl-model`
- `POST /clients/register`
- `POST /predict`
