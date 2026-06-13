# Fedstock Backend API 명세서

## 1. 통신 대상 정의

| 구분 | 대상 | 설명 |
| --- | --- | --- |
| 호출 클라이언트 | AI Client Backend | Fedstock Backend와 실제로 통신하는 클라이언트 |
| 서버 | Fedstock Backend | 클라이언트 등록, 클러스터 배정, 집계 모델 생성, 모델 다운로드를 담당하는 중앙 서버 |
| 참조 자산 | `reference_run/` | 기준 feature importance, clustering 결과, 사전학습 모델을 읽는 디렉토리 |
| 저장소 | `storage/` | 업로드 모델, importance, 집계 모델, registry 상태를 저장하는 디렉토리 |

### 통신 방식

- Protocol: `HTTP / HTTPS`
- Content-Type
  - `application/json`
  - `multipart/form-data`
  - `application/octet-stream`
- 인증: 불필요

### 실제 연동 대상 API

`Fedstock-AI-Client`가 `Fedstock-Backend`에 실제로 호출하는 API는 아래 3개다.

1. `GET /health`
2. `POST /clients/register`
3. `GET /clients/{client_id}/effective-model`

---

## 2. 엔드포인트 리스트

1. `GET /health`
2. `POST /clients/register`
3. `GET /clients/{client_id}/effective-model`

---

# **Health Check API**

## **1. 기본 정보**

| **항목** | **내용** |
| --- | --- |
| Method | `GET` |
| Endpoint | `/health` |
| 설명 | 서버 상태와 현재 레지스트리 요약 정보를 조회한다. |
| 인증 | 불필요 |

---

## **2. Request**

### **Path Parameters**

| **이름** | **타입** | **필수** | **설명** |
| --- | --- | --- | --- |
|  |  |  |  |

### **Query Parameters**

| **이름** | **타입** | **필수** | **설명** |
| --- | --- | --- | --- |
|  |  |  |  |

### **Request Body**

```json
{}
```

---

## **3. Response**

### **성공 응답**

**Status Code:** `200 OK`

```json
{
  "ok": true,
  "time": "2026-06-04 12:00:00",
  "summary": {
    "referenceRunDir": "/path/to/reference_run",
    "storageDir": "/path/to/storage",
    "selectedFeatures": [
      "lag_7",
      "lag_14",
      "rolling_mean_7"
    ],
    "clientCount": 48,
    "bubbleCount": 5,
    "isolatedCount": 3,
    "aggregatedModelCount": 2
  }
}
```

### **Response Fields**

| **필드** | **타입** | **설명** |
| --- | --- | --- |
| ok | boolean | 서버 정상 여부 |
| time | string | 서버 현재 시각 (`YYYY-MM-DD HH:mm:ss`) |
| summary.referenceRunDir | string | 현재 reference run 경로 |
| summary.storageDir | string | storage 경로 |
| summary.selectedFeatures | string[] | 현재 선택된 feature 목록 |
| summary.clientCount | number | 등록된 클라이언트 수 |
| summary.bubbleCount | number | bubble 수 |
| summary.isolatedCount | number | isolated 클라이언트 수 |
| summary.aggregatedModelCount | number | 생성된 집계 모델 수 |

---

## **4. Error**

| **Status Code** | **설명** |
| --- | --- |
| 500 | 서버 오류 |

---

## **5. 요청 예시**

```http
GET /health
```

---

# **Client Register API**

## **1. 기본 정보**

| **항목** | **내용** |
| --- | --- |
| Method | `POST` |
| Endpoint | `/clients/register` |
| 설명 | 모델과 importance를 등록하고 클러스터를 배정한 뒤 effective model 경로를 반환한다. |
| 인증 | 불필요 |

---

## **2. Request**

### **Path Parameters**

| **이름** | **타입** | **필수** | **설명** |
| --- | --- | --- | --- |
|  |  |  |  |

### **Query Parameters**

| **이름** | **타입** | **필수** | **설명** |
| --- | --- | --- | --- |
|  |  |  |  |

### **Request Body**

`multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| client_id | string | Y | 등록할 클라이언트 ID |
| model_file | file | Y | 업로드할 `.pt` 모델 파일 |
| importance_file | file | N | importance JSON 파일 |
| importance_json | string | N | importance JSON 문자열 |
| sample_weight | integer | N | 집계 가중치. 기본값 `1` |

주의:

- `importance_file` 또는 `importance_json` 중 하나는 반드시 필요하다.
- `Fedstock-AI-Client`는 실제로 `importance_file`을 사용한다.
- importance 벡터 길이는 현재 `selectedFeatures` 길이와 같아야 한다.
- 실제 응답의 `assignedTo`는 `bubble`, `new_bubble`, `isolated` 중 하나다.

예시:

```json
{
  "client_id": "NEW_CLIENT_01",
  "importance_json": "{\"noisyImportance\":[0.12,0.03,0.55,0.07]}",
  "sample_weight": 10
}
```

---

## **3. Response**

### **성공 응답**

**Status Code:** `200 OK`

```json
{
  "clientId": "NEW_CLIENT_01",
  "replacedExisting": false,
  "importancePath": "/path/to/storage/incoming/NEW_CLIENT_01/NEW_CLIENT_01_importance.json",
  "modelPath": "/path/to/storage/incoming/NEW_CLIENT_01/client_NEW_CLIENT_01.pt",
  "assignedTo": "new_bubble",
  "clusterId": 2,
  "clusterMembers": ["NEW_CLIENT_01", "CA_1_FOODS_1"],
  "similarClients": [
    {
      "clientId": "CA_1_FOODS_1",
      "distance": 0.021345
    }
  ],
  "distance": 0.021345,
  "threshold": 0.15,
  "aggregatedModelPath": "/path/to/storage/aggregated_models/cluster_2_20260604_120000_000000.pt",
  "effectiveModelPath": "/path/to/storage/aggregated_models/cluster_2_20260604_120000_000000.pt",
  "effectiveModelDownloadUrl": "/clients/NEW_CLIENT_01/effective-model",
  "bubbles": [["NEW_CLIENT_01", "CA_1_FOODS_1"]],
  "isolated": ["TX_1_HOBBIES_1"]
}
```

### **Response Fields**

| **필드** | **타입** | **설명** |
| --- | --- | --- |
| clientId | string | 등록된 클라이언트 ID |
| replacedExisting | boolean | 기존 동일 ID를 덮어썼는지 여부 |
| importancePath | string | 저장된 importance JSON 경로 |
| modelPath | string | 저장된 모델 파일 경로 |
| assignedTo | string | `bubble`, `new_bubble`, `isolated` 중 하나 |
| clusterId | number \| null | 배정된 bubble 인덱스. isolated면 `null` |
| clusterMembers | string[] | 배정 후 동일 클러스터 구성원 목록 |
| similarClients[].clientId | string | 유사한 기존 클라이언트 ID |
| similarClients[].distance | number | cosine distance |
| distance | number \| null | 최근접 거리 |
| threshold | number \| null | 클러스터 배정 threshold |
| aggregatedModelPath | string \| null | 집계 모델 경로 |
| effectiveModelPath | string | 실제 사용 모델 경로 |
| effectiveModelDownloadUrl | string | effective model 다운로드 API 경로 |
| bubbles | string[][] | 등록 후 전체 bubble 목록 |
| isolated | string[] | 등록 후 전체 isolated 목록 |

---

## **4. Error**

| **Status Code** | **설명** |
| --- | --- |
| 400 | 모델 파일이 비어 있거나 importance가 잘못되었거나 배정 처리에 실패함 |
| 500 | 서버 오류 |

---

## **5. 요청 예시**

```http
POST /clients/register
Content-Type: multipart/form-data
```

---

# **Effective Model Download API**

## **1. 기본 정보**
## 기본 정보

| 항목 | 내용 |
| --- | --- |
| Method | `GET` |
| Endpoint | `/clients/{client_id}/fl-model` |
| 설명 | 특정 클라이언트가 사용해야 하는 FL Model 다운로드 |
| 인증 | 불필요 |

# Request

### Path Parameters

| 이름 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `client_id` | string | Y | 대상 클라이언트 ID |

### Query Parameters

없음

### Request Body

없음

# Response

### Success

**Status Code:** `200 OK`

실제 응답은 JSON이 아니라 PyTorch 모델 파일(Binary Stream)이다.

예시 표현:

```json
{
  "file": "client_{client_id}_FL.pt"
}
```

### Response Headers

```
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="client_{client_id}_FL.pt"
```

### Response Fields

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `file` | binary | 다운로드되는 PyTorch 모델 파일 |

### Error Responses

| Status Code | 설명 |
| --- | --- |
| `404` | 클라이언트 또는 모델 파일 없음 |
| `500` | 서버 내부 오류 |

### Example

```
GET /clients/CA_1_FOODS_1/fl-model
```
