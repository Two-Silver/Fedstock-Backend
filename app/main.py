from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sklearn.preprocessing import RobustScaler, StandardScaler


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from src.fl.server_clustering import assign_new_client  # noqa: E402
from src.models.lstm import LightweightLSTM  # noqa: E402


DEFAULT_REFERENCE_RUN = _BACKEND_ROOT / "reference_run"
REFERENCE_RUN_DIR = Path(os.getenv("FEDSTOCK_REFERENCE_RUN") or str(DEFAULT_REFERENCE_RUN))
STORAGE_DIR = Path(os.getenv("FEDSTOCK_BACKEND_STORAGE") or str(_BACKEND_ROOT / "storage"))
STATE_FILE_NAME = "registry_state.json"

SEQ_LEN = 14


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _normalize_column_name(value: str) -> str:
    return (
        str(value)
        .lstrip("\ufeff")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_normalize_column_name(column): column for column in df.columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _parse_importance_payload(raw: Any) -> np.ndarray:
    if isinstance(raw, list):
        return np.asarray(raw, dtype=np.float32)
    if isinstance(raw, dict):
        for key in ("noisyImportance", "importance", "vector", "values"):
            if key in raw and isinstance(raw[key], list):
                return np.asarray(raw[key], dtype=np.float32)
    raise ValueError("importance payload must be a numeric list or a dict containing noisyImportance")


def _load_model_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(model_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported model file: {model_path}")
    return state_dict


def _build_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> LightweightLSTM:
    input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    model = LightweightLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _prepare_prediction_frame(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    item_col = _find_column(df, ["item_id", "itemId", "id", "상품 ID", "상품ID"])
    date_col = _find_column(df, ["sale_date", "date", "판매일", "날짜"])
    sales_col = _find_column(df, ["sales", "quantity", "판매량", "수량"])
    price_col = _find_column(df, ["sell_price", "price", "판매가", "가격"])

    missing = []
    if item_col is None:
        missing.append("item_id")
    if date_col is None:
        missing.append("sale_date")
    if sales_col is None:
        missing.append("sales")
    if price_col is None:
        missing.append("sell_price")
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    prepared = df.copy()
    prepared["item_id"] = prepared[item_col].astype(str)
    prepared["sale_date"] = pd.to_datetime(prepared[date_col], errors="coerce")
    prepared["sales"] = pd.to_numeric(prepared[sales_col], errors="coerce")
    prepared["sell_price"] = pd.to_numeric(prepared[price_col], errors="coerce")
    prepared = prepared.dropna(subset=["item_id", "sale_date", "sales", "sell_price"])
    if prepared.empty:
        raise HTTPException(status_code=400, detail="No usable rows for prediction")

    prepared = prepared.sort_values(["item_id", "sale_date"]).reset_index(drop=True)
    prepared["dayofweek"] = prepared["sale_date"].dt.dayofweek
    prepared["month"] = prepared["sale_date"].dt.month
    prepared["week_of_year"] = prepared["sale_date"].dt.isocalendar().week.astype(int)
    prepared["is_weekend"] = prepared["dayofweek"].isin([5, 6]).astype(int)
    prepared["is_month_start"] = prepared["sale_date"].dt.is_month_start.astype(int)
    prepared["is_month_end"] = prepared["sale_date"].dt.is_month_end.astype(int)
    prepared["is_holiday"] = 0

    grouped_sales = prepared.groupby("item_id", sort=False)["sales"]
    prepared["lag_7"] = grouped_sales.shift(7)
    prepared["lag_14"] = grouped_sales.shift(14)
    prepared["lag_28"] = grouped_sales.shift(28)
    prepared["rolling_mean_7"] = grouped_sales.transform(lambda values: values.shift(1).rolling(7, min_periods=1).mean())
    prepared["rolling_mean_28"] = grouped_sales.transform(lambda values: values.shift(1).rolling(28, min_periods=1).mean())
    prepared["rolling_std_7"] = grouped_sales.transform(lambda values: values.shift(1).rolling(7, min_periods=2).std()).fillna(0)
    prepared["rolling_std_28"] = grouped_sales.transform(lambda values: values.shift(1).rolling(28, min_periods=2).std()).fillna(0)
    prepared["price_change_rate"] = (
        prepared.groupby("item_id", sort=False)["sell_price"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    prepared = prepared.dropna(subset=selected_features + ["sales"]).reset_index(drop=True)
    if prepared.empty:
        raise HTTPException(status_code=400, detail="Not enough history to derive lag/rolling features")
    return prepared


def _predict_with_model(model_path: Path, df: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    prepared = _prepare_prediction_frame(df, selected_features)
    state_dict = _load_model_state_dict(model_path)
    model = _build_model_from_state_dict(state_dict)

    forecast_rows = []
    with torch.no_grad():
        for item_id, group in prepared.groupby("item_id", sort=False):
            group = group.sort_values("sale_date").reset_index(drop=True)
            if len(group) <= SEQ_LEN:
                continue

            features = group[selected_features].to_numpy(dtype=np.float32)
            sales = group["sales"].to_numpy(dtype=np.float32)
            x_scaler = StandardScaler()
            y_scaler = RobustScaler()
            scaled_features = x_scaler.fit_transform(features).astype(np.float32)
            y_scaler.fit(sales.reshape(-1, 1))

            sequence = scaled_features[-SEQ_LEN:]
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            scaled_prediction = model(tensor).cpu().numpy()
            forecast = float(y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0])

            forecast_rows.append(
                {
                    "item_id": str(item_id),
                    "last_date": group["sale_date"].iloc[-1].strftime("%Y-%m-%d"),
                    "forecast_qty": round(max(0.0, forecast), 2),
                    "latest_sales": round(float(group["sales"].iloc[-1]), 2),
                }
            )

    if not forecast_rows:
        raise HTTPException(status_code=400, detail="Prediction requires at least one item with more than 14 usable rows")

    total_forecast = round(sum(row["forecast_qty"] for row in forecast_rows), 2)
    return {
        "modelPath": str(model_path),
        "selectedFeatures": selected_features,
        "totalForecastQty": total_forecast,
        "items": forecast_rows,
    }


@dataclass
class RegisteredClient:
    client_id: str
    model_path: Path
    importance: np.ndarray
    source: str
    registered_at: str
    sample_weight: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "clientId": self.client_id,
            "modelPath": str(self.model_path),
            "source": self.source,
            "registeredAt": self.registered_at,
            "sampleWeight": self.sample_weight,
            "importancePreview": [round(float(value), 6) for value in self.importance[:5]],
        }


@dataclass
class CentralRegistry:
    reference_run_dir: Path
    storage_dir: Path
    selected_features: list[str] = field(default_factory=list)
    clients: dict[str, RegisteredClient] = field(default_factory=dict)
    bubbles: list[list[str]] = field(default_factory=list)
    isolated: list[str] = field(default_factory=list)
    aggregated_models: dict[str, Path] = field(default_factory=dict)

    @property
    def state_file(self) -> Path:
        return self.storage_dir / STATE_FILE_NAME

    def _persist_state(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "savedAt": _now(),
            "referenceRunDir": str(self.reference_run_dir),
            "selectedFeatures": self.selected_features,
            "bubbles": self.bubbles,
            "isolated": self.isolated,
            "aggregatedModels": {
                key: str(path)
                for key, path in self.aggregated_models.items()
            },
            "clients": {
                client_id: {
                    "clientId": client.client_id,
                    "modelPath": str(client.model_path),
                    "importance": [float(value) for value in client.importance],
                    "source": client.source,
                    "registeredAt": client.registered_at,
                    "sampleWeight": client.sample_weight,
                }
                for client_id, client in self.clients.items()
            },
        }
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def bootstrap_reference_run(self, reference_run_dir: Path | None = None) -> dict[str, Any]:
        run_dir = reference_run_dir or self.reference_run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Reference run not found: {run_dir}")

        config = _read_json(run_dir / "config.json", {})
        self.selected_features = [str(name) for name in config.get("selected_features", [])]
        if not self.selected_features:
            raise ValueError("Reference run is missing selected_features in config.json")

        raw_importances = _read_json(run_dir / "feature_importances.json", {})
        clustering = _read_json(run_dir / "clustering_results.json", {})
        latest_record = clustering.get("records", [])[-1] if clustering.get("records") else clustering

        self.bubbles = [
            [str(client_id) for client_id in bubble.get("clients", [])]
            for bubble in latest_record.get("multi_client_bubbles", [])
            if bubble.get("clients")
        ]
        self.isolated = [str(client_id) for client_id in latest_record.get("isolated_clients", [])]
        self.clients = {}
        self.aggregated_models = {}

        model_dir = run_dir / "models_local_pretrain" / "clients"
        for client_id, raw_vector in raw_importances.items():
            model_path = model_dir / f"client_{client_id}.pt"
            if not model_path.exists():
                continue
            self.clients[str(client_id)] = RegisteredClient(
                client_id=str(client_id),
                model_path=model_path,
                importance=np.asarray(raw_vector, dtype=np.float32),
                source="reference",
                registered_at=_now(),
            )

        self.reference_run_dir = run_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._persist_state()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "referenceRunDir": str(self.reference_run_dir),
            "storageDir": str(self.storage_dir),
            "selectedFeatures": self.selected_features,
            "clientCount": len(self.clients),
            "bubbleCount": len(self.bubbles),
            "isolatedCount": len(self.isolated),
            "aggregatedModelCount": len(self.aggregated_models),
        }

    def list_clients(self) -> list[dict[str, Any]]:
        return [self.clients[client_id].as_dict() for client_id in sorted(self.clients.keys())]

    def list_clusters(self) -> list[dict[str, Any]]:
        rows = []
        for idx, bubble in enumerate(self.bubbles):
            rows.append(
                {
                    "clusterId": idx,
                    "type": "bubble",
                    "size": len(bubble),
                    "clients": bubble,
                    "aggregatedModelPath": str(self.aggregated_models.get(f"cluster_{idx}")) if f"cluster_{idx}" in self.aggregated_models else None,
                }
            )
        for idx, client_id in enumerate(self.isolated):
            rows.append(
                {
                    "clusterId": len(self.bubbles) + idx,
                    "type": "isolated",
                    "size": 1,
                    "clients": [client_id],
                    "aggregatedModelPath": None,
                }
            )
        return rows

    def _existing_importances(self, exclude: str | None = None) -> dict[str, np.ndarray]:
        return {
            client_id: client.importance
            for client_id, client in self.clients.items()
            if exclude is None or client_id != exclude
        }

    def _cluster_members_after_assignment(self, client_id: str, result: dict[str, Any]) -> list[str]:
        bubble_index = result.get("bubble_index")
        if bubble_index is not None and 0 <= int(bubble_index) < len(result["bubbles"]):
            return list(result["bubbles"][int(bubble_index)])
        return [client_id]

    def _effective_model_path_for_client(self, client_id: str) -> Path:
        for idx, bubble in enumerate(self.bubbles):
            if client_id in bubble:
                cluster_key = f"cluster_{idx}"
                if cluster_key in self.aggregated_models:
                    return self.aggregated_models[cluster_key]
        if client_id not in self.clients:
            raise KeyError(client_id)
        return self.clients[client_id].model_path

    def _aggregate_models(self, member_ids: list[str], cluster_key: str) -> Path:
        state_dicts = [_load_model_state_dict(self.clients[client_id].model_path) for client_id in member_ids]
        if not state_dicts:
            raise ValueError("Cannot aggregate an empty member list")

        weights = torch.tensor(
            [max(1, int(self.clients[client_id].sample_weight)) for client_id in member_ids],
            dtype=torch.float32,
        )
        normalized_weights = weights / weights.sum()
        averaged: dict[str, torch.Tensor] = {}
        for key in state_dicts[0].keys():
            tensors = [state_dict[key].float() for state_dict in state_dicts]
            stacked = torch.stack(tensors, dim=0)
            reshape_dims = [len(member_ids)] + [1] * (stacked.dim() - 1)
            averaged[key] = (stacked * normalized_weights.view(*reshape_dims)).sum(dim=0)

        aggregated_dir = self.storage_dir / "aggregated_models"
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        output_path = aggregated_dir / f"{cluster_key}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pt"
        torch.save(averaged, output_path)
        self.aggregated_models[cluster_key] = output_path
        return output_path

    def register_client(
        self,
        client_id: str,
        model_bytes: bytes,
        importance_vector: np.ndarray,
        sample_weight: int = 1,
    ) -> dict[str, Any]:
        replaced_existing = client_id in self.clients
        incoming_dir = self.storage_dir / "incoming" / client_id
        incoming_dir.mkdir(parents=True, exist_ok=True)
        model_path = incoming_dir / f"client_{client_id}.pt"
        importance_path = incoming_dir / f"{client_id}_importance.json"
        model_path.write_bytes(model_bytes)
        importance_path.write_text(
            json.dumps({"clientId": client_id, "noisyImportance": [float(value) for value in importance_vector]}, indent=2),
            encoding="utf-8",
        )

        self.clients[client_id] = RegisteredClient(
            client_id=client_id,
            model_path=model_path,
            importance=np.asarray(importance_vector, dtype=np.float32),
            source="uploaded",
            registered_at=_now(),
            sample_weight=sample_weight,
        )

        assignment = assign_new_client(
            self.clients[client_id].importance,
            self._existing_importances(exclude=client_id),
            self.bubbles,
            self.isolated,
            metric="cosine",
            new_client_id=client_id,
        )
        self.bubbles = [list(bubble) for bubble in assignment["bubbles"] if len(bubble) > 1]
        self.isolated = list(assignment["isolated"])

        member_ids = self._cluster_members_after_assignment(client_id, assignment)
        similar_clients = sorted(
            [
                {
                    "clientId": other_id,
                    "distance": round(_safe_cosine_distance(self.clients[client_id].importance, self.clients[other_id].importance), 6),
                }
                for other_id in member_ids
                if other_id != client_id
            ],
            key=lambda item: item["distance"],
        )

        aggregated_model_path = None
        if len(member_ids) > 1:
            cluster_key = f"cluster_{assignment['bubble_index']}"
            aggregated_model_path = self._aggregate_models(member_ids, cluster_key)

        fl_model_path = aggregated_model_path or model_path
        self._persist_state()
        return {
            "clientId": client_id,
            "replacedExisting": replaced_existing,
            "importancePath": str(importance_path),
            "modelPath": str(model_path),
            "assignedTo": assignment["assigned_to"],
            "clusterId": assignment["bubble_index"],
            "clusterMembers": member_ids,
            "similarClients": similar_clients[:5],
            "distance": assignment["distance"],
            "threshold": assignment["threshold"],
            "aggregatedModelPath": str(aggregated_model_path) if aggregated_model_path else None,
            "flModelPath": str(fl_model_path),
            "flModelDownloadUrl": f"/clients/{client_id}/fl-model",
            "bubbles": self.bubbles,
            "isolated": self.isolated,
        }


registry = CentralRegistry(reference_run_dir=REFERENCE_RUN_DIR, storage_dir=STORAGE_DIR)
try:
    registry.bootstrap_reference_run()
except Exception:
    pass


app = FastAPI(title="Fedstock Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "time": _now(),
        "summary": registry.summary(),
    }


@app.post("/bootstrap/reference-run")
def bootstrap_reference_run(reference_run_dir: str | None = Form(default=None)) -> dict[str, Any]:
    run_dir = Path(reference_run_dir) if reference_run_dir else registry.reference_run_dir
    try:
        return registry.bootstrap_reference_run(run_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state/summary")
def state_summary() -> dict[str, Any]:
    return registry.summary()


@app.get("/clients")
def clients() -> dict[str, Any]:
    return {
        "count": len(registry.clients),
        "clients": registry.list_clients(),
    }


@app.get("/clusters")
def clusters() -> dict[str, Any]:
    return {
        "count": len(registry.bubbles) + len(registry.isolated),
        "clusters": registry.list_clusters(),
    }


@app.get("/clients/{client_id}/fl-model")
def download_fl_model(client_id: str) -> FileResponse:
    if client_id not in registry.clients:
        raise HTTPException(status_code=404, detail=f"Unknown client: {client_id}")

    try:
        model_path = registry._effective_model_path_for_client(client_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown client: {client_id}") from exc

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    return FileResponse(
        path=str(model_path),
        filename=f"client_{client_id}_FL.pt",
        media_type="application/octet-stream",
    )


@app.post("/clients/register")
async def register_client(
    client_id: str = Form(...),
    model_file: UploadFile = File(...),
    importance_file: UploadFile | None = File(default=None),
    importance_json: str | None = Form(default=None),
    sample_weight: int = Form(default=1),
) -> dict[str, Any]:
    model_bytes = await model_file.read()
    if not model_bytes:
        raise HTTPException(status_code=400, detail="Uploaded model file is empty")

    try:
        if importance_file is not None:
            importance_payload = json.loads((await importance_file.read()).decode("utf-8"))
        elif importance_json is not None:
            importance_payload = json.loads(importance_json)
        else:
            raise ValueError("importance_file or importance_json is required")
        importance_vector = _parse_importance_payload(importance_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid importance payload: {exc}") from exc

    if registry.selected_features and len(importance_vector) != len(registry.selected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Importance length {len(importance_vector)} does not match selected feature count {len(registry.selected_features)}",
        )

    try:
        return registry.register_client(
            client_id=client_id,
            model_bytes=model_bytes,
            importance_vector=importance_vector,
            sample_weight=sample_weight,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_path: str | None = Form(default=None),
    client_id: str | None = Form(default=None),
) -> dict[str, Any]:
    if model_path:
        target_model_path = Path(model_path)
    elif client_id and client_id in registry.clients:
        target_model_path = registry.clients[client_id].model_path
    elif registry.aggregated_models:
        target_model_path = sorted(registry.aggregated_models.values())[-1]
    else:
        raise HTTPException(status_code=400, detail="Provide model_path or client_id, or register a client first")

    if not target_model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {target_model_path}")

    content = await file.read()
    try:
        frame = pd.read_csv(BytesIO(content))
    except UnicodeDecodeError:
        frame = pd.read_csv(BytesIO(content), encoding="cp949")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    return _predict_with_model(target_model_path, frame, registry.selected_features)
