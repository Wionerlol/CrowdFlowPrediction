"""Thin HTTP client used by the Streamlit UI."""

from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    """A user-displayable backend error."""


class Week6ApiClient:
    def __init__(self, base_url: str, *, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.base_url}/{path.lstrip('/')}",
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise ApiError(f"无法连接 API：{exc}") from exc
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"API {response.status_code}: {detail}")
        return response.json()

    def status(self) -> dict[str, Any]:
        return self._request("GET", "status")

    def heatmap(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "heatmap", params={k: v for k, v in params.items() if v is not None})

    def forecast(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "predictions", params={k: v for k, v in params.items() if v is not None})

    def anomalies(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "anomalies", params={k: v for k, v in params.items() if v is not None})

    def alerts(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "alerts", params={k: v for k, v in params.items() if v is not None})

    def anomaly_detail(self, event_id: int, *, source: str) -> dict[str, Any]:
        return self._request("GET", f"anomalies/{event_id}", params={"source": source})

    def statistics(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "statistics", params={k: v for k, v in params.items() if v is not None})
