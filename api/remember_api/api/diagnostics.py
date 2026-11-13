"""Local API health, version, and metrics routes."""

import asyncio
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from remember_api.liveness import LivenessResponse, liveness_response


class FeaturesInfo(BaseModel):
    """Feature flags indicating which capabilities are enabled."""

    observations: bool = Field(description="Whether observations (auto-consolidation) are enabled")
    worker: bool = Field(description="Whether the background worker is enabled")
    bank_config_api: bool = Field(description="Whether per-bank configuration API is enabled")
    bank_llm_health: bool = Field(description="Whether the per-bank LLM connectivity probe is enabled")
    audit_log: bool = Field(description="Whether audit logging is enabled by default (overridable per bank)")
    llm_trace: bool = Field(description="Whether per-bank LLM request tracing is enabled")
    store_document_text: bool = Field(
        description="Whether raw source text is persisted. When false, document/chunk source text is not stored."
    )


class VersionResponse(BaseModel):
    """Response model for the version/info endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_version": "0.9.0",
                "features": {
                    "observations": False,
                    "worker": True,
                    "bank_config_api": False,
                    "bank_llm_health": True,
                    "audit_log": False,
                    "llm_trace": False,
                    "store_document_text": True,
                },
            }
        }
    )

    api_version: str = Field(description="API version string")
    features: FeaturesInfo = Field(description="Enabled feature flags")


def register_diagnostics_routes(app: FastAPI, *, source_file: str) -> None:
    """Register local health, version, and metrics endpoints."""

    async def _readiness_response() -> JSONResponse:
        """Shared body of /health and /health/ready: 200 if healthy, 503 if not."""
        health = await app.state.memory.health_check()
        health["checkout_source"] = os.path.realpath(source_file)
        status_code = 200 if health.get("status") == "healthy" else 503
        return JSONResponse(content=health, status_code=status_code)

    @app.get(
        "/health",
        summary="Health check endpoint",
        description="Readiness check: verifies the API can reach the database. "
        "Alias of /health/ready. Use /health/live for liveness probes — this one "
        "fails whenever the database is unreachable, which must gate traffic, not "
        "restart the process.",
        tags=["Monitoring"],
    )
    async def health_endpoint():
        """
        Health check endpoint that verifies database connectivity.

        Returns 200 if healthy, 503 if unhealthy.
        """
        return await _readiness_response()

    @app.get(
        "/health/ready",
        summary="Readiness probe",
        description="Returns 200 when the API can serve traffic (database reachable), "
        "503 otherwise. Identical to /health, which stays supported as its alias.",
        tags=["Monitoring"],
        operation_id="get_readiness",
    )
    async def readiness_endpoint():
        """
        Readiness probe that verifies database connectivity.

        Returns 200 if ready, 503 if not. A 503 should remove this pod from the
        Service; it must not restart it.
        """
        return await _readiness_response()

    @app.get(
        "/health/live",
        response_model=LivenessResponse,
        summary="Liveness probe",
        description="Returns 200 whenever the process can serve a request. Performs no "
        "database access, so a slow or unreachable database never restarts the pod. "
        "Point livenessProbe here and readinessProbe at /health.",
        tags=["Monitoring"],
        operation_id="get_liveness",
    )
    async def liveness_endpoint() -> LivenessResponse:
        """
        Liveness probe: in-process only, never touches the database.

        Answering at all is the check — Remember serves requests and task work on
        one event loop, so a wedged loop cannot respond within the probe timeout.
        """
        return liveness_response()

    @app.get(
        "/version",
        response_model=VersionResponse,
        summary="Get API version and feature flags",
        description="Returns API version information and enabled feature flags. "
        "Use this to check which capabilities are available in this deployment.",
        tags=["Monitoring"],
        operation_id="get_version",
    )
    # Same handler under /v1, kept out of the OpenAPI spec (it is an alias, not a new operation).
    # A reverse proxy that routes only /v1 to the API — the Helm chart's default ingress does —
    # sends a bare /version to the control plane instead, so clients probing capabilities there
    # silently saw "unsupported" (#5286). Clients try this path first.
    @app.get("/v1/version", response_model=VersionResponse, include_in_schema=False)
    async def version_endpoint() -> VersionResponse:
        """
        Get API version and enabled features.

        Returns version info and feature flags that can be used by clients
        to determine which capabilities are available.

        Note: the observations and audit_log flags show the global default.
        Individual banks may override these via bank-specific configuration.
        """
        from remember_api import __version__
        from remember_api.config import _get_raw_config

        config = _get_raw_config()
        return VersionResponse(
            api_version=__version__,
            features=FeaturesInfo(
                observations=config.enable_observations,
                worker=config.worker_enabled,
                bank_config_api=config.enable_bank_config_api,
                bank_llm_health=config.enable_bank_llm_health,
                audit_log=config.audit_log_enabled,
                llm_trace=config.llm_trace_enabled,
                store_document_text=config.store_document_text,
            ),
        )

    @app.get(
        "/metrics",
        summary="Prometheus metrics endpoint",
        description="Exports metrics in Prometheus format for scraping",
        tags=["Monitoring"],
    )
    async def metrics_endpoint():
        """Return Prometheus metrics."""
        from fastapi.responses import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        # Rendering can be expensive; keep the event loop responsive during a scrape.
        metrics_data = await asyncio.to_thread(generate_latest)
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
