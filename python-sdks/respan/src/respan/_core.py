"""Respan -- unified entry point for tracing and instrumentation plugins."""

import logging
import os
from typing import Any, Dict, Optional, Sequence

from respan_tracing import RespanTelemetry

from ._types import Instrumentation

logger = logging.getLogger(__name__)


class Respan:
    """Unified entry point for Respan tracing and instrumentation plugins.

    Sets up:
    1. ``RespanTelemetry`` -- OTEL TracerProvider for decorators and, when no
       plugins are provided, auto-instrumentation of LLM SDKs (OpenAI,
       Anthropic, etc.) via the OTEL pipeline.
    2. Activates any instrumentors passed via the ``instrumentations`` list.
    3. Registers OTLP enrichers for plugins that need them (e.g. Google ADK).

    When ``instrumentations`` are provided, OTEL auto-instrumentation is
    disabled by default to avoid duplicate spans (plugins capture LLM calls
    themselves).  Override with ``auto_instrument=True`` if you need both.

    Args:
        api_key: Respan API key. Falls back to ``RESPAN_API_KEY`` env var.
        base_url: Respan API base URL. Falls back to ``RESPAN_BASE_URL`` env var.
        app_name: Application name for telemetry identification.
        instrumentations: List of instrumentor instances to activate.
        auto_instrument: Auto-instrument LLM SDKs (OpenAI, Anthropic, etc.)
            via OTEL.  Defaults to ``True`` when no plugins are provided,
            ``False`` when plugins are provided (to avoid duplicate spans).
        customer_identifier: Default customer/user identifier for all spans.
        thread_identifier: Default conversation thread ID for all spans.
        metadata: Default metadata dict merged into all spans.
        environment: Default environment (e.g. ``"production"``).
        **telemetry_kwargs: Extra keyword arguments forwarded to
            ``RespanTelemetry`` (e.g. ``log_level``, ``is_batching_enabled``).

    Examples::

        # Direct LLM SDK usage -- auto-instruments OpenAI, Anthropic, etc.
        respan = Respan()

        # With plugins -- plugins handle tracing, auto-instrumentation off
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
        respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_name: str = "respan",
        instrumentations: Optional[Sequence[object]] = None,
        auto_instrument: Optional[bool] = None,
        customer_identifier: Optional[str] = None,
        thread_identifier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        **telemetry_kwargs,
    ):
        api_key = api_key or os.getenv("RESPAN_API_KEY")
        base_url = base_url or os.getenv(
            "RESPAN_BASE_URL", "https://api.respan.ai/api"
        )

        # Extract config from instrumentors (so instrumentor-level config
        # is used as fallback if Respan-level isn't set)
        for inst in instrumentations or []:
            if not customer_identifier:
                customer_identifier = getattr(inst, "customer_identifier", None)
            if not environment:
                environment = getattr(inst, "environment", None)

        # Auto-instrument LLM SDKs when no plugins are provided,
        # disable when plugins handle tracing to avoid duplicate spans.
        if auto_instrument is None:
            auto_instrument = not bool(instrumentations)

        # Pass environment and identifiers as resource attributes so they're
        # available in the OTLP pipeline.
        ra = telemetry_kwargs.get("resource_attributes") or {}
        if environment:
            ra["deployment.environment"] = environment
        if customer_identifier:
            ra["respan.customer_params.customer_identifier"] = customer_identifier
        if thread_identifier:
            ra["respan.threads.thread_identifier"] = thread_identifier
        if ra:
            telemetry_kwargs["resource_attributes"] = ra

        self.telemetry = RespanTelemetry(
            app_name=app_name,
            api_key=api_key,
            base_url=base_url,
            **telemetry_kwargs,
        )

        # Activate instrumentations and register OTLP enrichers
        self._instrumentations: Dict[str, object] = {}
        for inst in instrumentations or []:
            name = getattr(inst, "name", type(inst).__name__)
            self._activate(name, inst)
            # Register OTLP enrichers for plugins that enrich spans
            if name == "google-adk":
                from respan_tracing.exporters.adk_enrichment import _enrich_adk_spans
                self.telemetry.register_enricher(_enrich_adk_spans)

    def _activate(self, name: str, inst: object) -> None:
        """Activate a single instrumentor."""
        try:
            if hasattr(inst, "activate"):
                inst.activate()  # type: ignore[union-attr]
            self._instrumentations[name] = inst
            logger.info("Activated instrumentation: %s", name)
        except Exception as exc:
            logger.warning("Failed to activate instrumentation %s: %s", name, exc)

    def flush(self) -> None:
        """Flush the OTEL pipeline."""
        self.telemetry.flush()

    def shutdown(self) -> None:
        """Deactivate plugins and shut down exporters."""
        for name, inst in self._instrumentations.items():
            try:
                if hasattr(inst, "deactivate"):
                    inst.deactivate()  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("Error deactivating %s: %s", name, exc)
        self._instrumentations.clear()
