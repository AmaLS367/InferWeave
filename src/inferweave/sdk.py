"""Main InferWeave SDK client entrypoint."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from inferweave.adapters.auth import default_endpoint_auth
from inferweave.clients.inference import InferenceClient
from inferweave.clients.transport import InferenceConfig, InferenceTransport
from inferweave.core.exceptions import (
    AmbiguousDeploymentError,
    DeploymentNotActiveError,
    DeploymentNotFoundError,
    HealthcheckError,
    HealthcheckTimeoutError,
)
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.healthcheck import ProbeResult
from inferweave.domain.lifecycle import AutostopPolicy
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState
from inferweave.models.profile import ModelProfile
from inferweave.ports.auth import EndpointAuthPort
from inferweave.providers.base import ComputeProvider
from inferweave.providers.router import ProviderRouter
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel type distinguishing "argument not passed" from an explicit ``None``."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()
DEFAULT_DESTROY_AFTER_IDLE_MINS = 30


class InferWeave:
    """Unified AI inference deployment orchestrator across GPU cloud providers."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        router: ProviderRouter | None = None,
        healthcheck_service: HealthcheckService | None = None,
        lifecycle_service: LifecycleService | None = None,
        endpoint_auth: EndpointAuthPort | None = None,
        inference_config: InferenceConfig | None = None,
        inference_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Creates the SDK.

        Args:
            endpoint_auth: Resolves endpoint credentials (e.g. Modal proxy tokens) for readiness
                probes and inference calls. Defaults to environment-based resolution
                (``MODAL_PROXY_TOKEN_ID`` / ``MODAL_PROXY_TOKEN_SECRET``). Secrets are never
                persisted in deployment records.
            inference_config: Timeout/retry policy for ``synthesize()``/``render()``.
            inference_http_client: Optional shared ``httpx.AsyncClient`` (mainly for tests).
        """
        self.endpoint_auth: EndpointAuthPort = endpoint_auth or default_endpoint_auth()
        self.registry = registry or ModelRegistry()
        self.router = router or ProviderRouter(endpoint_auth=self.endpoint_auth)
        if router is not None and endpoint_auth is not None:
            self.router.adopt_endpoint_auth(endpoint_auth)
        self.healthcheck_service = healthcheck_service or HealthcheckService()
        self.lifecycle_service = lifecycle_service or LifecycleService(
            healthcheck_service=self.healthcheck_service,
            provider_resolver=self.router.get,
            endpoint_auth=self.endpoint_auth,
        )
        if self.lifecycle_service.endpoint_auth is None:
            self.lifecycle_service.endpoint_auth = self.endpoint_auth
        self._inference_config = inference_config
        self._inference_http_client = inference_http_client
        self._active_deployments: dict[str, Deployment] = {}
        self._attach_lock = asyncio.Lock()
        for name in self.router.list_providers():
            self.router.get(name).bind_repository(self.lifecycle_service.repository)

    async def deploy(
        self,
        model: str,
        provider: str = "auto",
        strategy: str = "cheapest",
        gpu_type: str | None = None,
        num_gpus: int | None = None,
        env: dict[str, str] | None = None,
        autostop_mins: int | None | _Unset = _UNSET,
        custom_args: dict[str, Any] | None = None,
        dry_run: bool = False,
        wait_for_ready: bool = True,
        destroy_after_idle_mins: int | None | _Unset = _UNSET,
        scaledown_window_seconds: int | None = None,
    ) -> Deployment:
        """Deploys a model to the requested compute provider.

        Args:
            model: Model identifier from registry (e.g. 'fish-s2-pro', 'meta-llama/Meta-Llama-3-8B-Instruct')
            provider: Target provider ('runpod', 'aws', 'modal', or 'auto')
            strategy: Routing strategy when provider='auto' ('cheapest', 'free_first')
            gpu_type: Optional GPU override ('A100', 'H100', 'L4')
            num_gpus: Optional GPU count override
            env: Custom environment variables
            autostop_mins: Legacy alias of ``destroy_after_idle_mins`` (kept for backward
                compatibility). Default 30.
            custom_args: Provider-specific extra configurations
            dry_run: When True, constructs the configuration without launching live cloud resources
            wait_for_ready: When True, actively polls endpoint until readiness healthcheck passes
            destroy_after_idle_mins: InferWeave's full-destroy timer: after this many idle
                minutes the lifecycle watchdog stops the whole deployment (for Modal: the app is
                stopped and the endpoint disappears). ``None``/``0`` disables it. Activity (such
                as ``synthesize()``/``render()`` calls) is persisted, so the timer survives
                process restarts. Pass this *or* ``autostop_mins``, not different values of both.
            scaledown_window_seconds: Modal scale-to-zero window: idle seconds before GPU
                containers shut down while the app stays deployed and wakes on the next request.
                Independent of ``destroy_after_idle_mins`` (default 1800s when unset). Ignored by
                providers without scale-to-zero.

        Returns:
            Deployment: Live deployment handle with lifecycle controls and endpoint details.
        """
        if (
            not isinstance(destroy_after_idle_mins, _Unset)
            and not isinstance(autostop_mins, _Unset)
            and destroy_after_idle_mins != autostop_mins
        ):
            raise ValueError(
                "autostop_mins is a legacy alias of destroy_after_idle_mins; "
                f"got conflicting values {autostop_mins!r} and {destroy_after_idle_mins!r}."
            )
        effective_mins: int | None
        if not isinstance(destroy_after_idle_mins, _Unset):
            effective_mins = destroy_after_idle_mins
        elif not isinstance(autostop_mins, _Unset):
            effective_mins = autostop_mins
        else:
            effective_mins = DEFAULT_DESTROY_AFTER_IDLE_MINS
        custom_args = dict(custom_args or {})
        if scaledown_window_seconds is not None:
            custom_args["scaledown_window_seconds"] = scaledown_window_seconds

        request = DeploymentRequest(
            model=model,
            provider=provider,
            strategy=strategy,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            env=env or {},
            autostop_mins=effective_mins,
            custom_args=custom_args,
            dry_run=dry_run,
            wait_for_ready=wait_for_ready,
        )

        # 1. Resolve model profile
        profile = self.registry.get(model)

        # 2. Pre-flight hardware validation if explicit GPU is requested
        if request.gpu_type:
            self.router.hardware_service.validate_deployment_hardware(profile, request)

        # 3. Resolve target compute provider (SkyPilot clouds or Modal) with VRAM-aware routing
        compute_provider = await self.router.aresolve(
            provider_name=request.provider,
            profile=profile,
            strategy=request.strategy,
            request=request,
        )

        # 4. Render runtime specification
        runtime_template = get_runtime_template(profile.default_runtime)
        runtime_spec = runtime_template.render(profile, request)

        # 5. Provision and launch deployment
        deployment = await compute_provider.deploy(
            request=request,
            profile=profile,
            runtime=runtime_spec,
        )

        try:
            # 6. Wire lifecycle management and autostop callbacks
            autostop_policy = (
                request.options.autostop
                if request.options
                else AutostopPolicy(idle_minutes=request.autostop_mins)
            )
            await self.lifecycle_service.register_deployment(
                deployment=deployment,
                policy=autostop_policy,
                is_dry_run=request.dry_run,
                provider=compute_provider,
                workload_type=profile.workload_type,
                options=request.options,
            )
            self._wire_deployment(
                deployment,
                profile=profile,
                provider=compute_provider,
                autostop_mins=autostop_policy.idle_minutes,
                cleanup_on_failure=bool(
                    request.options and getattr(request.options, "cleanup_on_failure", False)
                ),
                enable_inference=not request.dry_run,
            )

            # 7. Track active deployment
            self._active_deployments[deployment.id] = deployment

            # 8. Actively poll for readiness if wait_for_ready is enabled
            if (
                request.wait_for_ready
                and not request.dry_run
                and profile.healthcheck.enabled
            ):
                await deployment.wait_for_ready()

        except BaseException:
            if compute_provider.cleanup_failed_deployment:
                try:
                    await compute_provider.stop(deployment.id)
                except Exception:  # noqa: BLE001 - cleanup must preserve the original failure
                    logger.error("Failed to clean up deployment %s; retry stop using its ID.", deployment.id)
            raise

        return deployment

    def _auth_headers_for(self, deployment: Deployment) -> dict[str, str]:
        """Resolves runtime endpoint credentials for a deployment (never persisted)."""
        return self.endpoint_auth.headers_for(deployment.provider, deployment.endpoint_url)

    def _wire_deployment(
        self,
        deployment: Deployment,
        profile: ModelProfile,
        provider: ComputeProvider,
        autostop_mins: int | None,
        cleanup_on_failure: bool,
        enable_inference: bool = True,
    ) -> None:
        """Connects a Deployment handle to lifecycle, healthcheck, readiness and inference.

        Shared by ``deploy()`` (fresh deployments) and ``attach()`` (recovered ones), so both
        produce identically behaving handles.
        """
        lifecycle = self.lifecycle_service
        deployment._autostop_mins = autostop_mins
        deployment._workload_type = profile.workload_type
        deployment._record_activity_fn = lambda: lifecycle.record_activity(deployment.id)
        deployment._is_idle_fn = lambda: lifecycle.is_idle(deployment.id)
        deployment._last_activity_fn = lambda: (
            state.last_activity_at
            if (state := lifecycle.get_state(deployment.id)) is not None
            else None
        )
        deployment._stop_fn = lambda action=None: lifecycle.stop_deployment(
            deployment_id=deployment.id,
            action=action,
            provider=provider,
        )
        deployment._refresh_fn = lambda: lifecycle.refresh_status(
            deployment_id=deployment.id,
            provider=provider,
            probe=True,
            healthcheck_config=profile.healthcheck,
        )

        async def _check_health() -> ProbeResult:
            if not deployment.endpoint_url:
                raise HealthcheckError(
                    f"Deployment '{deployment.id}' has no endpoint URL available.",
                    deployment_id=deployment.id,
                )
            auth_headers = self._auth_headers_for(deployment)
            if auth_headers:
                result = await self.healthcheck_service.check_health(
                    endpoint_url=deployment.endpoint_url,
                    config=profile.healthcheck,
                    extra_headers=auth_headers,
                )
            else:
                result = await self.healthcheck_service.check_health(
                    endpoint_url=deployment.endpoint_url,
                    config=profile.healthcheck,
                )
            if result.is_healthy:
                deployment.record_activity()
            return result

        async def _wait_for_ready(timeout_secs: int | None = None) -> DeploymentStatus:
            try:
                await self.healthcheck_service.wait_for_ready(
                    endpoint_url=deployment.endpoint_url,
                    config=profile.healthcheck,
                    deployment_id=deployment.id,
                    timeout_override=float(timeout_secs)
                    if timeout_secs is not None
                    else None,
                    headers_provider=lambda: self._auth_headers_for(deployment),
                )
                deployment._status.state = DeploymentState.HEALTHY
                now = datetime.now(UTC)
                deployment._status.ready_at = now
                deployment.record_activity()
                rec = await lifecycle.get_record(deployment.id)
                if rec:
                    rec.mark_healthy(endpoint_url=deployment.endpoint_url, now=now)
                    await lifecycle.repository.save(rec)
            except HealthcheckTimeoutError as err:
                deployment._status.state = DeploymentState.FAILED
                deployment._status.error_message = str(err)
                rec = await lifecycle.get_record(deployment.id)
                if rec:
                    rec.mark_failed(error_message=str(err))
                    if deployment.endpoint_url:
                        rec.endpoint_url = deployment.endpoint_url
                    await lifecycle.repository.save(rec)
                if cleanup_on_failure:
                    logger.warning(
                        "Readiness timeout occurred for '%s'. cleanup_on_failure=True, terminating deployment.",
                        deployment.id,
                    )
                    try:
                        await deployment.stop()
                    except Exception as stop_err:  # noqa: BLE001
                        logger.warning(
                            "Failed to cleanup failed deployment '%s': %s",
                            deployment.id,
                            stop_err,
                        )
                raise
            return deployment._status

        deployment._healthcheck_fn = _check_health
        deployment._wait_ready_fn = _wait_for_ready

        if enable_inference:
            transport = InferenceTransport(
                deployment_id=deployment.id,
                endpoint_fn=lambda: deployment.endpoint_url,
                auth_headers_fn=lambda: self._auth_headers_for(deployment),
                config=self._inference_config,
                http_client=self._inference_http_client,
                default_port=profile.healthcheck.port,
            )
            deployment._inference_client = InferenceClient(
                deployment_id=deployment.id,
                workload_type=profile.workload_type,
                transport=transport,
                on_activity=lambda: lifecycle.touch_activity(deployment.id),
                on_request_start=lambda: lifecycle.begin_request(deployment.id),
                on_request_end=lambda: lifecycle.end_request(deployment.id),
            )

    async def attach(self, deployment_id: str) -> Deployment:
        """Recovers a live ``Deployment`` handle for a persisted deployment.

        Intended for long-lived services that restart: the handle is rebuilt from the
        persisted ``DeploymentRecord`` and behaves like the one returned by ``deploy()``
        (refresh, stop, health/readiness checks, inference, activity tracking). Identity,
        endpoint, timestamps, options and lifecycle state are preserved, and the idle destroy
        timer resumes from the persisted last activity instead of restarting. No cloud call is
        made; use ``await deployment.refresh()`` to reconcile with the provider.

        If a handle for this deployment is already tracked in this process it is returned as is.

        Raises:
            DeploymentNotFoundError: no persisted record for ``deployment_id``.
            DeploymentNotActiveError: the deployment is stopped, failed, or a dry run.
            ModelNotFoundError: the deployment's model profile is not registered in this
                process (call ``register_model`` for custom models before attaching).
            ProviderNotFoundError: the deployment's provider is unknown to this SDK instance.
        """
        async with self._attach_lock:
            record = await self.lifecycle_service.get_record(deployment_id)
            if record is None:
                raise DeploymentNotFoundError(deployment_id)
            if not record.is_attachable():
                reason = (
                    "it is a dry-run deployment with no live endpoint"
                    if record.is_dry_run
                    else f"it is {record.state.value}"
                )
                raise DeploymentNotActiveError(deployment_id, reason)

            existing = self._active_deployments.get(deployment_id)
            if existing is not None and existing.state != DeploymentState.STOPPED:
                return existing

            profile = self.registry.get(record.model)
            provider = self.router.get(record.provider)
            deployment = Deployment(
                status=DeploymentStatus(
                    id=record.id,
                    model=record.model,
                    provider=record.provider,
                    state=record.state,
                    endpoint_url=record.endpoint_url,
                    error_message=record.error_message,
                    created_at=record.created_at,
                    ready_at=record.ready_at,
                ),
            )
            self._wire_deployment(
                deployment,
                profile=profile,
                provider=provider,
                autostop_mins=record.options.autostop.idle_minutes,
                cleanup_on_failure=record.options.cleanup_on_failure,
            )
            await self.lifecycle_service.rehydrate_deployment(
                record, deployment, provider=provider
            )
            self._active_deployments[deployment.id] = deployment
            return deployment

    async def find(
        self,
        model: str | None = None,
        provider: str | None = None,
    ) -> Deployment | None:
        """Finds the single active persisted deployment matching ``model``/``provider``.

        Returns an attached handle (see ``attach``), or ``None`` when nothing matches.
        Stopped, failed and dry-run deployments are never considered.

        Raises:
            AmbiguousDeploymentError: more than one active deployment matches; the error lists
                the candidate IDs so the caller can ``attach()`` the right one explicitly.
        """
        matches = [
            rec
            for rec in await self.lifecycle_service.list_records()
            if rec.is_attachable()
            and (model is None or rec.model == model)
            and (provider is None or rec.provider.lower() == provider.lower())
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise AmbiguousDeploymentError(model, provider, [rec.id for rec in matches])
        return await self.attach(matches[0].id)

    async def close(self) -> None:
        """Releases inference connection pools, watchdog tasks and probe resources."""
        for deployment in self._active_deployments.values():
            if deployment._inference_client is not None:
                await deployment._inference_client.aclose()
        await self.lifecycle_service.close()
        await self.healthcheck_service.close()

    def register_model(self, profile: ModelProfile) -> None:
        """Registers a custom model profile in the registry."""
        self.registry.register(profile)

    def list_deployments(self) -> list[Deployment]:
        """Returns all actively tracked deployments in the local session."""
        return list(self._active_deployments.values())

    async def list_records(self) -> list[DeploymentRecord]:
        """Returns all persisted deployment records across sessions and CLI invocations."""
        return await self.lifecycle_service.list_records()

    async def stop(
        self,
        deployment_id: str,
        action: Any | None = None,
    ) -> None:
        """Terminates or pauses an active deployment by ID."""
        await self.lifecycle_service.stop_deployment(
            deployment_id=deployment_id,
            action=action,
        )
        dep = self._active_deployments.get(deployment_id)
        if dep:
            dep._status.state = DeploymentState.STOPPED

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves and reconciles the latest deployment status.

        Infrastructure state alone never yields HEALTHY; when the deployment's model profile
        is known, its readiness healthcheck is probed to establish application health.
        """
        healthcheck_config = None
        record = await self.lifecycle_service.get_record(deployment_id)
        if record is not None:
            try:
                healthcheck_config = self.registry.get(record.model).healthcheck
            except Exception as err:  # noqa: BLE001
                logger.debug(
                    "No model profile for '%s'; skipping readiness probe: %s",
                    record.model,
                    err,
                )
        return await self.lifecycle_service.refresh_status(
            deployment_id=deployment_id,
            healthcheck_config=healthcheck_config,
        )
