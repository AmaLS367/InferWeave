"""Unit tests for ModalProvider with a mocked Modal SDK.

These tests validate the adapter's translation, image building, per-account credentials and
error classification against a mocked Modal SDK, without executing live deployments on Modal.
"""

import inspect
import os
import tomllib
from pathlib import Path
from threading import get_ident
from unittest.mock import MagicMock, patch

import pytest
from packaging.requirements import Requirement
from support import modal_pool, modal_record

from inferweave.accounts import AccountManager, ProviderAccount
from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.core.exceptions import (
    NoAccountAvailableError,
    ProviderAuthError,
    ProviderOperationError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers import modal_provider as modal_provider_module
from inferweave.providers.modal_provider import ModalProvider
from inferweave.runtimes.base import RuntimeSpec
from inferweave.services.provisioning_service import (
    ProvisionCandidate,
    ProvisioningService,
)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
AMBIENT = ProviderAccount.ambient("modal")


@pytest.fixture
def sample_profile() -> ModelProfile:
    return ModelProfile(
        id="test/sample-model",
        name="Sample Model",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(
            min_vram_gb=16,
            gpu_count=1,
            recommended_gpus=["A10G"],
        ),
        healthcheck=HealthcheckConfig(port=8000),
    )


@pytest.fixture
def sample_runtime() -> RuntimeSpec:
    return RuntimeSpec(
        name="vllm",
        docker_image="vllm/vllm-openai:latest",
        setup_commands=["echo 'setting up'"],
        run_command="python -m vllm --port 8000",
        port=8000,
        env_vars={"MODEL": "test/sample-model"},
    )


def _request(profile: ModelProfile, **kwargs) -> DeploymentRequest:
    return DeploymentRequest(model=profile.id, provider="modal", **kwargs)


def _candidate(provider, profile, runtime, **kwargs) -> ProvisionCandidate:
    return ProvisionCandidate(
        provider=provider, request=_request(profile, **kwargs), runtime=runtime
    )


def _modal_error(name: str, message: str = "boom") -> Exception:
    return type(name, (Exception,), {})(message)


async def _provision(provider, profile, runtime, account=AMBIENT, **request_kwargs):
    """Runs ``provider.provision`` for a fresh record with Modal's network edges mocked."""
    request = _request(profile, **request_kwargs)
    record = modal_record(provider, profile, account, endpoint_url=None)
    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://live.modal.run"),
    ):
        result = await provider.provision(record, request, profile, runtime, account)
    return record, result


@pytest.mark.asyncio
async def test_modal_provider_provision_success(sample_profile, sample_runtime):
    """Provisioning deploys the app (off the event loop) and reports STARTING, not HEALTHY."""
    provider = ModalProvider()
    request = _request(sample_profile, gpu_type="A100", num_gpus=1, autostop_mins=15)
    record = modal_record(provider, sample_profile, AMBIENT, endpoint_url=None)
    event_loop_thread = get_ident()

    def get_web_url():
        assert get_ident() != event_loop_thread
        return "https://workspace--iw-modal-test.modal.run"

    with (
        patch("modal.App.deploy", return_value=None) as mock_deploy,
        patch("modal.Function.get_web_url", side_effect=get_web_url),
    ):
        result = await provider.provision(record, request, sample_profile, sample_runtime, AMBIENT)

    mock_deploy.assert_called_once()
    assert record.id.startswith("iw-modal-")
    assert mock_deploy.call_args.kwargs["name"] == record.id
    assert result.endpoint_url == "https://workspace--iw-modal-test.modal.run"
    # app.deploy() succeeding does not prove runtime readiness
    assert result.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_modal_provider_dry_run_makes_no_modal_call(sample_profile, sample_runtime):
    """Dry runs go through the provisioning service: no account, no Modal SDK interaction."""
    provider = ModalProvider()
    service = ProvisioningService(AccountManager(), InMemoryDeploymentRepository())
    candidate = _candidate(provider, sample_profile, sample_runtime, dry_run=True)
    with (
        patch("modal.App.deploy") as mock_deploy,
        patch("modal.Client.from_credentials") as mock_client,
    ):
        record, chosen = await service.provision([candidate], sample_profile)

    mock_deploy.assert_not_called()
    mock_client.assert_not_called()
    assert chosen is provider
    assert record.is_dry_run and record.account is None
    assert record.state == DeploymentState.PROVISIONING
    assert "dryrun" in (record.endpoint_url or "")


def test_modal_resource_ref_uses_account_environment(sample_profile):
    provider = ModalProvider()
    request = _request(sample_profile)
    manager = modal_pool("a", environment="staging")
    ref = provider.resource_ref("iw-modal-x", request, manager.resolve("modal", "a"))
    assert (ref.name, ref.scope) == ("iw-modal-x", "staging")
    assert provider.resource_ref("iw-modal-x", request, AMBIENT).scope is None


@pytest.mark.asyncio
@pytest.mark.parametrize("has_setup_commands", [True, False])
async def test_modal_image_build_steps_do_not_follow_local_mounts(
    sample_profile, sample_runtime, has_setup_commands
):
    """Validate Modal's build-before-mount contract without contacting the cloud."""
    import modal

    run_commands = modal.Image.run_commands
    env = modal.Image.env
    entrypoint = modal.Image.entrypoint
    add_source = modal.Image.add_local_python_source
    function = modal.App.function
    images = []
    mounted_images = set()
    entrypoints = []
    if not has_setup_commands:
        sample_runtime = sample_runtime.model_copy(update={"setup_commands": []})

    def mounted_source(image, *args, **kwargs):
        mounted = add_source(image, *args, **kwargs)
        mounted_images.add(mounted)
        return mounted

    def validated_commands(image, *args, **kwargs):
        # Modal rejects these chains during remote image resolution, after mounts load.
        assert image not in mounted_images, "Build commands cannot follow local mounts"
        return run_commands(image, *args, **kwargs)

    def validated_env(image, *args, **kwargs):
        assert image not in mounted_images, "Image ENV cannot follow local mounts"
        return env(image, *args, **kwargs)

    def capture_image(app, *args, **kwargs):
        images.append(kwargs["image"])
        return function(app, *args, **kwargs)

    def validated_entrypoint(image, commands):
        assert image not in mounted_images, "Image ENTRYPOINT cannot follow local mounts"
        entrypoints.append(commands)
        return entrypoint(image, commands)

    with (
        patch("modal.Image.run_commands", validated_commands),
        patch("modal.Image.env", validated_env),
        patch("modal.Image.entrypoint", validated_entrypoint),
        patch("modal.Image.add_local_python_source", mounted_source),
        patch("modal.App.function", capture_image),
    ):
        await _provision(ModalProvider(), sample_profile, sample_runtime)
    assert len(images) == 1 and images[0] in mounted_images
    assert entrypoints == [[]]  # InferWeave's structured argv starts the server.


@pytest.mark.asyncio
async def test_modal_provider_applies_runtime_registry_setup(sample_profile, sample_runtime):
    """Bootstrap directives reach registry import before Modal inspects Python."""
    import modal

    commands = ["USER root", "RUN ln -sf /usr/bin/python3 /usr/local/bin/python"]
    sample_runtime = sample_runtime.model_copy(
        update={"metadata": {"modal_setup_dockerfile_commands": commands}}
    )
    with patch("modal.Image.from_registry", wraps=modal.Image.from_registry) as from_registry:
        await _provision(ModalProvider(), sample_profile, sample_runtime)
    from_registry.assert_called_once_with(
        sample_runtime.docker_image, setup_dockerfile_commands=commands
    )


# --- stop / status / exists -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [None, AutostopAction.STOP, AutostopAction.DOWN])
async def test_modal_provider_stop_uses_public_stop_app(action, sample_profile):
    """STOP and DOWN both map to Modal's single app-level stop (public SDK API)."""
    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)

    with patch("modal.experimental.stop_app") as mock_stop:
        if action is None:
            await provider.stop(record, AMBIENT)
        else:
            await provider.stop(record, AMBIENT, action=action)

    mock_stop.assert_called_once_with(record.id, environment_name=None, client=None)


@pytest.mark.asyncio
async def test_modal_provider_is_stateless_across_instances(sample_profile):
    """Cross-process stop: a fresh provider needs nothing but the persisted record + account."""
    record = modal_record(ModalProvider(), sample_profile, AMBIENT)
    fresh = ModalProvider()
    with patch("modal.experimental.stop_app") as mock_stop:
        await fresh.stop(record, AMBIENT)
    mock_stop.assert_called_once()


@pytest.mark.asyncio
async def test_modal_provider_stop_is_idempotent_when_app_already_gone(sample_profile):
    import modal

    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)

    with patch("modal.experimental.stop_app", side_effect=modal.exception.NotFoundError("gone")):
        await provider.stop(record, AMBIENT)  # must not raise


@pytest.mark.asyncio
async def test_modal_provider_status_deployed_app_is_starting_not_healthy(sample_profile):
    """A deployed Modal app is infrastructure state only; it must never read as HEALTHY."""
    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)
    with patch("modal.App.lookup", return_value=MagicMock()) as mock_lookup:
        status = await provider.status(record, AMBIENT)

    mock_lookup.assert_called_once_with(record.id, environment_name=None, client=None)
    assert status.state == DeploymentState.STARTING
    assert status.endpoint_url == record.endpoint_url


@pytest.mark.asyncio
async def test_modal_provider_status_missing_app_is_stopped(sample_profile):
    import modal

    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)
    with patch("modal.App.lookup", side_effect=modal.exception.NotFoundError("gone")):
        status = await provider.status(record, AMBIENT)
        exists = await provider.resource_exists(record, AMBIENT)

    assert status.state == DeploymentState.STOPPED
    assert exists is False


@pytest.mark.asyncio
async def test_modal_provider_status_transient_error_is_classified(sample_profile):
    """Control-plane hiccups surface as TRANSIENT so the lifecycle keeps the last known state."""
    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)

    with (
        patch("modal.App.lookup", side_effect=OSError("network down")),
        pytest.raises(ProviderOperationError) as exc_info,
    ):
        await provider.status(record, AMBIENT)

    assert exc_info.value.kind is FailureKind.TRANSIENT


def test_modal_provider_uses_only_public_modal_apis():
    """Private Modal internals (modal.cli, modal.client._Client, modal_proto) must not be used."""
    source = inspect.getsource(modal_provider_module)
    for private in ("modal_proto", "modal.cli", "_Client", "resolve_app_identifier", "api_pb2"):
        assert private not in source, f"modal_provider relies on private API: {private}"


def test_modal_public_surface_required_by_provider_exists():
    """Guards the installed Modal SDK still exposes the public APIs the provider calls."""
    import modal

    assert callable(modal.experimental.stop_app)
    assert callable(modal.App.lookup)
    assert callable(modal.Client.from_credentials)
    assert issubclass(modal.exception.NotFoundError, Exception)


def test_modal_extra_supports_current_stable_sdk():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    (spec,) = [
        Requirement(dep)
        for dep in project["optional-dependencies"]["modal"]
        if dep.lower().startswith("modal")
    ]
    assert spec.specifier.contains("1.6.0"), "current stable Modal must be allowed"
    assert spec.specifier.contains("1.6.5"), "patch updates in verified 1.6 series allowed"
    assert not spec.specifier.contains("1.5.9"), "unverified pre-1.6 Modal must stay excluded"
    assert not spec.specifier.contains("1.7.0"), "untested 1.7+ Modal must stay excluded"
    assert not spec.specifier.contains("2.0.0"), "untested major version 2.0 must stay excluded"


@pytest.mark.asyncio
async def test_modal_stop_capability_detection(sample_profile):
    """A Modal SDK without a public stop API fails loudly instead of silently doing nothing."""
    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)

    # 1. Standard modal.experimental.stop_app
    mock_modal = MagicMock()
    mock_modal.experimental.stop_app = MagicMock()
    mock_modal.exception.NotFoundError = type("NotFoundError", (Exception,), {})
    with patch.object(provider, "_get_modal_module", return_value=mock_modal):
        await provider._stop_modal_app(record, AMBIENT)
        mock_modal.experimental.stop_app.assert_called_once_with(
            record.id, environment_name=None, client=None
        )

    # 2. Missing stop API is a classified, non-retryable operation error
    mock_modal3 = MagicMock(spec=["exception"])
    mock_modal3.exception.NotFoundError = type("NotFoundError", (Exception,), {})
    with (
        patch.object(provider, "_get_modal_module", return_value=mock_modal3),
        pytest.raises(ProviderOperationError, match="supported public app stop API") as exc_info,
    ):
        await provider._stop_modal_app(record, AMBIENT)
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST


@pytest.mark.asyncio
async def test_modal_provider_stop_failure_propagates(sample_profile):
    """Modal API failures propagate (classified) so the lifecycle keeps the deployment active."""
    provider = ModalProvider()
    record = modal_record(provider, sample_profile, AMBIENT)

    with (
        patch("modal.experimental.stop_app", side_effect=RuntimeError("Modal API unreachable")),
        pytest.raises(ProviderOperationError) as exc_info,
    ):
        await provider.stop(record, AMBIENT)

    assert exc_info.value.kind is FailureKind.TRANSIENT
    assert "RuntimeError" in str(exc_info.value)
    assert "Modal API unreachable" not in str(exc_info.value)
    assert record.state == DeploymentState.STARTING  # provider never mutates the record
    assert record.stopped_at is None


# --- per-account credentials ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pooled_accounts_each_use_their_own_modal_client(sample_profile, sample_runtime):
    """Two pooled accounts: each gets Client.from_credentials with its own token pair."""
    manager = modal_pool("a", "b", environment="env-x")
    provider = ModalProvider()
    clients: dict[tuple[str, str], object] = {}

    def make_client(token_id, token_secret):
        return clients.setdefault((token_id, token_secret), MagicMock(name=token_id))

    with (
        patch("modal.Client.from_credentials", side_effect=make_client) as from_credentials,
        patch("modal.App.deploy", return_value=None) as mock_deploy,
        patch("modal.Function.get_web_url", return_value="https://live.modal.run"),
        patch("modal.App.lookup", return_value=MagicMock()) as mock_lookup,
        patch("modal.experimental.stop_app") as mock_stop,
    ):
        for acct_id in ("a", "b"):
            account = manager.resolve("modal", acct_id)
            record = modal_record(provider, sample_profile, account)
            request = _request(sample_profile)
            await provider.provision(record, request, sample_profile, sample_runtime, account)
            await provider.resource_exists(record, account)
            await provider.stop(record, account)

            expected = clients[(f"SENTINEL-modal-{acct_id}-id", f"SENTINEL-modal-{acct_id}-secret")]
            for call in (mock_deploy.call_args, mock_lookup.call_args, mock_stop.call_args):
                assert call.kwargs["client"] is expected
                assert call.kwargs["environment_name"] == "env-x"

    assert sorted(c.args for c in from_credentials.call_args_list) == [
        ("SENTINEL-modal-a-id", "SENTINEL-modal-a-secret"),
        ("SENTINEL-modal-b-id", "SENTINEL-modal-b-secret"),
    ]
    assert (
        clients[("SENTINEL-modal-a-id", "SENTINEL-modal-a-secret")]
        is not clients[("SENTINEL-modal-b-id", "SENTINEL-modal-b-secret")]
    )


@pytest.mark.asyncio
async def test_ambient_account_passes_no_client_and_builds_none(sample_profile, sample_runtime):
    provider = ModalProvider()
    with (
        patch("modal.Client.from_credentials") as from_credentials,
        patch("modal.App.deploy", return_value=None) as mock_deploy,
        patch("modal.Function.get_web_url", return_value="https://live.modal.run"),
        patch("modal.App.lookup", return_value=MagicMock()) as mock_lookup,
        patch("modal.experimental.stop_app") as mock_stop,
    ):
        record = modal_record(provider, sample_profile, AMBIENT)
        await provider.provision(
            record, _request(sample_profile), sample_profile, sample_runtime, AMBIENT
        )
        await provider.resource_exists(record, AMBIENT)
        await provider.stop(record, AMBIENT)

    from_credentials.assert_not_called()
    for call in (mock_deploy.call_args, mock_lookup.call_args, mock_stop.call_args):
        assert call.kwargs["client"] is None
        assert call.kwargs["environment_name"] is None


@pytest.mark.asyncio
async def test_client_is_cached_per_account_and_rebuilt_when_secret_rotates(sample_profile):
    env: dict[str, str] = {}
    manager = modal_pool("a", environ=env)
    provider = ModalProvider()
    created = []

    def make_client(token_id, token_secret):
        client = MagicMock(name=f"{token_id}/{token_secret}")
        created.append((token_id, token_secret, client))
        return client

    with (
        patch("modal.Client.from_credentials", side_effect=make_client),
        patch("modal.App.lookup", return_value=MagicMock()) as mock_lookup,
    ):
        record = modal_record(provider, sample_profile, manager.resolve("modal", "a"))
        await provider.resource_exists(record, manager.resolve("modal", "a"))
        await provider.resource_exists(record, manager.resolve("modal", "a"))
        assert len(created) == 1  # cached across calls for the same account + secrets
        first_client = mock_lookup.call_args.kwargs["client"]

        env["T_A_SECRET"] = "SENTINEL-modal-a-rotated-secret"  # key rotation, same account id
        await provider.resource_exists(record, manager.resolve("modal", "a"))

    assert len(created) == 2
    assert created[1][:2] == ("SENTINEL-modal-a-id", "SENTINEL-modal-a-rotated-secret")
    assert mock_lookup.call_args.kwargs["client"] is created[1][2]
    assert mock_lookup.call_args.kwargs["client"] is not first_client


# --- preflight / proxy auth requirement -------------------------------------------------------


def test_pooled_account_without_proxy_tokens_is_refused_naming_the_account(
    sample_profile, sample_runtime
):
    account = modal_pool("team-a", proxy=False).resolve("modal", "team-a")
    with pytest.raises(ProviderOperationError, match="team-a") as exc_info:
        ModalProvider().preflight(_request(sample_profile), sample_profile, sample_runtime, account)
    assert exc_info.value.kind is FailureKind.PERMISSION
    assert not exc_info.value.resource_may_exist
    assert "requires_proxy_auth" in str(exc_info.value)
    assert "SENTINEL" not in str(exc_info.value)


def test_pooled_account_ignores_ambient_proxy_env_tokens(sample_profile, sample_runtime):
    """The conftest ambient MODAL_PROXY_TOKEN_* must not satisfy a pooled account."""
    assert os.environ["MODAL_PROXY_TOKEN_ID"]
    account = modal_pool("team-a", proxy=False).resolve("modal", "team-a")
    with pytest.raises(ProviderOperationError):
        ModalProvider().preflight(_request(sample_profile), sample_profile, sample_runtime, account)


def test_pooled_account_with_proxy_tokens_passes_preflight(sample_profile, sample_runtime):
    account = modal_pool("team-a").resolve("modal", "team-a")
    ModalProvider().preflight(_request(sample_profile), sample_profile, sample_runtime, account)


def test_pooled_account_may_opt_out_of_proxy_auth(sample_profile, sample_runtime):
    account = modal_pool("team-a", proxy=False).resolve("modal", "team-a")
    request = _request(sample_profile, custom_args={"requires_proxy_auth": False})
    ModalProvider().preflight(request, sample_profile, sample_runtime, account)


def test_ambient_account_without_env_tokens_is_refused(
    sample_profile, sample_runtime, monkeypatch
):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID")
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    with pytest.raises(ProviderAuthError, match="MODAL_PROXY_TOKEN_ID"):
        ModalProvider().preflight(_request(sample_profile), sample_profile, sample_runtime, AMBIENT)


@pytest.mark.asyncio
async def test_misconfigured_pooled_account_fails_locally_before_any_modal_call(
    sample_profile, sample_runtime
):
    manager = modal_pool("only", proxy=False, max_attempts=1)
    service = ProvisioningService(manager, InMemoryDeploymentRepository())
    candidate = _candidate(ModalProvider(), sample_profile, sample_runtime)
    with (
        patch("modal.Client.from_credentials") as from_credentials,
        patch("modal.App.deploy") as mock_deploy,
        pytest.raises(ProviderOperationError, match="only"),
    ):
        await service.provision([candidate], sample_profile)
    from_credentials.assert_not_called()
    mock_deploy.assert_not_called()
    [health] = manager.health()
    assert health.state == "rate_limited" and health.in_flight == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("pin", [None, "a"])
async def test_missing_proxy_tokens_fail_over_or_respect_pinned_account(
    sample_profile, sample_runtime, pin
):
    env = {}
    manager = modal_pool("a", "b", environ=env, strategy="failover")
    del env["T_A_PID"]
    del env["T_A_PSECRET"]
    repository = InMemoryDeploymentRepository()
    service = ProvisioningService(manager, repository)
    candidate = _candidate(ModalProvider(), sample_profile, sample_runtime)
    with (
        patch("modal.Client.from_credentials", side_effect=lambda token_id, _secret: token_id),
        patch("modal.App.deploy") as mock_deploy,
        patch("modal.Function.get_web_url", return_value="https://live.modal.run"),
        patch("modal.App.lookup") as mock_lookup,
    ):
        if pin:
            with pytest.raises(ProviderOperationError) as caught:
                await service.provision([candidate], sample_profile, account=pin)
            assert caught.value.kind is FailureKind.PERMISSION
            mock_deploy.assert_not_called()
            assert await repository.list_all() == []
        else:
            record, _ = await service.provision([candidate], sample_profile)
            assert record.account == "b"
            assert (await service.provision([candidate], sample_profile))[0].account == "b"
            assert all(
                call.kwargs["client"] == "SENTINEL-modal-b-id"
                for call in mock_deploy.call_args_list
            )
            assert len(mock_deploy.call_args_list) == 2
            assert all(r.account == "b" for r in await repository.list_all())
        mock_lookup.assert_not_called()
    health = {h.account_id: h for h in manager.health()}
    assert health["a"].state == "rate_limited"
    assert health["a"].consecutive_failures == 0
    assert all(h.in_flight == 0 for h in health.values())


# --- error classification / redaction ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_name", "kind"),
    [
        ("AuthError", FailureKind.AUTH),
        ("PermissionDeniedError", FailureKind.PERMISSION),
        ("ResourceExhaustedError", FailureKind.RATE_LIMIT),
        ("InvalidError", FailureKind.INVALID_REQUEST),
        ("NotFoundError", FailureKind.INVALID_REQUEST),
        ("ImageBuildError", FailureKind.INVALID_REQUEST),
        ("ConnectionError", FailureKind.TRANSIENT),
        ("SomethingElseError", FailureKind.TRANSIENT),
    ],
)
async def test_modal_errors_are_classified_by_exception_class(
    error_name, kind, sample_profile, sample_runtime
):
    provider = ModalProvider()
    account = modal_pool("a").resolve("modal", "a")
    record = modal_record(provider, sample_profile, account)

    with (
        patch("modal.Client.from_credentials", return_value=MagicMock()),
        patch("modal.App.deploy", side_effect=_modal_error(error_name)),
        pytest.raises(ProviderOperationError) as exc_info,
    ):
        await provider.provision(
            record, _request(sample_profile), sample_profile, sample_runtime, account
        )

    assert exc_info.value.kind is kind
    assert "'a'" in str(exc_info.value)
    assert exc_info.value.deployment_id == record.id
    assert exc_info.value.resource_may_exist is (not kind.is_definitive_rejection)
    assert exc_info.value.operation_may_continue is (not kind.is_definitive_rejection)


@pytest.mark.asyncio
async def test_modal_error_messages_redact_account_secrets(sample_profile, sample_runtime):
    provider = ModalProvider()
    account = modal_pool("a").resolve("modal", "a")
    record = modal_record(provider, sample_profile, account)
    leaked = (
        "auth failed for SENTINEL-modal-a-id with SENTINEL-modal-a-secret "
        "and proxy SENTINEL-modal-a-proxy-secret"
    )

    with (
        patch("modal.Client.from_credentials", return_value=MagicMock()),
        patch("modal.App.deploy", side_effect=_modal_error("AuthError", leaked)),
        pytest.raises(ProviderOperationError) as exc_info,
    ):
        await provider.provision(
            record, _request(sample_profile), sample_profile, sample_runtime, account
        )

    text = str(exc_info.value)
    assert "SENTINEL" not in text
    assert "AuthError" in text
    assert exc_info.value.__cause__ is None  # the raw SDK exception is not chained


@pytest.mark.asyncio
async def test_modal_not_found_on_stop_is_success_for_pooled_account(sample_profile):
    import modal

    provider = ModalProvider()
    account = modal_pool("a").resolve("modal", "a")
    record = modal_record(provider, sample_profile, account)
    with (
        patch("modal.Client.from_credentials", return_value=MagicMock()),
        patch("modal.experimental.stop_app", side_effect=modal.exception.NotFoundError("gone")),
    ):
        await provider.stop(record, account)


@pytest.mark.asyncio
async def test_provisioning_service_fails_over_modal_account_on_auth_error(
    sample_profile, sample_runtime
):
    """End to end with the real service: a rejected key fails over to the next account."""
    manager = modal_pool("a", "b", strategy="failover")
    repository = InMemoryDeploymentRepository()
    service = ProvisioningService(manager, repository)
    candidate = _candidate(ModalProvider(), sample_profile, sample_runtime)
    calls: list[object] = []

    def deploy(*args, **kwargs):
        calls.append(kwargs["client"])
        if len(calls) == 1:
            raise _modal_error("AuthError", "bad token")

    with (
        patch("modal.Client.from_credentials", side_effect=lambda token_id, _secret: token_id),
        patch("modal.App.deploy", side_effect=deploy),
        patch("modal.Function.get_web_url", return_value="https://live.modal.run"),
        patch("modal.App.lookup", side_effect=_modal_error("AuthError", "bad token")) as lookup,
    ):
        record, _ = await service.provision([candidate], sample_profile)

    assert calls == ["SENTINEL-modal-a-id", "SENTINEL-modal-b-id"]
    assert record.account == "b"
    assert record.state == DeploymentState.STARTING
    lookup.assert_not_called()
    health = {h.account_id: h for h in manager.health()}
    assert health["a"].state == "revoked" and health["a"].in_flight == 0
    [failed] = [r for r in await repository.list_all() if r.state == DeploymentState.FAILED]
    assert failed.account == "a"
    assert not failed.needs_reconciliation and not failed.creation_may_continue


@pytest.mark.asyncio
async def test_endpoint_lookup_rejection_after_deploy_still_cleans_up(
    sample_profile, sample_runtime
):
    import modal

    manager = modal_pool("a", "b", strategy="failover")
    repository = InMemoryDeploymentRepository()
    service = ProvisioningService(manager, repository)
    candidate = _candidate(ModalProvider(), sample_profile, sample_runtime)
    with (
        patch("modal.Client.from_credentials", side_effect=lambda token_id, _secret: token_id),
        patch("modal.App.deploy") as mock_deploy,
        patch("modal.Function.get_web_url", side_effect=[
            _modal_error("AuthError"), "https://live.modal.run",
        ]),
        patch("modal.App.lookup", side_effect=[
            MagicMock(), modal.exception.NotFoundError("gone"),
        ]) as lookup,
        patch("modal.experimental.stop_app") as stop,
    ):
        record, _ = await service.provision([candidate], sample_profile)
    assert record.account == "b"
    assert mock_deploy.call_count == 2
    stop.assert_called_once()
    assert stop.call_args.kwargs["client"] == "SENTINEL-modal-a-id"
    assert lookup.call_count == 2
    assert all(c.kwargs["client"] == "SENTINEL-modal-a-id" for c in lookup.call_args_list)
    [failed] = [r for r in await repository.list_all() if r.state == DeploymentState.FAILED]
    assert not failed.needs_reconciliation and not failed.creation_may_continue


@pytest.mark.asyncio
async def test_all_pooled_accounts_rejected_raises_no_account_available(
    sample_profile, sample_runtime
):
    manager = modal_pool("a", "b", strategy="failover", max_attempts=2)
    service = ProvisioningService(manager, InMemoryDeploymentRepository())
    candidate = _candidate(ModalProvider(), sample_profile, sample_runtime)

    with (
        patch("modal.Client.from_credentials", return_value=MagicMock()),
        patch("modal.App.deploy", side_effect=_modal_error("AuthError", "bad")),
        patch("modal.App.lookup", side_effect=_modal_error("AuthError", "bad")) as lookup,
        pytest.raises(NoAccountAvailableError),
    ):
        await service.provision([candidate], sample_profile)
    lookup.assert_not_called()
    assert all(h.state == "revoked" and h.in_flight == 0 for h in manager.health())


# --- custom container support (T1.3a) ---------------------------------------------------------


def _container_profile(
    model_id: str = "custom/container-svc",
    image: str = "ghcr.io/org/custom-inference:1.2.0",
    args: list[str] | None = None,
    port: int = 8080,
    health_path: str = "/v1/ready",
    **kwargs,
) -> ModelProfile:
    if args is None:
        args = ["python", "-m", "custom_server", "--port", str(port)]
    return ModelProfile.for_container(
        id=model_id,
        image=image,
        args=args,
        port=port,
        health_path=health_path,
        **kwargs,
    )


def _render_container(
    profile: ModelProfile, **request_kwargs
) -> tuple[DeploymentRequest, RuntimeSpec]:
    from inferweave.runtimes.templates import get_runtime_template

    request = _request(profile, **request_kwargs)
    template = get_runtime_template(profile.default_runtime)
    return request, template.render(profile, request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_ref",
    [
        "ghcr.io/org/custom-inference:1.2.0",
        "ghcr.io/org/custom-inference@sha256:45b23d06496e983443b4ff5f707f9f423806046399d50f16fbdb4094f43f972b",
    ],
)
async def test_modal_custom_container_image_construction_without_add_python(image_ref: str):
    """ContainerTemplate RuntimeSpec builds Modal image from registry without add_python when omitted."""
    import modal

    profile = _container_profile(
        image=image_ref,
        setup_commands=["pip install uv", "mkdir -p /data"],
        env={"MODEL_DIR": "/data", "LOG_LEVEL": "INFO"},
        metadata={"modal_setup_dockerfile_commands": ["USER root"]},
    )
    request, runtime = _render_container(profile, env={"REQUEST_FLAG": "1"})
    assert runtime.name == "container"
    assert runtime.add_python is None

    provider = ModalProvider()
    record = modal_record(provider, profile, AMBIENT, endpoint_url=None)
    steps: list[tuple[str, object]] = []

    orig_from_registry = modal.Image.from_registry
    orig_entrypoint = modal.Image.entrypoint
    orig_run_commands = modal.Image.run_commands
    orig_env = modal.Image.env
    orig_add_source = modal.Image.add_local_python_source

    def spy_from_registry(tag, **kwargs):
        steps.append(("from_registry", (tag, kwargs)))
        return orig_from_registry(tag, **kwargs)

    def spy_entrypoint(image, commands):
        steps.append(("entrypoint", commands))
        return orig_entrypoint(image, commands)

    def spy_run_commands(image, *cmds, **kwargs):
        steps.append(("run_commands", list(cmds)))
        return orig_run_commands(image, *cmds, **kwargs)

    def spy_env(image, env_dict):
        steps.append(("env", dict(env_dict)))
        return orig_env(image, env_dict)

    def spy_add_source(image, *args, **kwargs):
        steps.append(("add_local_python_source", args))
        return orig_add_source(image, *args, **kwargs)

    with (
        patch("modal.Image.from_registry", side_effect=spy_from_registry) as mock_from_reg,
        patch("modal.Image.entrypoint", autospec=True, side_effect=spy_entrypoint),
        patch("modal.Image.run_commands", autospec=True, side_effect=spy_run_commands),
        patch("modal.Image.env", autospec=True, side_effect=spy_env),
        patch("modal.Image.add_local_python_source", autospec=True, side_effect=spy_add_source),
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://custom.modal.run"),
    ):
        result = await provider.provision(record, request, profile, runtime, AMBIENT)

    assert result.state == DeploymentState.STARTING
    assert result.endpoint_url == "https://custom.modal.run"
    # add_python was None, so it must NOT be passed to from_registry
    mock_from_reg.assert_called_once_with(
        image_ref,
        setup_dockerfile_commands=["USER root"],
    )
    assert "add_python" not in mock_from_reg.call_args.kwargs
    assert [s[0] for s in steps] == [
        "from_registry",
        "entrypoint",
        "run_commands",
        "env",
        "add_local_python_source",
    ]
    assert steps[1][1] == []
    assert steps[2][1] == ["pip install uv", "mkdir -p /data"]
    assert steps[3][1] == {"MODEL_DIR": "/data", "LOG_LEVEL": "INFO", "REQUEST_FLAG": "1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("py_version", ["3.10", "3.11", "3.12", "3.12.3"])
async def test_modal_custom_container_passes_add_python_when_explicitly_provided(
    py_version: str,
):
    """When ContainerSpec.add_python is set, ModalProvider passes add_python to Image.from_registry."""
    import modal

    profile = _container_profile(
        image="nvidia/cuda:12.4.1-base-ubuntu22.04",
        add_python=py_version,
        setup_commands=["pip install fastapi uvicorn"],
        env={"PORT": "8080"},
    )
    request, runtime = _render_container(profile)
    assert runtime.add_python == py_version

    provider = ModalProvider()
    record = modal_record(provider, profile, AMBIENT, endpoint_url=None)
    mounted_images: set[object] = set()
    orig_add_source = modal.Image.add_local_python_source
    orig_run_commands = modal.Image.run_commands
    orig_env = modal.Image.env

    def mounted_source(image, *args, **kwargs):
        mounted = orig_add_source(image, *args, **kwargs)
        mounted_images.add(mounted)
        return mounted

    def validated_commands(image, *args, **kwargs):
        assert image not in mounted_images
        return orig_run_commands(image, *args, **kwargs)

    def validated_env(image, *args, **kwargs):
        assert image not in mounted_images
        return orig_env(image, *args, **kwargs)

    with (
        patch("modal.Image.from_registry", wraps=modal.Image.from_registry) as from_registry,
        patch("modal.Image.run_commands", validated_commands),
        patch("modal.Image.env", validated_env),
        patch("modal.Image.add_local_python_source", mounted_source),
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://py.modal.run"),
    ):
        await provider.provision(record, request, profile, runtime, AMBIENT)

    from_registry.assert_called_once_with(
        "nvidia/cuda:12.4.1-base-ubuntu22.04",
        setup_dockerfile_commands=[],
        add_python=py_version,
    )
    assert len(mounted_images) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_py", ["", "   ", "python3.12", "3.12; rm -rf /", "3.12-rc1", 312])
async def test_modal_rejects_invalid_add_python_on_runtime_spec(bad_py):
    """Direct RuntimeSpec mutation with invalid add_python is rejected before any Modal call."""
    profile = _container_profile()
    request, runtime = _render_container(profile)
    bad_runtime = runtime.model_copy(update={"add_python": bad_py})
    provider = ModalProvider()
    record = modal_record(provider, profile, AMBIENT)

    with pytest.raises(ValueError, match="add_python"):
        provider.preflight(request, profile, bad_runtime, AMBIENT)

    with (
        patch("modal.Image.from_registry") as from_registry,
        patch("modal.App.deploy") as mock_deploy,
        pytest.raises(ValueError, match="add_python"),
    ):
        await provider.provision(record, request, profile, bad_runtime, AMBIENT)

    from_registry.assert_not_called()
    mock_deploy.assert_not_called()
    assert not record.creation_may_continue


def _assert_literal_popen_argv(expected_argv: list[str]) -> None:
    """Verify real subprocess.Popen(..., shell=False) delivers literal argv without shell expansion."""
    import json
    import subprocess
    import sys

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json, sys; print(json.dumps(sys.argv[1:]))",
            *expected_argv,
        ],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0, stderr.decode("utf-8", errors="replace")
    assert json.loads(stdout.decode("utf-8").strip()) == expected_argv


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adversarial_args",
    [
        ["python3", "-m", "server", "--prompt", "; touch /tmp/pwned", "--port", "8080"],
        ["/app/bin/serve", "--path", "dir with spaces/model", "$(rm -rf /)"],
        ["python3", "run.py", "`id`", "| cat /etc/passwd", "&& whoami", "$HOME"],
        ["/usr/bin/env", "python3", "-c", "print('hello \"world\"')", "arg\nwith\nnewlines", ""],
    ],
)
async def test_modal_custom_container_executes_run_args_without_shell_interpretation(
    adversarial_args: list[str],
):
    """ModalProvider executes runtime.run_args directly with shell=False and ignores run_command."""
    profile = _container_profile(args=adversarial_args, port=8080)
    request, runtime = _render_container(
        profile,
        custom_args={"extra_cli_args": ["--extra", "literal;value"]},
    )
    expected_argv = [*adversarial_args, "--extra", "literal;value"]
    assert runtime.run_args == expected_argv

    # Even if run_command is mutated to contain shell injection, ModalProvider ignores it.
    tampered_runtime = runtime.model_copy(
        update={"run_command": "echo pwned && touch /tmp/injected"}
    )
    provider = ModalProvider()
    record = modal_record(provider, profile, AMBIENT)
    captured_serve = None
    web_server_kwargs: dict = {}

    def fake_web_server(**kwargs):
        web_server_kwargs.update(kwargs)

        def decorator(fn):
            nonlocal captured_serve
            captured_serve = fn
            return fn

        return decorator

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://adv.modal.run"),
        patch("modal.web_server", side_effect=fake_web_server),
        patch("subprocess.Popen") as mock_popen,
    ):
        await provider.provision(record, request, profile, tampered_runtime, AMBIENT)
        assert captured_serve is not None
        assert web_server_kwargs["port"] == 8080
        captured_serve()

    mock_popen.assert_called_once_with(expected_argv, shell=False)
    _assert_literal_popen_argv(expected_argv)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_args",
    [
        [],
        [""],
        ["   "],
        ["   ", "--port", "8080"],
    ],
)
async def test_modal_custom_container_rejects_empty_or_blank_argv(bad_args: list[str]):
    """Empty argv or blank executable is explicitly rejected in preflight and provision."""
    from inferweave.core.exceptions import DeploymentError, InferWeaveError
    from inferweave.providers.modal_provider import ModalConfigurationError

    profile = _container_profile(args=bad_args, port=8080)
    request, runtime = _render_container(profile)
    dry_run_request, _ = _render_container(profile, dry_run=True)
    public_request, _ = _render_container(
        profile, custom_args={"requires_proxy_auth": False}
    )

    provider = ModalProvider()
    for req in (request, dry_run_request, public_request):
        with pytest.raises((ModalConfigurationError, DeploymentError, InferWeaveError, ValueError)) as exc_info:
            provider.preflight(req, profile, runtime, AMBIENT)
        assert isinstance(exc_info.value, ValueError)
        assert isinstance(exc_info.value, DeploymentError)
        assert "argv" in str(exc_info.value).lower() or "run_args" in str(exc_info.value).lower()

    record = modal_record(provider, profile, AMBIENT)
    record.creation_may_continue = True
    with (
        patch("modal.Image.from_registry") as from_registry,
        patch("modal.App.deploy") as mock_deploy,
        pytest.raises(ModalConfigurationError) as exc_info,
    ):
        await provider.provision(record, request, profile, runtime, AMBIENT)

    from_registry.assert_not_called()
    mock_deploy.assert_not_called()
    assert record.creation_may_continue is False


@pytest.mark.asyncio
async def test_modal_rejects_empty_run_args_even_when_run_command_is_non_empty(
    sample_profile, sample_runtime
):
    """ModalProvider never falls back to shlex.split(run_command) when run_args is empty."""
    provider = ModalProvider()
    emptied = sample_runtime.model_copy(
        update={"run_args": [], "run_command": "python -m vllm --port 8000"}
    )
    request = _request(sample_profile)
    record = modal_record(provider, sample_profile, AMBIENT)

    with pytest.raises(ValueError, match="run_args"):
        provider.preflight(request, sample_profile, emptied, AMBIENT)

    with (
        patch("modal.App.deploy") as mock_deploy,
        pytest.raises(ValueError, match="run_args"),
    ):
        await provider.provision(record, request, sample_profile, emptied, AMBIENT)
    mock_deploy.assert_not_called()


@pytest.mark.asyncio
async def test_modal_empty_argv_through_provisioning_service_does_not_penalize_account():
    """Empty container argv fails locally in ProvisioningService without blaming the Modal account."""
    profile = _container_profile(args=[], port=8080)
    request, runtime = _render_container(profile)
    manager = modal_pool("a", "b")
    repository = InMemoryDeploymentRepository()
    service = ProvisioningService(manager, repository)
    candidate = ProvisionCandidate(provider=ModalProvider(), request=request, runtime=runtime)

    with (
        patch("modal.Client.from_credentials") as from_credentials,
        patch("modal.App.deploy") as mock_deploy,
        pytest.raises(ValueError, match="argv"),
    ):
        await service.provision([candidate], profile)

    from_credentials.assert_not_called()
    mock_deploy.assert_not_called()
    assert await repository.list_all() == []
    assert all(
        h.state == "available" and h.in_flight == 0 and h.consecutive_failures == 0
        for h in manager.health()
    )


@pytest.mark.asyncio
async def test_modal_custom_container_allows_empty_args_when_extra_cli_args_provides_argv():
    """If ContainerSpec.args is empty but request extra_cli_args supplies the command, it succeeds."""
    profile = _container_profile(args=[], port=8080)
    request, runtime = _render_container(
        profile,
        custom_args={"extra_cli_args": ["python3", "-m", "http.server", "8080"]},
    )
    assert runtime.run_args == ["python3", "-m", "http.server", "8080"]

    provider = ModalProvider()
    provider.preflight(request, profile, runtime, AMBIENT)
    record = modal_record(provider, profile, AMBIENT, endpoint_url=None)
    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://cli-supplied.modal.run"),
    ):
        result = await provider.provision(record, request, profile, runtime, AMBIENT)
    assert result.state == DeploymentState.STARTING
    assert result.endpoint_url == "https://cli-supplied.modal.run"


@pytest.mark.asyncio
async def test_modal_custom_container_port_hardware_scaling_and_auth():
    """Custom containers use configured port, GPU hardware, scaling options, and account isolation."""
    import modal

    hw = HardwareRequirements(
        min_vram_gb=40.0,
        gpu_count=2,
        recommended_gpus=["A100-80GB", "H100"],
    )
    hc = HealthcheckConfig(
        port=9090,
        path="/v2/health/ready",
        method="HEAD",
        expected_status_codes=[200, 204],
        consecutive_successes=2,
    )
    profile = _container_profile(
        model_id="custom/heavy-worker",
        image="ghcr.io/org/heavy:2.0.0",
        args=["/app/server", "--port", "9090"],
        port=9090,
        hardware=hw,
        healthcheck=hc,
    )
    request, runtime = _render_container(
        profile,
        custom_args={
            "scaledown_window_seconds": 420,
            "timeout_seconds": 7200,
            "cpu": 8.0,
            "memory": 32768,
        },
    )

    manager = modal_pool("team-prod", environment="prod-us")
    account = manager.resolve("modal", "team-prod")
    provider = ModalProvider()
    provider.preflight(request, profile, runtime, account)
    record = modal_record(provider, profile, account, endpoint_url=None)

    captured_fn_kwargs: dict = {}
    captured_ws_kwargs: dict = {}
    orig_function = modal.App.function
    orig_web_server = modal.web_server

    def spy_function(app, *args, **kwargs):
        captured_fn_kwargs.update(kwargs)
        return orig_function(app, *args, **kwargs)

    def spy_web_server(*args, **kwargs):
        captured_ws_kwargs.update(kwargs)
        return orig_web_server(*args, **kwargs)

    sentinel_client = MagicMock(name="team-prod-client")
    with (
        patch("modal.Client.from_credentials", return_value=sentinel_client) as from_creds,
        patch("modal.App.function", spy_function),
        patch("modal.web_server", spy_web_server),
        patch("modal.App.deploy", return_value=None) as mock_deploy,
        patch("modal.Function.get_web_url", return_value="https://heavy.modal.run"),
    ):
        result = await provider.provision(record, request, profile, runtime, account)

    from_creds.assert_called_once_with(
        "SENTINEL-modal-team-prod-id", "SENTINEL-modal-team-prod-secret"
    )
    mock_deploy.assert_called_once_with(
        name=record.id, environment_name="prod-us", client=sentinel_client
    )
    assert captured_fn_kwargs["gpu"] == "A100-80GB:2"
    assert captured_fn_kwargs["scaledown_window"] == 420
    assert captured_fn_kwargs["timeout"] == 7200
    assert captured_fn_kwargs["cpu"] == 8.0
    assert captured_fn_kwargs["memory"] == 32768
    assert captured_ws_kwargs["port"] == 9090
    assert captured_ws_kwargs["requires_proxy_auth"] is True
    assert captured_ws_kwargs["startup_timeout"] == 300
    assert result.state == DeploymentState.STARTING
    assert result.endpoint_url == "https://heavy.modal.run"


@pytest.mark.asyncio
async def test_modal_custom_container_rejects_mismatched_or_invalid_ports_and_env():
    """ModalProvider rejects mutated RuntimeSpec with mismatched healthcheck port or sensitive env."""
    profile = _container_profile(port=8080)
    request, runtime = _render_container(profile)
    provider = ModalProvider()

    # 1. Mismatched runtime.port vs runtime.healthcheck.port
    mismatched_port = runtime.model_copy(update={"port": 9000})
    with pytest.raises(ValueError, match="port"):
        provider.preflight(request, profile, mismatched_port, AMBIENT)

    # 2. Invalid port numbers / boolean port
    for invalid_port in (0, -1, 65536, True, False):
        bad_port_spec = runtime.model_copy(
            update={"port": invalid_port, "healthcheck": None}
        )
        with pytest.raises(ValueError, match="port"):
            provider.preflight(request, profile.model_copy(update={"container": None}), bad_port_spec, AMBIENT)

    # 3. Sensitive environment variable injected into mutated container RuntimeSpec
    leaked_env = runtime.model_copy(update={"env_vars": {"MODAL_TOKEN_SECRET": "leaked"}})
    with pytest.raises(ValueError, match="forbidden"):
        provider.preflight(request, profile, leaked_env, AMBIENT)

    # 4. Invalid image reference on mutated RuntimeSpec
    for bad_img in ("", "   ", "image with space:latest"):
        bad_img_spec = runtime.model_copy(update={"docker_image": bad_img})
        with pytest.raises(ValueError, match="image"):
            provider.preflight(request, profile, bad_img_spec, AMBIENT)


@pytest.mark.asyncio
async def test_modal_custom_container_end_to_end_readiness_and_lifecycle(tmp_path: Path):
    """End-to-end deploy, readiness probing (custom path/port/headers + proxy auth), and stop."""
    import modal
    from support import make_weave, modal_pool, patched_modal

    from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
    from inferweave.domain.healthcheck import ProbeOutcome, ProbeResult

    probe_calls: list[dict] = []

    class RecordingProbe(MockHealthcheckProbeAdapter):
        async def probe(
            self,
            url: str,
            method: str = "GET",
            timeout_seconds: float = 5.0,
            headers=None,
            expected_status_codes=None,
        ):
            probe_calls.append(
                {
                    "url": url,
                    "method": method,
                    "headers": dict(headers or {}),
                    "expected_status_codes": list(expected_status_codes or []),
                }
            )
            return await super().probe(
                url, method, timeout_seconds, headers, expected_status_codes
            )

    probe = RecordingProbe(
        canned_results=[
            ProbeResult(is_healthy=False, outcome=ProbeOutcome.CONNECTION_REFUSED),
            ProbeResult(is_healthy=True, outcome=ProbeOutcome.SUCCESS, status_code=204),
            ProbeResult(is_healthy=True, outcome=ProbeOutcome.SUCCESS, status_code=204),
        ]
    )
    manager = modal_pool("acct-1", environment="staging")
    weave = make_weave(tmp_path / "custom.db", probe=probe, accounts=manager)

    hc = HealthcheckConfig(
        port=9091,
        path="/custom/readyz",
        initial_delay_seconds=0.0,
        timeout_seconds=1.0,
        probe_interval_seconds=0.01,
        expected_status_codes=[200, 204],
        consecutive_successes=2,
        method="HEAD",
        headers={"X-Custom-Check": "enabled"},
    )
    profile = ModelProfile.for_container(
        id="org/custom-ready-container",
        image="ghcr.io/org/ready-srv:1.0.0",
        args=["/bin/ready-srv", "--port", "9091"],
        port=9091,
        add_python="3.12",
        healthcheck=hc,
        min_vram_gb=16.0,
        recommended_gpus=["A10G"],
    )
    weave.register_model(profile)

    with (
        patched_modal() as mocks,
        patch("modal.Image.from_registry", wraps=modal.Image.from_registry) as from_registry,
        patch("modal.web_server", wraps=modal.web_server) as web_server,
    ):
        dep = await weave.deploy(
            model="org/custom-ready-container",
            provider="modal",
            wait_for_ready=True,
            destroy_after_idle_mins=45,
            scaledown_window_seconds=120,
        )
        assert dep.state == DeploymentState.HEALTHY
        assert dep.account == "acct-1"
        assert dep.autostop_mins == 45
        from_registry.assert_called_once_with(
            "ghcr.io/org/ready-srv:1.0.0",
            setup_dockerfile_commands=[],
            add_python="3.12",
        )
        assert web_server.call_args.kwargs["port"] == 9091
        assert web_server.call_args.kwargs["requires_proxy_auth"] is True

        # Readiness probes used the custom path, method, expected status codes, custom header,
        # and the owning Modal account's proxy auth headers.
        assert len(probe_calls) == 3
        for call in probe_calls:
            assert call["url"].endswith("/custom/readyz")
            assert call["method"] == "HEAD"
            assert call["expected_status_codes"] == [200, 204]
            assert call["headers"]["X-Custom-Check"] == "enabled"
            assert call["headers"]["Modal-Key"] == "SENTINEL-modal-acct-1-proxy-id"
            assert call["headers"]["Modal-Secret"] == "SENTINEL-modal-acct-1-proxy-secret"

        await dep.stop()
        mocks.stop_app.assert_awaited_once()
        assert dep.state == DeploymentState.STOPPED

    await weave.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template_name", "model_id"),
    [
        ("vllm", "meta-llama/Meta-Llama-3-8B-Instruct"),
        ("fish-speech-s2", "fish-s2-pro"),
        ("flux-diffusers", "black-forest-labs/FLUX.1-schnell"),
        ("wan-video", "wan-video/wan-2.1"),
    ],
)
async def test_modal_builtin_models_regression_behavior_preserved(
    template_name: str, model_id: str
):
    """Built-in models preserve existing Modal image setup without add_python injection."""
    import modal

    from inferweave.registry.base import ModelRegistry
    from inferweave.runtimes.templates import get_runtime_template

    profile = ModelRegistry().get(model_id)
    assert profile.default_runtime == template_name
    request = _request(profile)
    runtime = get_runtime_template(template_name).render(profile, request)
    assert runtime.add_python is None

    provider = ModalProvider()
    provider.preflight(request, profile, runtime, AMBIENT)

    with (
        patch("modal.Image.from_registry", wraps=modal.Image.from_registry) as from_registry,
        patch("modal.web_server", wraps=modal.web_server) as web_server,
    ):
        _, result = await _provision(provider, profile, runtime)

    expected_setup_cmds = runtime.metadata.get("modal_setup_dockerfile_commands", [])
    from_registry.assert_called_once_with(
        runtime.docker_image,
        setup_dockerfile_commands=expected_setup_cmds,
    )
    assert "add_python" not in from_registry.call_args.kwargs
    assert web_server.call_args.kwargs["port"] == runtime.port
    assert result.state == DeploymentState.STARTING
