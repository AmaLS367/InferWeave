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
    with pytest.raises(ProviderAuthError, match="team-a") as exc_info:
        ModalProvider().preflight(_request(sample_profile), sample_profile, sample_runtime, account)
    assert "requires_proxy_auth" in str(exc_info.value)
    assert "SENTINEL" not in str(exc_info.value)


def test_pooled_account_ignores_ambient_proxy_env_tokens(sample_profile, sample_runtime):
    """The conftest ambient MODAL_PROXY_TOKEN_* must not satisfy a pooled account."""
    assert os.environ["MODAL_PROXY_TOKEN_ID"]
    account = modal_pool("team-a", proxy=False).resolve("modal", "team-a")
    with pytest.raises(ProviderAuthError):
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
        pytest.raises(ProviderAuthError, match="only"),
    ):
        await service.provision([candidate], sample_profile)
    from_credentials.assert_not_called()
    mock_deploy.assert_not_called()


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
    import modal

    manager = modal_pool("a", "b", strategy="failover")
    service = ProvisioningService(manager, InMemoryDeploymentRepository())
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
        patch("modal.App.lookup", side_effect=modal.exception.NotFoundError("x")),
    ):
        record, _ = await service.provision([candidate], sample_profile)

    assert calls == ["SENTINEL-modal-a-id", "SENTINEL-modal-b-id"]
    assert record.account == "b"
    assert record.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_all_pooled_accounts_rejected_raises_no_account_available(
    sample_profile, sample_runtime
):
    import modal

    manager = modal_pool("a", "b", strategy="failover", max_attempts=2)
    service = ProvisioningService(manager, InMemoryDeploymentRepository())
    candidate = _candidate(ModalProvider(), sample_profile, sample_runtime)

    with (
        patch("modal.Client.from_credentials", return_value=MagicMock()),
        patch("modal.App.deploy", side_effect=_modal_error("AuthError", "bad")),
        patch("modal.App.lookup", side_effect=modal.exception.NotFoundError("x")),
        pytest.raises(NoAccountAvailableError),
    ):
        await service.provision([candidate], sample_profile)
