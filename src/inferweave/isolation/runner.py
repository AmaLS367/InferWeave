"""Runs provider SDK operations in isolated worker processes.

Lightning's SDK authenticates once per process from environment variables, and SkyPilot keeps
credentials and cluster state under the user's home directory. Neither can serve two accounts
in one process, so each operation runs in a short-lived worker process:

* secrets travel on the worker's **stdin** (never on its command line);
* pooled accounts get a minimal environment with an account-private ``HOME`` instead of the
  parent's environment, so no other account's (or the user's default) credentials leak in;
* the parent's ``os.environ`` is never modified;
* the worker may run under a different Python interpreter, which isolates SDKs whose
  dependencies conflict (Lightning SDK vs SkyPilot) in separate virtual environments;
* worker stderr is never surfaced (it may contain SDK debug output); results and errors are
  structured JSON on stdout.
"""

import asyncio
import contextlib
import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inferweave.core.exceptions import ProviderOperationError
from inferweave.core.failures import FailureKind

logger = logging.getLogger(__name__)

_PASSTHROUGH_ENV = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
"""Nonsecret variables a pooled-account worker inherits (networking, locale, temp dirs)."""

WORKER_DIR = Path(__file__).resolve().parent


class WorkerCancelledError(asyncio.CancelledError):
    """Cancellation with an explicit indication of whether the remote create may continue."""

    def __init__(self, operation_may_continue: bool) -> None:
        super().__init__("Provider operation cancelled.")
        self.operation_may_continue = operation_may_continue


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "***")
        return value
    if isinstance(value, dict):
        return {_redact(k, secrets): _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    return value


def account_environment(
    home: Path, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Minimal environment for a pooled-account worker with an account-private home."""
    env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
    home.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        home.chmod(0o700)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)  # Windows: Path.home() reads USERPROFILE, not HOME
    if extra:
        env.update(extra)
    return env


def ambient_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The ambient account keeps the user's own environment (native SDK behavior)."""
    env = dict(os.environ)
    if extra:
        env.update(extra)
    return env


class WorkerRunner:
    """Launches ``python -I <worker script>`` and exchanges one JSON request/response.

    Args:
        script: Worker script (self-contained; needs only the provider SDK installed).
        python: Interpreter that has the provider SDK installed (an isolated venv works).
        python_flags: Interpreter flags; ``-I`` keeps the worker from importing anything from
            the current directory, user site-packages or ``PYTHON*`` variables.
    """

    def __init__(
        self,
        provider: str,
        script: Path,
        python: str | None = None,
        python_flags: tuple[str, ...] = ("-I",),
    ) -> None:
        self.provider = provider
        self.script = script
        self.python = python or sys.executable
        self.python_flags = python_flags

    def command(self) -> list[str]:
        return [self.python, *self.python_flags, str(self.script)]

    async def run(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        env: Mapping[str, str],
        secrets: Mapping[str, str] | None = None,
        timeout: float,
        deployment_id: str | None = None,
        account_id: str = "",
    ) -> Any:
        """Runs one operation and returns its JSON result.

        Raises :class:`ProviderOperationError` for classified worker failures. A timeout or a
        crashed worker is TRANSIENT with ``resource_may_exist=True``: the operation may have
        reached the provider, so the caller must reconcile before retrying elsewhere.
        """
        # Some Windows SDKs install SelectorEventLoopPolicy globally. That loop cannot
        # create subprocesses. Use a dedicated Proactor loop without changing global policy.
        if (
            sys.platform == "win32"
            and type(asyncio.get_running_loop()).__name__ == "_WindowsSelectorEventLoop"
        ):

            def windows_run() -> Any:
                with asyncio.Runner(loop_factory=asyncio.ProactorEventLoop) as runner:
                    return runner.run(
                        self.run(
                            operation,
                            payload,
                            env=env,
                            secrets=secrets,
                            timeout=timeout,
                            deployment_id=deployment_id,
                            account_id=account_id,
                        )
                    )

            task = asyncio.create_task(asyncio.to_thread(windows_run))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                may_continue = True
                try:
                    await task
                    may_continue = False
                except ProviderOperationError as err:
                    may_continue = err.operation_may_continue
                raise WorkerCancelledError(may_continue) from None
        request = json.dumps(
            {"op": operation, "payload": dict(payload), "secrets": dict(secrets or {})}
        ).encode("utf-8")
        env = {**env, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        where = f"{self.provider} {operation} (account '{account_id}')"
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError:
            raise ProviderOperationError(
                f"Could not start the {self.provider} worker interpreter '{self.python}'. "
                f"Check the configured Python for the {self.provider} provider.",
                FailureKind.INVALID_REQUEST,
                resource_may_exist=False,
                deployment_id=deployment_id,
                operation_may_continue=False,
            ) from None
        exchange = asyncio.ensure_future(process.communicate(request))
        try:
            # Cancellation must not abandon an in-flight create call: the worker is drained
            # (bounded by the timeout) so its outcome is known before cleanup decisions.
            stdout, _stderr = await asyncio.wait_for(asyncio.shield(exchange), timeout)
        except asyncio.CancelledError:
            may_continue = True
            try:
                stdout, _stderr = await asyncio.wait_for(
                    asyncio.shield(exchange), timeout
                )
                self._parse(stdout, process.returncode, where, deployment_id)
                may_continue = False
            except ProviderOperationError as err:
                may_continue = err.operation_may_continue
            except TimeoutError:
                pass
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await exchange
            raise WorkerCancelledError(may_continue) from None
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await exchange
            raise ProviderOperationError(
                f"{where} timed out after {timeout:.0f}s; its outcome is unknown.",
                FailureKind.TRANSIENT,
                resource_may_exist=True,
                deployment_id=deployment_id,
                operation_may_continue=True,
            ) from None
        return self._parse(
            stdout,
            process.returncode,
            where,
            deployment_id,
            tuple((secrets or {}).values()),
        )

    @staticmethod
    def _parse(
        stdout: bytes,
        returncode: int | None,
        where: str,
        deployment_id: str | None,
        secrets: tuple[str, ...] = (),
    ) -> Any:
        message: dict[str, Any] | None = None
        for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
            line = line.strip()
            if line.startswith("{") and '"iw_worker"' in line:
                with contextlib.suppress(ValueError):
                    message = json.loads(line)
                    break
        if message is None:
            raise ProviderOperationError(
                f"{where} ended without a result (exit code {returncode}); its outcome is unknown.",
                FailureKind.TRANSIENT,
                resource_may_exist=True,
                deployment_id=deployment_id,
                operation_may_continue=True,
            )
        message = _redact(message, secrets)
        if message.get("ok"):
            return message.get("result")
        error = message.get("error") or {}
        try:
            kind = FailureKind(error.get("kind", "transient"))
        except ValueError:
            kind = FailureKind.TRANSIENT
        status = error.get("status")
        retry_after = error.get("retry_after")
        detail = error.get("message") or kind.value
        raise ProviderOperationError(
            f"{where} failed: {detail}",
            kind,
            status_code=status if isinstance(status, int) else None,
            retry_after=float(retry_after)
            if isinstance(retry_after, int | float)
            else None,
            resource_may_exist=error.get("resource_may_exist"),
            deployment_id=deployment_id,
            operation_may_continue=bool(error.get("operation_may_continue", False)),
        )
