"""Lightning SDK worker: runs one operation for one account in its own process.

Self-contained on purpose (stdlib + ``lightning_sdk`` only) so it can run under an isolated
interpreter that has Lightning SDK installed but not InferWeave. Protocol: one JSON request on
stdin ``{"op", "payload", "secrets"}``; one JSON line on stdout tagged ``"iw_worker"``.

Credentials arrive on stdin and are placed in *this* process's environment only, because the
SDK authenticates once per process from ``LIGHTNING_USER_ID``/``LIGHTNING_API_KEY``. Error
messages never include SDK text (it can echo request bodies); only a numeric HTTP status and a
classification leave the worker.
"""

import contextlib
import io
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

_STATUS_IN_MESSAGE = re.compile(r"response:\s*(\d{3})")
_MISSING = "was not found"


class WorkerError(Exception):
    def __init__(
        self,
        kind: str,
        message: str,
        status: int | None = None,
        resource_may_exist: bool | None = None,
        retry_after: float | None = None,
        operation_may_continue: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.resource_may_exist = resource_may_exist
        self.retry_after = retry_after
        self.operation_may_continue = operation_may_continue


def _emit(message: dict[str, Any]) -> None:
    print(json.dumps({"iw_worker": 1, **message}), flush=True)


def _kind_for_status(status: int | None) -> str:
    return {
        401: "auth",
        403: "permission",
        402: "quota",
        429: "rate_limit",
        400: "invalid_request",
        404: "invalid_request",
        409: "invalid_request",
        422: "invalid_request",
    }.get(status or 0, "transient")


def classify(err: BaseException) -> WorkerError:
    """Maps SDK exceptions to (kind, status) without keeping their text."""
    if isinstance(err, WorkerError):
        return err
    status = getattr(err, "status", None)
    if status is None:
        status = getattr(getattr(err, "response", None), "status_code", None)
    if not isinstance(status, int) or status <= 0:
        status = None
    if status is None and isinstance(err, ConnectionError) and "Authentication failed" in str(err):
        status = 401  # the REST client turns 401 into a builtin ConnectionError
    if status is None:
        # Retried 4xx responses surface as plain Exception("...response: 4xx").
        match = _STATUS_IN_MESSAGE.search(str(err))
        if match:
            status = int(match.group(1))
    kind = _kind_for_status(status)
    text = str(err).lower()
    if status != 401:
        if "quota exceeded" in text or "insufficient balance" in text or "insufficient credit" in text:
            kind = "quota"
        elif "out of stock" in text or "no gpu capacity" in text:
            kind = "capacity"
    label = f"HTTP {status}" if status else type(err).__name__
    return WorkerError(
        kind,
        f"Lightning SDK error ({label}); details withheld to protect credentials.",
        status,
        retry_after=_retry_after(err),
    )


def _retry_after(err: BaseException) -> float | None:
    """Numeric ``Retry-After`` header of an API error, if the SDK kept the response headers."""
    headers = getattr(err, "headers", None)
    if headers is None:
        headers = getattr(getattr(err, "response", None), "headers", None)
    try:
        value = headers.get("Retry-After") if headers is not None else None
        try:
            seconds = float(value) if value is not None else None
        except ValueError:
            seconds = max(0.0, (parsedate_to_datetime(str(value)) - datetime.now(UTC)).total_seconds())
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if seconds is not None and 0 <= seconds < float("inf") else None


def _cli(args: list[str]) -> tuple[bool, str, str]:
    """Runs a Lightning CLI command in-process; returns (ok, stdout, error message)."""
    from click.exceptions import ClickException, Exit
    from lightning_sdk.cli.entrypoint import main_cli  # type: ignore

    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            main_cli.main(args=args, prog_name="lightning", standalone_mode=False)
    except ClickException as exc:
        return False, out.getvalue(), exc.format_message()
    except Exit as exc:
        return exc.exit_code == 0, out.getvalue(), ""
    return True, out.getvalue(), ""


def _json_or_fail(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        raise WorkerError("transient", f"Lightning returned an invalid {what} response.") from None


def op_whoami(_: dict[str, Any]) -> dict[str, Any]:
    ok, out, _msg = _cli(["auth", "whoami", "--json"])
    if not ok:
        raise WorkerError("transient", "Lightning identity check failed.", None, False)
    identity = _json_or_fail(out, "identity")
    return {"auth_type": identity.get("auth_type") if isinstance(identity, dict) else None}


def op_start(payload: dict[str, Any]) -> dict[str, Any]:
    import lightning_sdk as sdk  # type: ignore
    from lightning_sdk import deployment as config  # type: ignore

    remote = sdk.Deployment(payload["name"], teamspace=payload["teamspace"])
    if remote.is_started:
        raise WorkerError(
            "invalid_request",
            "Lightning resource name collision; the existing resource was preserved.",
            None,
            False,
        )
    try:
        remote.start(
            image=payload["image"],
            machine=getattr(sdk.Machine, payload["machine"]),
            ports=[payload["port"]],
            entrypoint="/bin/sh",
            command=payload["command"],
            env=payload["env"],
            include_credentials=False,
            spot=False,
            replicas=1,
            auth=config.ApiKeyAuth(),
            health_check=config.HttpHealthCheck(
                path=payload["healthcheck_path"], port=payload["port"]
            ),
            autoscale=config.AutoScaleConfig(
                min_replicas=payload["min_replicas"],
                max_replicas=payload["max_replicas"],
                metric="GPU",
                threshold=90,
                idle_threshold_seconds=str(payload["idle_threshold_seconds"]),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - classified without its text
        error = classify(exc)
        # Only an explicit HTTP rejection proves nothing was created.
        # start() includes discovery and several API calls. A 4xx from a later call
        # does not prove that the initial create was rejected.
        error.resource_may_exist = True
        error.operation_may_continue = True
        raise error from None
    resource_id = remote.id
    deadline = time.monotonic() + float(payload.get("discovery_timeout", 60))
    urls: list[str] = []
    while True:
        with contextlib.suppress(Exception):
            urls = list(remote.urls or [])
        if urls or time.monotonic() >= deadline:
            break
        time.sleep(1)
    return {"resource_id": resource_id, "urls": urls}


def op_inspect(payload: dict[str, Any]) -> Any:
    ok, out, msg = _cli(
        ["deployment", "inspect", payload["target"], "--teamspace", payload["teamspace"], "--json"]
    )
    if not ok:
        if _MISSING in msg:
            return None
        raise WorkerError("transient", "Lightning deployment inspect failed.")
    data = _json_or_fail(out, "deployment status")
    if not isinstance(data, dict):
        raise WorkerError("transient", "Lightning returned an invalid deployment status.")
    return data


def op_delete(payload: dict[str, Any]) -> dict[str, bool]:
    ok, _out, msg = _cli(
        ["deployment", "delete", payload["target"], "--teamspace", payload["teamspace"], "--yes"]
    )
    if not ok:
        if _MISSING in msg:
            return {"deleted": False}
        raise WorkerError("transient", "Lightning deployment delete failed.")
    return {"deleted": True}


def op_probe(_: dict[str, Any]) -> dict[str, str]:
    """Network-free check that this interpreter can run the worker (used by CI and doctors)."""
    import lightning_sdk  # type: ignore

    return {"sdk": str(getattr(lightning_sdk, "__version__", "unknown"))}


OPERATIONS = {
    "probe": op_probe,
    "whoami": op_whoami,
    "start": op_start,
    "inspect": op_inspect,
    "delete": op_delete,
}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        operation = OPERATIONS[request["op"]]
    except (ValueError, KeyError):
        _emit({"ok": False, "error": {"kind": "invalid_request", "message": "bad worker request"}})
        return 2
    os.environ.update({k: v for k, v in (request.get("secrets") or {}).items() if v})
    os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
    try:
        from lightning_sdk.cli.utils.auth import browser_authentication  # type: ignore
    except ImportError:
        _emit(
            {
                "ok": False,
                "error": {
                    "kind": "invalid_request",
                    "message": "lightning-sdk is not installed in the Lightning worker interpreter",
                    "resource_may_exist": False,
                },
            }
        )
        return 3
    try:
        # Never fall back to an interactive browser login inside a worker.
        with browser_authentication(False):
            result = operation(request.get("payload") or {})
    except Exception as exc:  # noqa: BLE001 - classified without its text
        error = classify(exc)
        _emit(
            {
                "ok": False,
                "error": {
                    "kind": error.kind,
                    "status": error.status,
                    "retry_after": error.retry_after,
                    "message": str(error),
                    "resource_may_exist": error.resource_may_exist,
                    "operation_may_continue": error.operation_may_continue,
                },
            }
        )
        return 1
    _emit({"ok": True, "result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
