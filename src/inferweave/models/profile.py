"""Model hardware requirements, healthcheck parameters, and profile definitions."""

import re
from collections.abc import Iterable
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    FieldSerializationInfo,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails

from inferweave.core.credentials import (
    _REDACTED,
    contains_redacted,
    is_sensitive_header,
    is_sensitive_key,
    known_secret_values,
    sanitize_args,
    sanitize_metadata,
    sanitize_text,
)
from inferweave.models.enums import WorkloadType


class HardwareRequirements(BaseModel):
    """Hardware requirements and hints for GPU scheduling."""

    min_vram_gb: float = Field(..., description="Minimum GPU VRAM in gigabytes")
    recommended_gpus: list[str] = Field(
        default_factory=lambda: ["A10G", "L4", "A100", "H100"],
        description="List of recommended GPU types in order of preference",
    )
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs required")
    min_cpu_cores: int = Field(default=4, ge=1, description="Minimum CPU cores")
    min_ram_gb: float = Field(
        default=16.0, ge=1.0, description="Minimum system RAM in gigabytes"
    )
    min_cuda_version: str | None = Field(
        default="12.1", description="Minimum CUDA version"
    )

    def to_sanitized(self, secrets: Iterable[str] = ()) -> "HardwareRequirements":
        """Returns a copy of the hardware requirements with sensitive strings redacted."""
        secret_tuple = known_secret_values(secrets)
        copy_hw = self.model_copy(deep=True)
        copy_hw.recommended_gpus = [
            sanitize_text(gpu, secret_tuple) for gpu in copy_hw.recommended_gpus
        ]
        if copy_hw.min_cuda_version is not None:
            copy_hw.min_cuda_version = sanitize_text(
                copy_hw.min_cuda_version, secret_tuple
            )
        return copy_hw

    def to_sanitized_dict(self, secrets: Iterable[str] = ()) -> dict[str, Any]:
        """Returns a dict representation with sensitive strings redacted."""
        return self.to_sanitized(secrets).model_dump()

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        sanitized = self.to_sanitized()
        return [
            ("min_vram_gb", sanitized.min_vram_gb),
            ("recommended_gpus", sanitized.recommended_gpus),
            ("gpu_count", sanitized.gpu_count),
            ("min_cpu_cores", sanitized.min_cpu_cores),
            ("min_ram_gb", sanitized.min_ram_gb),
            ("min_cuda_version", sanitized.min_cuda_version),
        ]


def _is_sensitive_header(key: str) -> bool:
    return is_sensitive_header(key)


def _redact_headers(
    headers: dict[str, str], secrets: Iterable[str] = ()
) -> dict[str, str]:
    secret_tuple = known_secret_values(secrets)
    return {
        k: (_REDACTED if _is_sensitive_header(k) else sanitize_text(v, secret_tuple))
        for k, v in headers.items()
    }


class HealthcheckConfig(BaseModel):
    """Parameters for monitoring endpoint readiness."""

    enabled: bool = Field(
        default=True, description="Whether readiness healthcheck polling is enabled"
    )
    path: str = Field(
        default="/health", description="HTTP endpoint path for readiness probe"
    )
    port: int = Field(default=8000, description="Exposed service port")
    initial_delay_seconds: float = Field(
        default=10.0, ge=0.0, description="Delay before first probe check in seconds"
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=0.01,
        description="Total timeout waiting for model readiness in seconds",
    )
    probe_interval_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Interval between subsequent health checks in seconds",
    )
    request_timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        description="Timeout for an individual HTTP probe request in seconds",
    )
    expected_status_codes: list[int] = Field(
        default_factory=lambda: [200],
        description="HTTP status codes considered healthy",
    )
    consecutive_successes: int = Field(
        default=1,
        ge=1,
        description="Consecutive successful probes required for readiness",
    )
    method: str = Field(
        default="GET",
        description="HTTP method for probe, e.g. 'GET' or 'HEAD'",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers sent with probe requests",
    )

    @field_serializer("headers")
    def _serialize_headers(
        self, headers: dict[str, str], info: FieldSerializationInfo
    ) -> dict[str, str]:
        if info.context and info.context.get("redact_sensitive", False):
            return _redact_headers(headers)
        return headers

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        args: list[tuple[str | None, Any]] = []
        for k, v in super().__repr_args__():
            if k == "headers" and isinstance(v, dict):
                args.append((k, _redact_headers(v)))
            elif k == "path" and isinstance(v, str):
                args.append((k, sanitize_text(v)))
            else:
                args.append((k, v))
        return args

    def to_sanitized(self, secrets: Iterable[str] = ()) -> "HealthcheckConfig":
        """Returns a copy of the healthcheck config with sensitive headers redacted."""
        secret_tuple = known_secret_values(secrets)
        copy_cfg = self.model_copy(deep=True)
        copy_cfg.path = sanitize_text(copy_cfg.path, secret_tuple)
        copy_cfg.headers = _redact_headers(copy_cfg.headers, secret_tuple)
        return copy_cfg

    def to_sanitized_dict(self, secrets: Iterable[str] = ()) -> dict[str, Any]:
        """Returns a dict representation with sensitive headers redacted."""
        return self.to_sanitized(secrets).model_dump()


class ContainerSpec(BaseModel):
    """Specification for running a model or service in a custom container."""

    image: str = Field(
        ...,
        description="Non-empty container image reference (tag or SHA256 digest)",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Argv token list for container entrypoint, never a shell command",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Exposed container port (integer from 1 to 65535)",
    )
    setup_commands: list[str] = Field(
        default_factory=list,
        description="Commands executed before container process execution",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of nonsecret environment variables",
    )
    add_python: str | None = Field(
        default=None,
        description="Optional Python version string to install, e.g. '3.12'",
    )

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("image must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("Container image reference cannot be empty")
        if any(c.isspace() for c in stripped):
            raise ValueError(
                f"Container image reference cannot contain whitespace: {value!r}"
            )
        return stripped

    @field_validator("args", mode="before")
    @classmethod
    def _validate_args(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raise ValueError(  # noqa: TRY004
                "args must be a list of argv strings, never a shell command string"
            )
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"args must be a list of strings, got {type(value).__name__}"
            )
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"Each arg in args must be a string, got {type(item).__name__}"
                )
        return [str(item) for item in value]

    @field_validator("port", mode="before")
    @classmethod
    def _validate_port(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"port must be an integer, got {type(value).__name__}: {value!r}"
            )
        if value < 1 or value > 65535:
            raise ValueError(f"port must be an integer from 1 to 65535, got {value}")
        return value

    @field_validator("setup_commands", mode="before")
    @classmethod
    def _validate_setup_commands(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raise ValueError(  # noqa: TRY004
                "setup_commands must be a list of strings, not a single string"
            )
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"setup_commands must be a list of strings, got {type(value).__name__}"
            )
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"Each setup command must be a string, got {type(item).__name__}"
                )
        return [str(item) for item in value]

    @field_validator("env")
    @classmethod
    def _validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        if not isinstance(value, dict):
            raise TypeError(f"env must be a dictionary, got {type(value).__name__}")
        for key, val in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Environment variable name must be a string, got {type(key).__name__}"
                )
            if not isinstance(val, str):
                raise TypeError(
                    f"Environment variable value for '{key}' must be a string, got {type(val).__name__}"
                )
            if is_sensitive_key(key):
                raise ValueError(
                    f"Platform credentials and sensitive keys are forbidden in ContainerSpec.env: '{key}'"
                )
        return value

    @field_validator("add_python")
    @classmethod
    def _validate_add_python(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"add_python must be a string or None, got {type(value).__name__}"
            )
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("add_python cannot be empty")
        if not re.fullmatch(r"^\d+(\.\d+)*$", cleaned):
            raise ValueError(
                f"Invalid Python version string '{value}'; expected format like '3.12'"
            )
        return cleaned

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        sanitized = self.to_sanitized()
        sanitized_map: dict[str | None, Any] = {
            "image": sanitized.image,
            "args": sanitized.args,
            "port": self.port,
            "setup_commands": sanitized.setup_commands,
            "env": sanitized.env,
            "add_python": sanitized.add_python,
        }
        return [
            (k, sanitized_map.get(k, v))
            for k, v in super().__repr_args__()
        ]

    def to_sanitized(self, secrets: Iterable[str] = ()) -> "ContainerSpec":
        """Returns a copy of the container spec with credentials and tokens redacted."""
        secret_tuple = known_secret_values(secrets)
        copy_spec = self.model_copy(deep=True)
        copy_spec.image = sanitize_text(copy_spec.image, secret_tuple)
        copy_spec.args = sanitize_args(copy_spec.args, secret_tuple)
        copy_spec.setup_commands = [
            sanitize_text(cmd, secret_tuple) for cmd in copy_spec.setup_commands
        ]
        copy_spec.env = {
            k: (_REDACTED if is_sensitive_key(k) else sanitize_text(v, secret_tuple))
            for k, v in copy_spec.env.items()
        }
        if copy_spec.add_python is not None:
            copy_spec.add_python = sanitize_text(copy_spec.add_python, secret_tuple)
        return copy_spec

    def to_sanitized_dict(self, secrets: Iterable[str] = ()) -> dict[str, Any]:
        """Returns a dict representation with credentials and tokens redacted."""
        return self.to_sanitized(secrets).model_dump()


class ModelProfile(BaseModel):
    """Metadata and resource profile for a deployable AI model."""

    id: str = Field(..., description="Unique model identifier, e.g. 'fish-s2-pro'")
    name: str = Field(..., description="Human-readable model name")
    workload_type: WorkloadType = Field(
        ..., description="Category of the model workload"
    )
    default_runtime: str = Field(
        ..., description="Identifier of the default runtime template, e.g. 'vllm'"
    )
    source: str = Field(
        default="huggingface",
        description="Source model hub or repository (e.g. 'huggingface', 'local')",
    )
    artifact_id: str | None = Field(
        default=None,
        description="External repository or hub artifact identifier (e.g. 'fishaudio/s2-pro')",
    )
    hardware: HardwareRequirements = Field(..., description="Hardware prerequisites")
    healthcheck: HealthcheckConfig = Field(default_factory=HealthcheckConfig)
    default_env: dict[str, str] = Field(
        default_factory=dict, description="Default environment variables"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional arbitrary metadata"
    )
    container: ContainerSpec | None = Field(
        default=None,
        description="Custom container configuration for container-backed workloads",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("id must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("ModelProfile id cannot be empty")
        if contains_redacted(stripped):
            raise ValueError("ModelProfile id cannot contain redaction markers")
        if sanitize_text(stripped) != stripped:
            raise ValueError(
                "ModelProfile id contains sensitive credentials or patterns"
            )
        return stripped

    @model_validator(mode="wrap")
    @classmethod
    def _sanitize_validation_errors(cls, v: Any, handler: Any) -> Any:
        try:
            return handler(v)
        except ValidationError as exc:
            new_errors: list[InitErrorDetails] = []
            for err in exc.errors():
                loc = err.get("loc", ())
                raw_input = err.get("input")
                err_type = err.get("type", "value_error")
                msg = err.get("msg", "")

                if isinstance(raw_input, str):
                    if sanitize_text(raw_input) != raw_input:
                        sanitized_input: Any = _REDACTED
                    else:
                        sanitized_input = raw_input
                elif isinstance(raw_input, dict):
                    sanitized_dict = dict(raw_input)
                    if (
                        "id" in sanitized_dict
                        and isinstance(sanitized_dict["id"], str)
                        and (
                            sanitize_text(sanitized_dict["id"]) != sanitized_dict["id"]
                            or contains_redacted(sanitized_dict["id"])
                        )
                    ):
                        sanitized_dict["id"] = _REDACTED
                    sanitized_input = sanitize_metadata(sanitized_dict)
                elif isinstance(raw_input, (list, tuple)):
                    sanitized_input = [
                        _REDACTED
                        if isinstance(item, str) and sanitize_text(item) != item
                        else (sanitize_metadata(item) if isinstance(item, dict) else item)
                        for item in raw_input
                    ]
                else:
                    sanitized_input = raw_input

                if loc == ("id",) and "contains sensitive credentials or patterns" in msg:
                    sanitized_input = _REDACTED

                sanitized_msg = sanitize_text(msg)

                ctx = dict(err.get("ctx") or {})
                if "error" in ctx:
                    inner_err = ctx["error"]
                    if isinstance(inner_err, Exception):
                        clean_err_msg = sanitize_text(str(inner_err))
                        ctx["error"] = type(inner_err)(clean_err_msg)
                    elif isinstance(inner_err, str):
                        ctx["error"] = ValueError(sanitize_text(inner_err))
                elif err_type in ("value_error", "assertion_error"):
                    clean_msg = sanitized_msg.removeprefix("Value error, ")
                    ctx["error"] = ValueError(clean_msg)

                for k, val in list(ctx.items()):
                    if k != "error" and isinstance(val, str):
                        ctx[k] = sanitize_text(val)

                new_errors.append(
                    InitErrorDetails(
                        type=err_type,
                        loc=loc,
                        input=sanitized_input,
                        ctx=ctx,
                    )
                )
            raise ValidationError.from_exception_data(cls.__name__, new_errors) from None

    @model_validator(mode="after")
    def _validate_container_profile(self) -> "ModelProfile":
        is_container_profile = (
            self.container is not None or self.default_runtime == "container"
        )
        if is_container_profile and self.default_env:
            for key in self.default_env:
                if is_sensitive_key(key):
                    raise ValueError(
                        f"Platform credentials and sensitive keys are forbidden in profile default_env: '{key}'"
                    )

        if (
            self.container is not None
            and self.healthcheck is not None
            and self.container.port != self.healthcheck.port
        ):
            raise ValueError(
                f"Inconsistent configuration: container.port ({self.container.port}) "
                f"and healthcheck.port ({self.healthcheck.port}) must match."
            )
        return self

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        secret_tuple = known_secret_values(())
        safe_id = sanitize_text(self.id, secret_tuple)
        safe_name = sanitize_text(self.name, secret_tuple)
        safe_source = sanitize_text(self.source, secret_tuple)
        safe_runtime = sanitize_text(self.default_runtime, secret_tuple)
        safe_artifact_id = (
            sanitize_text(self.artifact_id, secret_tuple)
            if self.artifact_id is not None
            else None
        )
        safe_hardware = (
            self.hardware.to_sanitized(secret_tuple)
            if self.hardware is not None
            else None
        )
        safe_healthcheck = (
            self.healthcheck.to_sanitized(secret_tuple)
            if self.healthcheck is not None
            else None
        )
        safe_default_env = {
            k: (_REDACTED if is_sensitive_key(k) else sanitize_text(v, secret_tuple))
            for k, v in self.default_env.items()
        }
        safe_metadata = sanitize_metadata(self.metadata, secret_tuple)
        safe_container = (
            self.container.to_sanitized(secret_tuple)
            if self.container is not None
            else None
        )
        sanitized_map: dict[str | None, Any] = {
            "id": safe_id,
            "name": safe_name,
            "workload_type": self.workload_type,
            "default_runtime": safe_runtime,
            "source": safe_source,
            "artifact_id": safe_artifact_id,
            "hardware": safe_hardware,
            "healthcheck": safe_healthcheck,
            "default_env": safe_default_env,
            "metadata": safe_metadata,
            "container": safe_container,
        }
        return [
            (k, sanitized_map.get(k, v))
            for k, v in super().__repr_args__()
        ]

    def to_sanitized(self, secrets: Iterable[str] = ()) -> "ModelProfile":
        """Returns a deep copy of the profile with all sensitive fields redacted."""
        secret_tuple = known_secret_values(secrets)
        if sanitize_text(self.id, secret_tuple) != self.id:
            raise ValueError(
                "ModelProfile id contains sensitive credentials and cannot "
                "be safely sanitized without breaking required identity."
            )
        copy_prof = self.model_copy(deep=True)
        copy_prof.name = sanitize_text(copy_prof.name, secret_tuple)
        copy_prof.source = sanitize_text(copy_prof.source, secret_tuple)
        if copy_prof.default_runtime:
            copy_prof.default_runtime = sanitize_text(
                copy_prof.default_runtime, secret_tuple
            )
        if copy_prof.artifact_id is not None:
            copy_prof.artifact_id = sanitize_text(copy_prof.artifact_id, secret_tuple)
        if copy_prof.hardware is not None:
            copy_prof.hardware = copy_prof.hardware.to_sanitized(secret_tuple)
        if copy_prof.healthcheck is not None:
            copy_prof.healthcheck = copy_prof.healthcheck.to_sanitized(secret_tuple)
        if copy_prof.default_env:
            copy_prof.default_env = {
                k: (_REDACTED if is_sensitive_key(k) else sanitize_text(v, secret_tuple))
                for k, v in copy_prof.default_env.items()
            }
        if copy_prof.metadata:
            copy_prof.metadata = sanitize_metadata(copy_prof.metadata, secret_tuple)
        if copy_prof.container is not None:
            copy_prof.container = copy_prof.container.to_sanitized(secret_tuple)
        return copy_prof

    def to_sanitized_dict(self, secrets: Iterable[str] = ()) -> dict[str, Any]:
        """Returns a dict representation of the profile with all sensitive fields redacted."""
        return self.to_sanitized(secrets).model_dump()

    @classmethod
    def for_container(
        cls,
        id: str,
        image: str,
        args: list[str] | None = None,
        port: int | None = None,
        health_path: str = "/health",
        min_vram_gb: float = 0.0,
        *,
        name: str | None = None,
        setup_commands: list[str] | None = None,
        env: dict[str, str] | None = None,
        add_python: str | None = None,
        gpu_count: int = 1,
        recommended_gpus: list[str] | None = None,
        min_cpu_cores: int = 4,
        min_ram_gb: float = 16.0,
        min_cuda_version: str | None = "12.1",
        hardware: HardwareRequirements | None = None,
        healthcheck: HealthcheckConfig | None = None,
        source: str = "container",
        artifact_id: str | None = None,
        default_env: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ModelProfile":
        """Convenience factory to construct a ModelProfile for a containerized workload.

        Always produces a profile with ``workload_type=WorkloadType.CUSTOM`` and
        ``default_runtime="container"``. These settings cannot be overridden through kwargs.

        In container profiles, the container port and healthcheck port must be consistent.
        If an explicit healthcheck is provided without a port argument, container.port inherits
        healthcheck.port. Providing conflicting port and healthcheck.port values raises a ValueError.
        """
        if "workload_type" in kwargs:
            val = kwargs.pop("workload_type")
            if val != WorkloadType.CUSTOM:
                raise ValueError(
                    f"Cannot override workload_type in for_container(); must be {WorkloadType.CUSTOM}, got {val!r}"
                )

        if "default_runtime" in kwargs:
            val = kwargs.pop("default_runtime")
            if val != "container":
                raise ValueError(
                    f"Cannot override default_runtime in for_container(); must be 'container', got {val!r}"
                )

        if port is not None and healthcheck is not None and port != healthcheck.port:
            raise ValueError(
                f"Inconsistent configuration: container port ({port}) and "
                f"healthcheck.port ({healthcheck.port}) must match."
            )

        resolved_port = port if port is not None else (healthcheck.port if healthcheck is not None else 8000)

        container_args = list(args) if args is not None else []
        container_spec = ContainerSpec(
            image=image,
            args=container_args,
            port=resolved_port,
            setup_commands=list(setup_commands) if setup_commands is not None else [],
            env=dict(env or {}),
            add_python=add_python,
        )

        if hardware is not None:
            hw = hardware
        else:
            hw_kwargs: dict[str, Any] = {
                "min_vram_gb": min_vram_gb,
                "gpu_count": gpu_count,
                "min_cpu_cores": min_cpu_cores,
                "min_ram_gb": min_ram_gb,
                "min_cuda_version": min_cuda_version,
            }
            if recommended_gpus is not None:
                hw_kwargs["recommended_gpus"] = recommended_gpus
            hw = HardwareRequirements(**hw_kwargs)

        if healthcheck is not None:
            hc = healthcheck
        else:
            hc = HealthcheckConfig(port=resolved_port, path=health_path)

        profile_env = dict(env or {})
        if default_env:
            profile_env.update(default_env)

        return cls(
            id=id,
            name=name or id,
            workload_type=WorkloadType.CUSTOM,
            default_runtime="container",
            source=source,
            artifact_id=artifact_id,
            hardware=hw,
            healthcheck=hc,
            default_env=profile_env,
            metadata=metadata or {},
            container=container_spec,
            **kwargs,
        )

    @property
    def target_artifact(self) -> str:
        """Returns the external repository/artifact ID if defined, else falls back to model ID."""
        return self.artifact_id or self.id
