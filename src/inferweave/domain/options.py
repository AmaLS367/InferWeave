"""Domain models and value objects for structured runtime, provider, and deployment options."""

from typing import Any

from pydantic import BaseModel, Field

from inferweave.domain.lifecycle import AutostopAction, AutostopPolicy


class RuntimeOptions(BaseModel):
    """Configuration options for model inference runtime engines (e.g. vLLM, SGLang)."""

    engine_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured engine parameters (e.g. max_model_len, gpu_memory_utilization, dtype)",
    )
    extra_cli_args: list[str] = Field(
        default_factory=list,
        description="Arbitrary raw CLI arguments appended to runtime server launch command",
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables injected into runtime container",
    )

    def to_cli_args(self) -> list[str]:
        """Converts structured engine arguments into formatted CLI flags."""
        args: list[str] = []
        for key, val in self.engine_args.items():
            if val is None or val is False:
                continue

            # Format flag name (--snake-case to --kebab-case)
            flag_name = key if key.startswith("-") else f"--{key.replace('_', '-')}"

            if val is True:
                args.append(flag_name)
            else:
                args.extend([flag_name, str(val)])

        args.extend(self.extra_cli_args)
        return args


class ProviderOptions(BaseModel):
    """Configuration options for cloud infrastructure and compute backends."""

    allow_spot: bool = Field(
        default=True,
        description="Whether spot/preemptible GPU instances are permissible",
    )
    max_price_per_hour: float | None = Field(
        default=None,
        description="Maximum hourly billing cap in USD for compute instance",
    )
    preferred_regions: list[str] = Field(
        default_factory=list,
        description="List of target cloud regions in order of preference",
    )
    disk_size_gb: int | None = Field(
        default=None,
        ge=10,
        description="Attached storage disk capacity in gigabytes",
    )
    scaledown_window_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Idle container timeout before down-scaling to zero replicas (Modal)",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Maximum serverless execution/function lifecycle timeout",
    )
    autodown: bool = Field(
        default=False,
        description="If True, SkyPilot will terminate (down) cluster on idle instead of stopping",
    )
    extra_provider_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific pass-through parameters (e.g. cpu, memory, secrets)",
    )


class DeploymentOptions(BaseModel):
    """Aggregate domain configuration unifying runtime, provider, and lifecycle policies."""

    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)
    provider: ProviderOptions = Field(default_factory=ProviderOptions)
    autostop: AutostopPolicy = Field(default_factory=AutostopPolicy)
    cleanup_on_failure: bool = Field(
        default=False,
        description="Whether to automatically destroy compute resources if readiness probe times out",
    )

    @classmethod
    def from_custom_args(
        cls,
        custom_args: dict[str, Any] | None = None,
        autostop_mins: int | None = 30,
    ) -> "DeploymentOptions":
        """Constructs a typed DeploymentOptions instance from raw dictionaries and parameters.

        Routes known provider, runtime, and autostop parameters into their respective value objects,
        preserving backwards-compatibility with untyped user dictionaries.
        """
        raw = dict(custom_args or {})
        cleanup_on_failure = bool(raw.pop("cleanup_on_failure", False))

        # Extract nested dictionaries if provided
        runtime_data = dict(raw.pop("runtime_args", None) or {})
        engine_args = dict(raw.pop("engine_args", None) or {})
        provider_data = dict(raw.pop("provider_args", None) or {})
        extra_cli_args = list(
            raw.pop("extra_cli_args", None) or raw.pop("extra_args", None) or []
        )
        extra_env = dict(raw.pop("extra_env", None) or {})

        # 1. Process Provider parameters
        allow_spot = raw.pop("allow_spot", provider_data.pop("allow_spot", True))
        max_price = raw.pop(
            "max_price_per_hour", provider_data.pop("max_price_per_hour", None)
        )
        regions = raw.pop(
            "preferred_regions", provider_data.pop("preferred_regions", [])
        )
        disk_size = raw.pop(
            "disk_size_gb",
            raw.pop(
                "disk_size",
                provider_data.pop("disk_size_gb", provider_data.pop("disk_size", None)),
            ),
        )
        scaledown_window = raw.pop(
            "scaledown_window_seconds",
            raw.pop(
                "scaledown_window",
                provider_data.pop(
                    "scaledown_window_seconds",
                    provider_data.pop("scaledown_window", None),
                ),
            ),
        )
        timeout_seconds = raw.pop(
            "timeout_seconds",
            raw.pop(
                "timeout",
                provider_data.pop(
                    "timeout_seconds", provider_data.pop("timeout", None)
                ),
            ),
        )
        autodown = raw.pop("autodown", provider_data.pop("autodown", False))

        # 2. Process Autostop parameters
        mins = raw.pop("autostop_mins", autostop_mins)
        if mins == 0 or mins == "0":
            mins = None
            autostop_enabled = False
        else:
            autostop_enabled = raw.pop("autostop_enabled", True)

        action_val = raw.pop("autostop_action", "down" if autodown else "stop")
        action = (
            AutostopAction(action_val) if isinstance(action_val, str) else action_val
        )

        # 3. Process known runtime / engine keys
        known_engine_keys = {
            "max_model_len",
            "gpu_memory_utilization",
            "dtype",
            "quantization",
            "enforce_eager",
            "chat_template",
            "tensor_parallel_size",
            "trust_remote_code",
            "max_num_seqs",
            "kv_cache_dtype",
            "pipeline_parallel_size",
            "block_size",
            "swap_space",
            "cpu_offload_gb",
            "enable_chunked_prefill",
        }

        for key in list(raw.keys()):
            if key in known_engine_keys:
                engine_args[key] = raw.pop(key)

        # Merge remaining raw items: any leftover provider keys or engine keys
        for key, val in list(raw.items()):
            if key in {"cpu", "memory", "secrets", "volumes", "cloud", "zone"}:
                provider_data[key] = raw.pop(key)
            else:
                # Store in engine args for CLI flags by default
                engine_args[key] = raw.pop(key)

        runtime_options = RuntimeOptions(
            engine_args={**runtime_data.get("engine_args", {}), **engine_args},
            extra_cli_args=[*runtime_data.get("extra_cli_args", []), *extra_cli_args],
            extra_env={**runtime_data.get("extra_env", {}), **extra_env},
        )

        provider_options = ProviderOptions(
            allow_spot=allow_spot,
            max_price_per_hour=max_price,
            preferred_regions=regions,
            disk_size_gb=disk_size,
            scaledown_window_seconds=scaledown_window,
            timeout_seconds=timeout_seconds,
            autodown=autodown,
            extra_provider_args=provider_data,
        )

        autostop_policy = AutostopPolicy(
            idle_minutes=mins,
            action=action,
            enabled=autostop_enabled and (mins is not None and mins > 0),
        )

        return cls(
            runtime=runtime_options,
            provider=provider_options,
            autostop=autostop_policy,
            cleanup_on_failure=cleanup_on_failure,
        )
