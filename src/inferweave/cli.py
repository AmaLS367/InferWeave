"""Command Line Interface (CLI) for InferWeave.

Unified orchestration and management of AI inference deployments across cloud GPU providers.
"""

import asyncio
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferweave import AutostopAction, InferWeave, WorkloadType
from inferweave.core.exceptions import (
    DeploymentNotFoundError,
    HealthcheckTimeoutError,
    InferWeaveError,
    ProviderNotFoundError,
)

app = typer.Typer(
    name="inferweave",
    help="InferWeave: Unified AI inference deployment SDK across cloud GPU providers.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(
            "[bold cyan]InferWeave CLI[/bold cyan] version [bold green]0.1.0[/bold green]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show the InferWeave version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """InferWeave command-line interface."""


@app.command(name="deploy", help="Deploy a model to cloud GPU infrastructure.")
def deploy(
    model: Annotated[
        str,
        typer.Argument(
            help="Model identifier from registry (e.g. 'fish-s2-pro', 'meta-llama/Meta-Llama-3-8B-Instruct')"
        ),
    ],
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Target provider ('runpod', 'aws', 'modal', 'auto')",
        ),
    ] = "auto",
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Routing strategy when provider='auto' ('cheapest', 'free_first')",
        ),
    ] = "cheapest",
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            "-g",
            help="Explicit GPU override (e.g. 'A100', 'H100', 'L4')",
        ),
    ] = None,
    num_gpus: Annotated[
        int | None,
        typer.Option(
            "--num-gpus",
            "-n",
            help="Explicit GPU count override",
        ),
    ] = None,
    autostop: Annotated[
        int,
        typer.Option(
            "--autostop",
            "-a",
            help="Auto-terminate after idle minutes",
        ),
    ] = 30,
    env: Annotated[
        list[str] | None,
        typer.Option(
            "--env",
            "-e",
            help="Environment variable in KEY=VALUE format (can be specified multiple times)",
        ),
    ] = None,
    custom_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--custom-arg",
            "-c",
            help="Provider or runtime custom argument in key=value format",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate and plan deployment without launching live cloud resources",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait",
            help="Block until readiness healthcheck probe passes",
        ),
    ] = True,
    cleanup_on_failure: Annotated[
        bool,
        typer.Option(
            "--cleanup-on-failure",
            help="Automatically terminate cloud resources if readiness probe times out",
        ),
    ] = False,
) -> None:
    """Deploys a model to the requested compute provider."""
    # Parse env pairs
    env_dict: dict[str, str] = {}
    if env:
        for item in env:
            if "=" in item:
                k, v = item.split("=", 1)
                env_dict[k.strip()] = v.strip()
            else:
                console.print(
                    f"[yellow]Warning: Ignoring env parameter '{item}'. Expected format is KEY=VALUE.[/yellow]"
                )

    # Parse custom arguments
    custom_args_dict: dict[str, Any] = {}
    if custom_arg:
        for item in custom_arg:
            if "=" in item:
                k, v = item.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    custom_args_dict[k] = True
                elif v.lower() == "false":
                    custom_args_dict[k] = False
                else:
                    try:
                        custom_args_dict[k] = int(v)
                    except ValueError:
                        try:
                            custom_args_dict[k] = float(v)
                        except ValueError:
                            custom_args_dict[k] = v
            else:
                console.print(
                    f"[yellow]Warning: Ignoring custom arg '{item}'. Expected format is key=value.[/yellow]"
                )

    if cleanup_on_failure:
        custom_args_dict["cleanup_on_failure"] = True

    # Support --autostop 0 as disabling the timer
    effective_autostop: int | None = autostop
    if autostop is not None and autostop <= 0:
        effective_autostop = None

    weave = InferWeave()
    status_msg = f"[bold green]Deploying {model} via provider '{provider}'...[/bold green]"
    if dry_run:
        status_msg = f"[bold yellow][DRY-RUN] Simulating deployment of {model} on '{provider}'...[/bold yellow]"

    try:
        with console.status(status_msg, spinner="line"):
            deployment = asyncio.run(
                weave.deploy(
                    model=model,
                    provider=provider,
                    strategy=strategy,
                    gpu_type=gpu,
                    num_gpus=num_gpus,
                    env=env_dict,
                    autostop_mins=effective_autostop,
                    custom_args=custom_args_dict,
                    dry_run=dry_run,
                    wait_for_ready=wait,
                )
            )

        if dry_run:
            console.print(
                "\n[bold yellow][+] Dry-run completed successfully! (No cloud resources were allocated)[/bold yellow]\n"
            )
        else:
            console.print(
                "\n[bold green][+] Deployment launched successfully![/bold green]\n"
            )

        table = Table(
            title="Deployment Overview", show_header=False, border_style="cyan"
        )
        table.add_column("Key", style="bold")
        table.add_column("Value")
        table.add_row("Deployment ID", deployment.id)
        table.add_row("Model", deployment.model)
        table.add_row("Provider", deployment.provider)
        table.add_row("State", deployment.state.value)
        table.add_row("Endpoint URL", deployment.endpoint_url or "[dim]Pending[/dim]")
        table.add_row("Autostop Idle Mins", str(deployment.autostop_mins))
        console.print(table)

    except HealthcheckTimeoutError as err:
        console.print(f"\n[bold red]Readiness timeout:[/bold red] {err}")
        dep_id = getattr(err, "deployment_id", None) or "unknown"
        console.print(
            f"\n[bold yellow]Warning:[/bold yellow] Deployment [cyan]{dep_id}[/cyan] was launched in cloud infrastructure, "
            f"but failed readiness checks."
        )
        console.print(
            f"[yellow]To prevent unintended compute charges, you can terminate it with:[/yellow]\n"
            f"  [bold cyan]inferweave stop {dep_id}[/bold cyan]\n"
        )
        raise typer.Exit(code=1) from err
    except ValueError as err:
        console.print(f"\n[bold red]Validation error:[/bold red] {err}")
        raise typer.Exit(code=1) from err
    except InferWeaveError as err:
        console.print(f"\n[bold red]Deployment failed:[/bold red] {err}")
        raise typer.Exit(code=1) from err
    except Exception as err:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {err}")
        raise typer.Exit(code=1) from err


@app.command(name="models", help="List registered models available for deployment.")
def list_models(
    workload: Annotated[
        str | None,
        typer.Option(
            "--workload",
            "-w",
            help="Filter models by workload type: 'audio', 'llm', 'image', 'video'",
        ),
    ] = None,
) -> None:
    """Displays models available in the InferWeave registry."""
    weave = InferWeave()
    workload_type = None
    if workload:
        try:
            workload_type = WorkloadType(workload.lower())
        except ValueError:
            valid_types = ", ".join([w.value for w in WorkloadType])
            console.print(
                f"[bold red]Invalid workload type '{workload}'.[/bold red] Valid options are: {valid_types}"
            )
            raise typer.Exit(code=1)

    models = weave.registry.list_models(workload_type=workload_type)

    table = Table(title="InferWeave Model Registry", border_style="blue")
    table.add_column("Model ID", style="bold cyan")
    table.add_column("Name")
    table.add_column("Workload", style="magenta")
    table.add_column("Runtime", style="yellow")
    table.add_column("Min VRAM", justify="right")
    table.add_column("Recommended GPUs")

    for m in models:
        table.add_row(
            m.id,
            m.name,
            m.workload_type.value,
            m.default_runtime,
            f"{m.hardware.min_vram_gb} GB",
            ", ".join(m.hardware.recommended_gpus),
        )

    console.print(table)


@app.command(name="providers", help="List supported cloud GPU providers.")
def list_providers() -> None:
    """Displays available compute providers and underlying backends."""
    table = Table(title="InferWeave Supported Providers", border_style="green")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Engine", style="magenta")
    table.add_column("Description")

    weave = InferWeave()
    registered = weave.router.list_providers()
    provider_meta: dict[str, tuple[str, str]] = {
        "runpod": ("SkyPilot", "RunPod GPU Cloud instances"),
        "modal": ("Modal SDK", "Modal Serverless GPU Functions & Web Endpoints"),
        "aws": ("SkyPilot", "Amazon Web Services EC2 GPU instances"),
        "gcp": ("SkyPilot", "Google Cloud Platform Compute Engine GPUs"),
        "azure": ("SkyPilot", "Microsoft Azure GPU Virtual Machines"),
        "lambda": ("SkyPilot", "Lambda GPU Cloud instances"),
        "nebius": ("SkyPilot", "Nebius AI Cloud instances"),
        "vast": ("SkyPilot", "Vast.ai GPU Cloud instances"),
        "oci": ("SkyPilot", "Oracle Cloud Infrastructure GPU instances"),
        "kubernetes": ("SkyPilot", "Self-hosted or managed Kubernetes clusters"),
        "fluidstack": ("SkyPilot", "FluidStack GPU Cloud instances"),
    }
    for p in registered:
        engine, desc = provider_meta.get(
            p, ("SkyPilot", f"{p.title()} GPU Cloud instances")
        )
        table.add_row(p, engine, desc)

    table.add_row("auto", "Smart Router", "Automatic VRAM- and cost-aware provider selection")
    console.print(table)


@app.command(name="list", help="List all tracked inference deployments.")
def list_deployments() -> None:
    """Displays all deployments recorded in persistent storage."""
    weave = InferWeave()
    records = asyncio.run(weave.list_records())
    if not records:
        console.print("[dim]No active or recorded deployments found.[/dim]")
        return

    table = Table(title="InferWeave Deployments", border_style="cyan")
    table.add_column("Deployment ID", style="bold cyan", no_wrap=True)
    table.add_column("Model")
    table.add_column("Provider", style="magenta")
    table.add_column("State")
    table.add_column("Endpoint URL")
    table.add_column("Created At", style="dim")

    for rec in records:
        st_val = rec.state.value if hasattr(rec.state, "value") else str(rec.state)
        state_style = (
            "green"
            if st_val == "healthy"
            else ("yellow" if st_val in ("pending", "provisioning", "starting") else "red")
        )
        created_str = (
            rec.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if rec.created_at
            else "[dim]N/A[/dim]"
        )
        table.add_row(
            rec.id,
            rec.model,
            rec.provider,
            f"[{state_style}]{st_val}[/{state_style}]",
            rec.endpoint_url or "[dim]None[/dim]",
            created_str,
        )

    console.print(table)


@app.command(name="status", help="Check the live status of an active deployment.")
def get_status(
    deployment_id: Annotated[
        str,
        typer.Argument(help="Deployment ID (e.g. iw-modal-...)"),
    ],
) -> None:
    """Queries the operational status of a deployment."""
    weave = InferWeave()
    try:
        status = asyncio.run(weave.get_status(deployment_id))
        table = Table(
            title=f"Deployment Status: {deployment_id}",
            show_header=False,
            border_style="cyan",
        )
        table.add_column("Property", style="bold")
        table.add_column("Value")
        table.add_row("ID", status.id)
        table.add_row("Model", status.model)
        table.add_row("Provider", status.provider)
        table.add_row("State", status.state.value)
        table.add_row("Endpoint URL", status.endpoint_url or "[dim]None[/dim]")
        if status.error_message:
            table.add_row("Error", f"[bold red]{status.error_message}[/bold red]")
        console.print(table)
    except DeploymentNotFoundError:
        console.print(
            f"[bold red]Deployment '{deployment_id}' was not found.[/bold red] Run 'inferweave list' to see all tracked deployments."
        )
        raise typer.Exit(code=1)
    except Exception as err:
        console.print(
            f"[bold red]Failed to retrieve status for '{deployment_id}':[/bold red] {err}"
        )
        raise typer.Exit(code=1) from err


@app.command(name="stop", help="Stop or shut down an active deployment.")
def stop(
    deployment_id: Annotated[
        str,
        typer.Argument(help="Deployment ID to terminate"),
    ],
    action: Annotated[
        AutostopAction,
        typer.Option(
            "--action",
            help="Lifecycle action: 'stop' (pause/stop) or 'down' (terminate)",
            case_sensitive=False,
        ),
    ] = AutostopAction.STOP,
) -> None:
    """Terminates or pauses a deployment."""
    weave = InferWeave()
    try:
        with console.status(
            f"[bold yellow]Stopping deployment '{deployment_id}' (action={action.value})...[/bold yellow]",
            spinner="line",
        ):
            asyncio.run(weave.stop(deployment_id, action=action))
        console.print(
            f"[bold green][+][/bold green] Deployment '{deployment_id}' stopped successfully."
        )
    except DeploymentNotFoundError:
        console.print(
            f"[bold red]Cannot stop deployment '{deployment_id}':[/bold red] Deployment not found."
        )
        raise typer.Exit(code=1)
    except ProviderNotFoundError as err:
        console.print(
            f"[bold red]Provider error while stopping '{deployment_id}':[/bold red] {err}"
        )
        raise typer.Exit(code=1) from err
    except Exception as err:
        console.print(
            f"[bold red]Failed to stop deployment '{deployment_id}':[/bold red] {err}"
        )
        raise typer.Exit(code=1) from err
