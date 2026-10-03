# Lifecycle and cost

## Lightning scaling and full deletion

Lightning defaults to zero minimum replicas, one maximum and a 300-second replica
idle threshold. Scale-to-zero preserves an authenticated reusable endpoint.
A positive minimum can consume credits continuously. `destroy_after_idle_mins`
is independent: explicit stop/watchdog destruction delete the owned Deployment
and confirm its absence. The SDK's scale-down-only `stop()` is not used for full
termination. No Studio is created. Startup/downloads and replicas can consume
credits; review current [Lightning pricing](https://lightning.ai/pricing).

Lightning attempts cleanup on provisioning and SDK deploy/readiness failures;
cancellation drains creation first. Cleanup failure preserves the original error
and logs only the deployment ID for retry with `weave.stop(id)`.
`close()` releases local monitoring, not remote compute. Recovered idle activity
and local in-flight protection use the shared lifecycle service.
See [Lightning guide](../how-to/use-lightning.md) for timeouts and paid tests.

**Modal container scale-to-zero:** `scaledown_window_seconds` (default 1800)
controls warm idle time after input. The app/URL remain after GPU containers
disappear. New requests reload runtime/weights. Modal manages this remotely,
even when the Python process exits.

**InferWeave full idle stop:** `destroy_after_idle_mins` (omitted: 30) controls
a local watchdog that stops the Modal app and removes endpoint availability.
Choose a longer interval than the container window or `None`/`0` to retain the
app. Legacy `autostop_mins` aliases full stop, not scaling.

Explicit `deployment.stop()`/`weave.stop(id)` stops the deployment. Modal
STOP/DOWN both stop apps. SkyPilot STOP may retain disks/stopped resources;
DOWN terminates clusters. SkyPilot also receives native idle/autodown settings.
STOP does not universally remove all billable provider resources.

## Activity and restarts

Inference resets `last_activity_at` at start/end, including errors. Retries
remain inside one call. In-flight requests protect that SDK's watchdog even
when generation exceeds its idle window. Explicit stop requires request draining.

In-memory timestamps are exact. Inference writes are throttled to once per
30 seconds by default, so restart can restore activity lagging by that interval.
Use `LifecycleService(activity_persist_interval_seconds=0)` for every touch.
`close()` does not flush pending timestamps.

Successful `check_health()`, readiness completion and refresh/status probes
count as in-memory activity but do not persist that timestamp. Synchronous
`record_activity()` is the same. This existing behavior expresses local demand;
restart restores persisted inference/registration activity. Failed probes do
not reset timers. Aggressive polling can keep apps alive and wake/retain GPUs.
The watchdog evaluates local state rather than continuously probing endpoints.

Recovery restores persisted timers and arms a local watchdog. Old missing
activity gets a fresh timestamp on first attach, then is backfilled. Monitoring
only runs while the owning SDK/process lives and a handle is registered/attached.
`close()` cancels it without stopping compute; full destruction is not remotely
enforced during downtime. Listing records alone does not arm watchdogs.
SQLite sharing does not share in-flight counts: use one owner or disable idle
stop for shared deployments.

## Cost

Modal bills loading, processing inputs and warm time after input. Scale-to-zero
ends container compute charges; cold starts trade reload latency/cost against
warm idle cost. Storage/egress/plan charges may be separate. Consult
[official pricing](https://modal.com/pricing); InferWeave sets no spend cap.

Readiness failure may leave resources provisioned. Set
`custom_args={"cleanup_on_failure": True}` to stop on readiness timeout and use
`finally` for one-shot scripts. Closing a local SDK does not stop remote apps.
