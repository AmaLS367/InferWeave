"""Unit tests for Lifecycle domain entities and evaluators."""

from datetime import UTC, datetime, timedelta

from inferweave.domain.lifecycle import (
    AutostopAction,
    AutostopPolicy,
    DeploymentLifecycleEvaluator,
    LifecycleState,
)


def test_evaluator_calculate_idle_seconds():
    now = datetime.now(UTC)
    last_act = now - timedelta(seconds=120)
    state = LifecycleState(
        deployment_id="test-dep-1",
        created_at=last_act,
        last_activity_at=last_act,
        policy=AutostopPolicy(idle_minutes=10),
    )

    idle = DeploymentLifecycleEvaluator.calculate_idle_seconds(state, now=now)
    assert abs(idle - 120.0) < 0.1


def test_evaluator_is_idle_true_when_threshold_exceeded():
    now = datetime.now(UTC)
    last_act = now - timedelta(minutes=15)
    state = LifecycleState(
        deployment_id="test-dep-2",
        created_at=last_act,
        last_activity_at=last_act,
        policy=AutostopPolicy(idle_minutes=10),
    )

    assert DeploymentLifecycleEvaluator.is_idle(state, now=now) is True
    assert DeploymentLifecycleEvaluator.should_autostop(state, now=now) is True


def test_evaluator_is_idle_false_when_active():
    now = datetime.now(UTC)
    last_act = now - timedelta(minutes=5)
    state = LifecycleState(
        deployment_id="test-dep-3",
        created_at=last_act,
        last_activity_at=last_act,
        policy=AutostopPolicy(idle_minutes=10),
    )

    assert DeploymentLifecycleEvaluator.is_idle(state, now=now) is False
    assert DeploymentLifecycleEvaluator.should_autostop(state, now=now) is False


def test_evaluator_disabled_policy():
    now = datetime.now(UTC)
    last_act = now - timedelta(minutes=60)
    state = LifecycleState(
        deployment_id="test-dep-4",
        created_at=last_act,
        last_activity_at=last_act,
        policy=AutostopPolicy(idle_minutes=10, enabled=False),
    )

    assert DeploymentLifecycleEvaluator.is_idle(state, now=now) is False
    assert DeploymentLifecycleEvaluator.should_autostop(state, now=now) is False


def test_evaluator_none_idle_minutes():
    now = datetime.now(UTC)
    last_act = now - timedelta(minutes=60)
    state = LifecycleState(
        deployment_id="test-dep-5",
        created_at=last_act,
        last_activity_at=last_act,
        policy=AutostopPolicy(idle_minutes=None),
    )

    assert DeploymentLifecycleEvaluator.is_idle(state, now=now) is False
    assert DeploymentLifecycleEvaluator.should_autostop(state, now=now) is False


def test_evaluator_already_stopped():
    now = datetime.now(UTC)
    last_act = now - timedelta(minutes=60)
    state = LifecycleState(
        deployment_id="test-dep-6",
        created_at=last_act,
        last_activity_at=last_act,
        policy=AutostopPolicy(idle_minutes=10),
        is_stopped=True,
    )

    assert DeploymentLifecycleEvaluator.is_idle(state, now=now) is True
    assert DeploymentLifecycleEvaluator.should_autostop(state, now=now) is False


def test_autostop_action_enum():
    assert AutostopAction.STOP.value == "stop"
    assert AutostopAction.DOWN.value == "down"


def test_evaluator_reconcile_state():
    from inferweave.models.enums import DeploymentState

    # 1. Stopped always returns STOPPED
    assert (
        DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=DeploymentState.HEALTHY,
            is_stopped=True,
        )
        == DeploymentState.STOPPED
    )

    # 2. Infra Failed returns FAILED
    assert (
        DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=DeploymentState.FAILED,
        )
        == DeploymentState.FAILED
    )

    # 3. Infra Provisioning returns PROVISIONING
    assert (
        DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=DeploymentState.PROVISIONING,
        )
        == DeploymentState.PROVISIONING
    )

    # 4. Infra UP/HEALTHY + probe healthy -> HEALTHY
    assert (
        DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=DeploymentState.HEALTHY,
            probe_is_healthy=True,
        )
        == DeploymentState.HEALTHY
    )

    # 5. Infra UP/HEALTHY + probe unhealthy -> UNHEALTHY
    assert (
        DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=DeploymentState.HEALTHY,
            probe_is_healthy=False,
            current_state=DeploymentState.HEALTHY,
        )
        == DeploymentState.UNHEALTHY
    )

    # 6. Infra UP/HEALTHY + probe unhealthy while still provisioning -> PROVISIONING
    assert (
        DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=DeploymentState.HEALTHY,
            probe_is_healthy=False,
            current_state=DeploymentState.PROVISIONING,
        )
        == DeploymentState.PROVISIONING
    )
