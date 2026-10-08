"""Validation and loading of the secret-free accounts configuration."""

import json
from pathlib import Path

import pytest
from fakes import modal_spec

from inferweave.accounts import (
    ACCOUNTS_FILE_ENV,
    AccountsConfig,
    AccountSpec,
    ProviderAccounts,
    lightning_account,
    modal_account,
    runpod_account,
    vast_account,
)
from inferweave.core.exceptions import AccountConfigurationError


def _modal(*specs: AccountSpec, **kwargs: object) -> AccountsConfig:
    return AccountsConfig({"modal": ProviderAccounts(accounts=list(specs), **kwargs)})  # type: ignore[arg-type]


# --- helpers ------------------------------------------------------------------------------------


def test_helpers_map_secret_fields_to_env_var_names() -> None:
    modal = modal_account(
        "m",
        token_id_env="TID",
        token_secret_env="TSEC",
        proxy_token_id_env="PID",
        proxy_token_secret_env="PSEC",
        environment="main",
        weight=2,
        priority=1,
        max_concurrency=3,
    )
    assert dict(modal.env) == {
        "token_id": "TID",
        "token_secret": "TSEC",
        "proxy_token_id": "PID",
        "proxy_token_secret": "PSEC",
    }
    assert modal.credweave_metadata() == {
        "environment": "main",
        "weight": 2,
        "priority": 1,
        "max_concurrency": 3,
    }
    lightning = lightning_account(
        "l", user_id_env="UID", api_key_env="KEY", teamspace="org/ts"
    )
    assert dict(lightning.env) == {"user_id": "UID", "api_key": "KEY"}
    assert lightning.metadata == {"teamspace": "org/ts"}
    assert dict(runpod_account("r", api_key_env="RK").env) == {"api_key": "RK"}
    assert dict(vast_account("v", api_key_env="VK").env) == {"api_key": "VK"}
    # Unset scheduling values are not sent to CredWeave.
    assert runpod_account("r", api_key_env="RK").credweave_metadata() == {}


def test_modal_proxy_tokens_must_be_paired_in_helper() -> None:
    with pytest.raises(AccountConfigurationError, match="together"):
        modal_account(
            "m", token_id_env="A", token_secret_env="B", proxy_token_id_env="C"
        )
    with pytest.raises(AccountConfigurationError, match="together"):
        modal_account(
            "m", token_id_env="A", token_secret_env="B", proxy_token_secret_env="C"
        )


def test_proxy_tokens_must_be_paired_in_raw_spec() -> None:
    spec = AccountSpec(
        "m", env={"token_id": "A", "token_secret": "B", "proxy_token_id": "C"}
    )
    with pytest.raises(
        AccountConfigurationError, match="proxy_token_id.*proxy_token_secret"
    ):
        _modal(spec)


def test_missing_required_secret_field() -> None:
    with pytest.raises(
        AccountConfigurationError, match="missing required secret.*token_secret"
    ):
        _modal(AccountSpec("m", env={"token_id": "A"}))
    with pytest.raises(
        AccountConfigurationError, match="missing required secret.*user_id"
    ):
        AccountsConfig(
            {
                "lightning": ProviderAccounts(
                    accounts=[AccountSpec("l", env={"api_key": "K"})]
                )
            }
        )


def test_unknown_secret_field() -> None:
    with pytest.raises(AccountConfigurationError, match="unsupported secret.*password"):
        _modal(
            AccountSpec(
                "m", env={"token_id": "A", "token_secret": "B", "password": "P"}
            )
        )


def test_unknown_metadata_rejected_and_typed() -> None:
    with pytest.raises(AccountConfigurationError, match="unsupported metadata.*region"):
        _modal(
            AccountSpec(
                "m",
                env={"token_id": "A", "token_secret": "B"},
                metadata={"region": "eu"},
            )
        )
    # Lightning's teamspace is not valid metadata for Modal.
    with pytest.raises(
        AccountConfigurationError, match="unsupported metadata.*teamspace"
    ):
        _modal(
            AccountSpec(
                "m",
                env={"token_id": "A", "token_secret": "B"},
                metadata={"teamspace": "o/t"},
            )
        )
    with pytest.raises(AccountConfigurationError, match="must be a string"):
        AccountsConfig(
            {
                "lightning": ProviderAccounts(
                    accounts=[
                        AccountSpec(
                            "l",
                            env={"user_id": "U", "api_key": "K"},
                            metadata={"teamspace": 7},
                        )
                    ]
                )
            }
        )
    # Scheduling metadata is always allowed.
    _modal(
        AccountSpec(
            "m",
            env={"token_id": "A", "token_secret": "B"},
            metadata={"tags": ("gpu",), "environment": "dev"},
            weight=1.5,
        )
    )


def test_duplicate_account_ids() -> None:
    with pytest.raises(
        AccountConfigurationError, match="'a' is defined more than once"
    ):
        _modal(modal_spec("a"), modal_spec("a"))


def test_blank_account_id_rejected() -> None:
    with pytest.raises(AccountConfigurationError, match="non-empty"):
        _modal(AccountSpec("  ", env={"token_id": "A", "token_secret": "B"}))


@pytest.mark.parametrize("value", ["SENTINEL-literal=value", "", "   "])
def test_env_value_must_be_a_variable_name(value: str) -> None:
    with pytest.raises(
        AccountConfigurationError, match="environment variable name"
    ) as info:
        _modal(AccountSpec("m", env={"token_id": value, "token_secret": "B"}))
    if value.strip():
        assert value not in str(info.value)


def test_unsupported_provider() -> None:
    with pytest.raises(
        AccountConfigurationError, match="not supported for provider 'aws'"
    ):
        AccountsConfig({"aws": ProviderAccounts(accounts=[modal_spec("a")])})
    with pytest.raises(AccountConfigurationError, match="not supported"):
        AccountsConfig.from_dict({"providers": {"gcp": {"accounts": []}}})


def test_provider_name_is_case_insensitive() -> None:
    config = AccountsConfig({"Modal": ProviderAccounts(accounts=[modal_spec("a")])})
    assert list(config.providers) == ["modal"]


def test_unknown_strategy_and_aliases() -> None:
    with pytest.raises(
        AccountConfigurationError, match="Unknown account selection strategy"
    ):
        _modal(modal_spec("a"), strategy="fastest")
    for alias in ("lru", "priority", "round-robin", "Least_Used", "random", "weighted"):
        _modal(modal_spec("a"), strategy=alias)


@pytest.mark.parametrize("value", [0, -1])
def test_max_attempts_must_be_positive(value: int) -> None:
    with pytest.raises(AccountConfigurationError, match="max_attempts"):
        AccountsConfig({}, max_attempts=value)
    with pytest.raises(AccountConfigurationError, match="max_attempts"):
        AccountsConfig.from_dict({"max_attempts": value})


def test_max_attempts_default_and_custom() -> None:
    assert AccountsConfig().max_attempts == 3
    assert AccountsConfig.from_dict({"max_attempts": 5}).max_attempts == 5


def test_empty_pool_and_wrong_pool_type() -> None:
    with pytest.raises(AccountConfigurationError, match="at least one account"):
        AccountsConfig({"modal": ProviderAccounts()})
    with pytest.raises(AccountConfigurationError, match="ProviderAccounts instance"):
        AccountsConfig({"modal": {"accounts": []}})  # type: ignore[dict-item]


def test_default_state_dir_and_empty_config() -> None:
    config = AccountsConfig()
    assert config.providers == {}
    assert config.state_dir == Path.home() / ".inferweave" / "accounts"


# --- from_dict ----------------------------------------------------------------------------------


def _document() -> dict:
    return {
        "max_attempts": 4,
        "providers": {
            "modal": {
                "strategy": "weighted",
                "cooldown_seconds": 5,
                "max_consecutive_failures": 7,
                "permission_cooldown_seconds": 11,
                "quota_cooldown_seconds": 13,
                "accounts": [
                    {
                        "id": "team-a",
                        "env": {"token_id": "A_ID", "token_secret": "A_SECRET"},
                        "weight": 3,
                        "environment": "main",
                    },
                    {
                        "id": "team-b",
                        "env": {"token_id": "B_ID", "token_secret": "B_SECRET"},
                        "priority": 1,
                        "metadata": {"environment": "dev"},
                    },
                ],
            },
            "lightning": {
                "accounts": [
                    {
                        "id": "l1",
                        "env": {"user_id": "L_UID", "api_key": "L_KEY"},
                        "teamspace": "org/ts",
                    }
                ]
            },
        },
    }


def test_from_dict_full_document() -> None:
    config = AccountsConfig.from_dict(_document())
    assert config.max_attempts == 4
    modal = config.providers["modal"]
    assert modal.strategy == "weighted"
    assert (modal.cooldown_seconds, modal.max_consecutive_failures) == (5.0, 7)
    assert (modal.permission_cooldown_seconds, modal.quota_cooldown_seconds) == (
        11.0,
        13.0,
    )
    a, b = modal.accounts
    assert a.id == "team-a" and a.weight == 3 and a.metadata == {"environment": "main"}
    assert b.priority == 1 and b.metadata == {"environment": "dev"}
    assert config.providers["lightning"].accounts[0].metadata == {"teamspace": "org/ts"}
    # Defaults for an unconfigured knob.
    assert config.providers["lightning"].strategy == "round_robin"
    assert config.providers["lightning"].cooldown_seconds == 60.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(extra=1), "unsupported field.*extra"),
        (
            lambda d: d["providers"]["modal"].update(retries=2),
            "providers.modal has unsupported",
        ),
        (
            lambda d: d["providers"]["modal"]["accounts"][0].update(token="x"),
            r"accounts\[0\] has unsupported field.*token",
        ),
        (
            lambda d: d["providers"]["modal"]["accounts"][0].update(teamspace="o/t"),
            r"accounts\[0\] has unsupported field.*teamspace",
        ),
        (
            lambda d: d["providers"]["modal"]["accounts"][0].update(env=["A_ID"]),
            "env must map",
        ),
        (
            lambda d: d["providers"]["modal"]["accounts"].append("bad"),
            "must be a mapping",
        ),
        (
            lambda d: d["providers"].update(modal=[]),
            "providers.modal must be a mapping",
        ),
        (lambda d: d.update(providers=["modal"]), "'providers' must be a mapping"),
        (
            lambda d: d["providers"]["modal"].update(strategy="nope"),
            "Unknown account selection",
        ),
    ],
)
def test_from_dict_rejects_invalid_structure(mutate, message: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(AccountConfigurationError, match=message):
        AccountsConfig.from_dict(document)


def test_from_dict_requires_mapping() -> None:
    with pytest.raises(AccountConfigurationError, match="must be a mapping"):
        AccountsConfig.from_dict(["providers"])  # type: ignore[arg-type]


# --- from_file / from_env -----------------------------------------------------------------------


def test_from_file_yaml_resolves_relative_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    path = config_dir / "accounts.yaml"
    path.write_text(
        "max_attempts: 2\n"
        "state_dir: state\n"
        "providers:\n"
        "  runpod:\n"
        "    strategy: failover\n"
        "    credentials_file: secrets/runpod.json\n"
        "  modal:\n"
        "    accounts:\n"
        "      - id: a\n"
        "        env: {token_id: A_ID, token_secret: A_SECRET}\n",
        encoding="utf-8",
    )
    config = AccountsConfig.from_file(path)
    assert config.max_attempts == 2
    assert config.state_dir == config_dir / "state"
    assert (
        Path(config.providers["runpod"].credentials_file)
        == config_dir / "secrets" / "runpod.json"
    )
    assert config.providers["runpod"].strategy == "failover"
    assert config.providers["modal"].accounts[0].id == "a"


def test_from_file_json_and_absolute_credentials_file(tmp_path: Path) -> None:
    absolute = tmp_path / "elsewhere" / "vast.json"
    path = tmp_path / "accounts.json"
    path.write_text(
        json.dumps({"providers": {"vast": {"credentials_file": str(absolute)}}}),
        encoding="utf-8",
    )
    config = AccountsConfig.from_file(path)
    assert Path(config.providers["vast"].credentials_file) == absolute


def test_from_file_yml_extension_and_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "accounts.yml"
    path.write_text("", encoding="utf-8")
    assert AccountsConfig.from_file(path).providers == {}


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("bad.yaml", "providers:\n  modal: [token: SENTINEL-yaml-secret-123\n  - {"),
        ("bad.json", '{"providers": {"modal": "SENTINEL-json-secret-456", }'),
    ],
)
def test_parse_errors_do_not_echo_file_content(
    tmp_path: Path, name: str, content: str
) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AccountConfigurationError, match="not valid YAML/JSON") as info:
        AccountsConfig.from_file(path)
    rendered = f"{info.value} {info.value!r}"
    assert "SENTINEL" not in rendered
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_missing_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(
        AccountConfigurationError, match="Cannot read accounts configuration"
    ):
        AccountsConfig.from_file(tmp_path / "missing.yaml")


def test_from_env_uses_accounts_file_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "accounts.yaml"
    path.write_text(
        "providers:\n  runpod:\n    accounts:\n      - id: r1\n        env: {api_key: RP_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ACCOUNTS_FILE_ENV, str(path))
    config = AccountsConfig.from_env()
    assert config.providers["runpod"].accounts[0].id == "r1"
    monkeypatch.delenv(ACCOUNTS_FILE_ENV)
    assert AccountsConfig.from_env().providers == {}


def test_repr_never_contains_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IW_MODAL_A_TOKEN_ID", "SENTINEL-repr-token")
    config = _modal(modal_spec("a"))
    assert repr(config) == "AccountsConfig(providers=['modal'], max_attempts=3)"
    assert "SENTINEL" not in repr(config)
    assert "SENTINEL" not in repr(config.providers["modal"])
