"""Integration tests for CLI config + parser flow."""

from pathlib import Path

import atom.cli as cli


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "experiment_name: from_file",
                "seed: 5",
                "training:",
                "  max_steps: 123",
                "  save_interval: 7",
                "logging:",
                "  level: WARNING",
            ]
        ),
        encoding="utf-8",
    )


def test_main_with_argv_train_honors_file_and_cli_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "train.yaml"
    _write_config(config_path)

    captured = {"config": None, "ran": False}

    class FakeOrchestrator:
        def __init__(self, config=None):
            captured["config"] = config
            self.brain = object()
            self.memory = object()
            self.scientist = object()

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(cli, "AtomOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(cli, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli, "log_experiment_start", lambda *_: None)

    rc = cli._main_with_argv(
        [
            "--config",
            str(config_path),
            "--experiment",
            "from_cli",
            "--seed",
            "42",
            "--debug",
            "train",
            "--max-steps",
            "11",
            "--checkpoint-interval",
            "9",
        ]
    )

    assert rc == 0
    assert captured["ran"] is True
    cfg = captured["config"]
    assert cfg is not None
    assert cfg.training.max_steps == 11
    assert cfg.training.save_interval == 9
    assert cfg.experiment_name == "from_cli"
    assert cfg.seed == 42
    assert cfg.logging.level == "DEBUG"


def test_train_entrypoint_accepts_global_flags_before_subcommand(monkeypatch, tmp_path):
    config_path = tmp_path / "entry.yaml"
    _write_config(config_path)

    captured = {"config": None}

    class FakeOrchestrator:
        def __init__(self, config=None):
            captured["config"] = config
            self.brain = object()
            self.memory = object()
            self.scientist = object()

        def run(self):
            return None

    monkeypatch.setattr(cli, "AtomOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(cli, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli, "log_experiment_start", lambda *_: None)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "atom-train",
            "--config",
            str(config_path),
            "--experiment",
            "entry_cli",
            "--max-steps",
            "5",
        ],
    )

    rc = cli.main_train_entry()

    assert rc == 0
    cfg = captured["config"]
    assert cfg is not None
    assert cfg.experiment_name == "entry_cli"
    assert cfg.training.max_steps == 5
