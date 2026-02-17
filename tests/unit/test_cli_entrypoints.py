"""Unit tests for CLI entrypoint and command wiring."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import tomllib

import atom.cli as cli


def _base_args(**kwargs):
    defaults = {
        "config": None,
        "experiment": None,
        "seed": None,
        "verbose": False,
        "debug": False,
        "resume": None,
        "max_steps": None,
        "checkpoint_interval": None,
        "output_dir": None,
        "steps": None,
        "checkpoint": Path("checkpoint.pth"),
        "output_file": None,
        "episodes": 10,
        "render": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_main_train_entry_dispatches_train_command(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli,
        "_parse_entrypoint_args",
        lambda command: argparse.Namespace(command=command),
    )

    def fake_dispatch(args):
        captured["command"] = args.command
        return 0

    monkeypatch.setattr(cli, "_dispatch_command", fake_dispatch)

    rc = cli.main_train_entry()

    assert rc == 0
    assert captured["command"] == "train"


def test_main_teacher_entry_dispatches_teacher_command(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli,
        "_parse_entrypoint_args",
        lambda command: argparse.Namespace(command=command),
    )

    def fake_dispatch(args):
        captured["command"] = args.command
        return 0

    monkeypatch.setattr(cli, "_dispatch_command", fake_dispatch)

    rc = cli.main_teacher_entry()

    assert rc == 0
    assert captured["command"] == "teacher"


def test_main_dream_entry_dispatches_dream_command(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli,
        "_parse_entrypoint_args",
        lambda command: argparse.Namespace(command=command),
    )

    def fake_dispatch(args):
        captured["command"] = args.command
        return 0

    monkeypatch.setattr(cli, "_dispatch_command", fake_dispatch)

    rc = cli.main_dream_entry()

    assert rc == 0
    assert captured["command"] == "dream"


def test_main_eval_entry_dispatches_eval_command(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli,
        "_parse_entrypoint_args",
        lambda command: argparse.Namespace(command=command),
    )

    def fake_dispatch(args):
        captured["command"] = args.command
        return 0

    monkeypatch.setattr(cli, "_dispatch_command", fake_dispatch)

    rc = cli.main_eval_entry()

    assert rc == 0
    assert captured["command"] == "eval"


def test_parse_entrypoint_args_accepts_global_and_command_specific_options():
    args = cli._parse_entrypoint_args(
        "train",
        raw_argv=[
            "--max-steps",
            "5",
            "--config",
            "config/training.yaml",
            "--experiment",
            "exp-unit",
            "--seed",
            "7",
            "--debug",
        ],
    )

    assert args.command == "train"
    assert args.max_steps == 5
    assert args.config == Path("config/training.yaml")
    assert args.experiment == "exp-unit"
    assert args.seed == 7
    assert args.debug is True


def test_load_configuration_preserves_file_values_and_applies_overrides(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "\n".join(
            [
                "experiment_name: from_file",
                "seed: 5",
                "training:",
                "  max_steps: 123",
                "logging:",
                "  level: WARNING",
            ]
        ),
        encoding="utf-8",
    )

    args = _base_args(
        config=yaml_file,
        experiment="from_cli",
        seed=42,
        debug=True,
    )

    config = cli.load_configuration(args)

    assert config.training.max_steps == 123
    assert config.experiment_name == "from_cli"
    assert config.seed == 42
    assert config.logging.level == "DEBUG"


def test_main_train_passes_loaded_config_to_orchestrator(monkeypatch):
    class ConfigStub(SimpleNamespace):
        def dict(self):
            return {}

    config = ConfigStub(
        experiment_name="unit-test",
        training=SimpleNamespace(max_steps=100, save_interval=10, checkpoint_dir=Path("checkpoints")),
    )
    captured = {"config": None, "ran": False}

    class FakeOrchestrator:
        def __init__(self, config=None):
            captured["config"] = config
            self.brain = object()
            self.memory = object()
            self.scientist = object()

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(cli, "load_configuration", lambda _args: config)
    monkeypatch.setattr(cli, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli, "log_experiment_start", lambda *_: None)
    monkeypatch.setattr(cli, "AtomOrchestrator", FakeOrchestrator)

    args = _base_args(max_steps=777, checkpoint_interval=33, resume=None)

    rc = cli.main_train(args)

    assert rc == 0
    assert captured["config"] is config
    assert captured["ran"] is True
    assert config.training.max_steps == 777
    assert config.training.save_interval == 33


def test_main_teacher_applies_step_and_output_overrides(monkeypatch, tmp_path):
    fake_ground_truth = SimpleNamespace(
        num_steps=10,
        OUTPUT_DIR="logs/ground_truth",
        run_simulation=lambda: None,
    )

    config = SimpleNamespace(output_dir=Path("outputs"))

    called = {"ran": False}

    def fake_run_simulation():
        called["ran"] = True

    fake_ground_truth.run_simulation = fake_run_simulation

    monkeypatch.setattr(cli, "load_configuration", lambda _args: config)
    monkeypatch.setattr(cli, "setup_logging", lambda _config: None)

    import atom.sim as sim_pkg

    monkeypatch.setattr(sim_pkg, "ground_truth", fake_ground_truth, raising=False)

    out_dir = tmp_path / "teacher-output"
    args = _base_args(output_dir=out_dir, steps=321)

    rc = cli.main_teacher(args)

    assert rc == 0
    assert called["ran"] is True
    assert config.output_dir == out_dir
    assert fake_ground_truth.num_steps == 321
    assert fake_ground_truth.OUTPUT_DIR == str(out_dir)
    assert out_dir.exists()


def test_main_eval_loads_checkpoint_and_runs_inference_pass(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ckpt.pth"
    checkpoint_path.write_text("stub", encoding="utf-8")

    class ConfigStub(SimpleNamespace):
        def get_device(self):
            return "cpu"

    config = ConfigStub(
        training=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
        experiment_name="eval-unit",
        brain=SimpleNamespace(action_dim=1, theory_dim=1),
    )

    class FakeWorld:
        def reset(self):
            return ([[[[[0.0]] * 1] * 1] * 4], None)

    class FakeBrain:
        theory_dim = 1

        def __call__(self, obs_t, theory_t, last_action, hx, theory_confidence=None):
            import torch

            mu = torch.zeros(1, 1)
            std = torch.ones(1, 1)
            value = torch.zeros(1, 1)
            return (mu, std), value, None, torch.zeros(1, 1)

    class FakeOrchestrator:
        def __init__(self, config=None):
            self.brain = FakeBrain()
            self.memory = object()
            self.scientist = object()
            self.world = FakeWorld()

        def _tensorify(self, obs):
            import torch

            return torch.zeros(1, 4, 1, 1, 1)

    monkeypatch.setattr(cli, "load_configuration", lambda _args: config)
    monkeypatch.setattr(cli, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli, "CheckpointManager", lambda *a, **k: object())
    monkeypatch.setattr(cli, "AtomOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(cli, "load_system_checkpoint", lambda *a, **k: None)

    args = _base_args(checkpoint=checkpoint_path, episodes=3, render=False)
    rc = cli.main_eval(args)

    assert rc == 0


def test_pyproject_console_scripts_target_noarg_wrappers():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["atom-train"] == "atom.cli:main_train_entry"
    assert scripts["atom-teacher"] == "atom.cli:main_teacher_entry"
    assert scripts["atom-dream"] == "atom.cli:main_dream_entry"
    assert scripts["atom-eval"] == "atom.cli:main_eval_entry"
