"""Command-line interface for ATOM system.

Provides production-grade CLI with proper argument parsing, configuration loading,
and error handling.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from .config import AtomConfig
from .exceptions import AtomError, ConfigurationError
from .logging import setup_logging, log_experiment_start, log_error
from .sim.training_loop import AtomOrchestrator
from .checkpoint import CheckpointManager, load_system_checkpoint, TrainingState
from .challenges.supersonic_wedge_challenge import run_challenge as run_supersonic
from .challenges.cylinder import run_challenge as run_cylinder


def _add_global_cli_options(parser: argparse.ArgumentParser) -> None:
    """Add global CLI options shared by main and script entrypoints."""
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file (YAML)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="ATOM: Neuro-Symbolic General Purpose Scientific Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the full system
  atom-train --config config/training.yaml --experiment my_experiment

  # Run physics teacher
  atom-teacher --config config/physics.yaml

  # Resume training from checkpoint
  atom-train --resume checkpoints/my_experiment/latest_checkpoint.json

  # Evaluate a trained model
  atom-eval --checkpoint checkpoints/my_experiment/best_model.pth --episodes 100

  # Run the supersonic challenge
  atom run supersonic --steps 1000
        """
    )

    # Global options
    _add_global_cli_options(parser)

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the ATOM system")
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint"
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps"
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Checkpoint saving interval"
    )

    # Teacher command
    teacher_parser = subparsers.add_parser("teacher", help="Run physics teacher simulation")
    teacher_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for simulation data"
    )
    teacher_parser.add_argument(
        "--steps",
        type=int,
        help="Number of simulation steps"
    )

    # Dream command
    dream_parser = subparsers.add_parser("dream", help="Run symbolic distillation (dream mode)")
    dream_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to load for dreaming"
    )
    dream_parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file for discovered laws"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint to evaluate"
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes"
    )

    # Run command (Challenges)
    run_parser = subparsers.add_parser("run", help="Run a specific challenge or experiment")
    run_parser.add_argument(
        "challenge",
        choices=["supersonic", "cylinder", "custom"],
        help="Name of the challenge to run"
    )
    run_parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps to run"
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI visualization"
    )
    run_parser.add_argument(
        "--geometry",
        type=str,
        default=None,
        help="Path to .stl file (for custom challenge)"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument(
        "action",
        choices=["show", "validate", "template"],
        help="Configuration action"
    )
    config_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for template"
    )

    return parser


def load_configuration(args: argparse.Namespace) -> AtomConfig:
    """Load and configure the system."""
    # Load from file if specified
    if args.config:
        if not args.config.exists():
            raise ConfigurationError(f"Configuration file not found: {args.config}")
        try:
            config = AtomConfig.from_yaml(args.config)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {args.config}: {e}")
    else:
        config = AtomConfig()

    # Apply command-line overrides
    if args.experiment:
        config.experiment_name = args.experiment
    if args.seed is not None:
        config.seed = args.seed
    if args.verbose or args.debug:
        config.logging.level = "DEBUG" if args.debug else "INFO"

    return config


def main_train(args: argparse.Namespace) -> int:
    """Run training command."""
    try:
        config = load_configuration(args)

        # Apply training-specific overrides
        if args.max_steps:
            config.training.max_steps = args.max_steps
        if args.checkpoint_interval:
            config.training.save_interval = args.checkpoint_interval

        # Setup logging
        setup_logging(config)

        # Log experiment start
        log_experiment_start(config.experiment_name, config.dict())

        # Initialize orchestrator
        orchestrator = AtomOrchestrator(config=config)

        # Resume from checkpoint if specified
        if args.resume:
            checkpoint_manager = CheckpointManager(
                config.training.checkpoint_dir,
                config.experiment_name
            )
            training_state = TrainingState()
            load_system_checkpoint(
                checkpoint_manager,
                orchestrator.brain,
                orchestrator.memory,
                orchestrator.scientist,
                training_state,
                args.resume
            )

        # Run training
        orchestrator.run()

        return 0

    except AtomError as e:
        log_error(e)
        print(f"ATOM Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        error = AtomError(f"Unexpected error: {e}")
        log_error(error)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main_teacher(args: argparse.Namespace) -> int:
    """Run teacher simulation command."""
    try:
        from .sim import ground_truth as ground_truth_sim

        config = load_configuration(args)
        setup_logging(config)

        # Apply teacher-specific overrides
        if args.output_dir:
            config.output_dir = args.output_dir
            ground_truth_sim.OUTPUT_DIR = str(args.output_dir)
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.steps:
            ground_truth_sim.num_steps = args.steps

        # Run simulation
        ground_truth_sim.run_simulation()

        return 0

    except AtomError as e:
        log_error(e)
        print(f"ATOM Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        error = AtomError(f"Unexpected error: {e}")
        log_error(error)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main_dream(args: argparse.Namespace) -> int:
    """Run dream/symbolic distillation command."""
    try:
        config = load_configuration(args)
        setup_logging(config)

        # Load checkpoint
        checkpoint_manager = CheckpointManager(
            config.training.checkpoint_dir,
            config.experiment_name
        )

        # Initialize components
        orchestrator = AtomOrchestrator(config=config)
        training_state = TrainingState()

        # Load from checkpoint
        load_system_checkpoint(
            checkpoint_manager,
            orchestrator.brain,
            orchestrator.memory,
            orchestrator.scientist,
            training_state,
            args.checkpoint
        )

        # Run symbolic distillation
        print("🧠 Starting symbolic distillation (dream mode)...")
        orchestrator.run_dream_mode()

        # Save discovered laws
        if args.output_file:
            orchestrator.scientist.save_laws(args.output_file)

        return 0

    except AtomError as e:
        log_error(e)
        print(f"ATOM Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        error = AtomError(f"Unexpected error: {e}")
        log_error(error)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main_run(args: argparse.Namespace) -> int:
    """Run challenge command."""
    try:
        # No config loading needed for simple challenge run, 
        # as it builds its own config. But we could load if needed.
        
        if args.challenge == "supersonic":
            run_supersonic(max_steps=args.steps, headless=args.headless)
        elif args.challenge == "cylinder":
            run_cylinder(max_steps=args.steps, headless=args.headless)
        elif args.challenge == "custom":
            if not args.geometry:
                print("❌ Error: --geometry <path_to.stl> is required for 'custom' challenge.")
                return 1
            from atom.challenges.custom import run_custom_challenge
            run_custom_challenge(args.geometry, args.steps, args.headless)
        
        return 0

    except Exception as e:
        print(f"Error running challenge: {e}", file=sys.stderr)
        return 1


def main_eval(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    try:
        config = load_configuration(args)
        setup_logging(config)

        if not args.checkpoint.exists():
            raise ConfigurationError(f"Checkpoint not found: {args.checkpoint}")

        print("🚀 Starting evaluation...")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Render: {args.render}")

        checkpoint_manager = CheckpointManager(
            config.training.checkpoint_dir,
            config.experiment_name,
        )

        orchestrator = AtomOrchestrator(config=config)
        training_state = TrainingState()
        load_system_checkpoint(
            checkpoint_manager,
            orchestrator.brain,
            orchestrator.memory,
            orchestrator.scientist,
            training_state,
            args.checkpoint,
        )

        obs, _ = orchestrator.world.reset()
        obs_t = orchestrator._tensorify(obs)
        hx = None
        device = config.get_device()
        action_dim = int(getattr(config.brain, "action_dim", 1))
        theory_dim = int(getattr(orchestrator.brain, "theory_dim", getattr(config.brain, "theory_dim", 1)))
        last_action = torch.zeros(1, action_dim, device=device)
        theory_t = torch.zeros(1, theory_dim, device=device)
        theory_conf_t = torch.ones(1, 1, dtype=torch.float32, device=device)

        max_passes = max(1, min(int(args.episodes), 1024))
        action_l1_total = 0.0
        value_total = 0.0

        with torch.no_grad():
            for _ in range(max_passes):
                (mu, _std), value, hx_new, _stress = orchestrator.brain(
                    obs_t, theory_t, last_action, hx, theory_confidence=theory_conf_t
                )
                action = torch.tanh(mu).detach()
                action_l1_total += float(action.abs().mean().item())
                value_total += float(value.mean().item())
                last_action = action
                hx = hx_new

        mean_action_l1 = action_l1_total / max_passes
        mean_value = value_total / max_passes
        loaded_step = int(getattr(training_state, "step", 0))
        loaded_epoch = int(getattr(training_state, "epoch", 0))
        print("✅ Evaluation complete")
        print(f"  Loaded step: {loaded_step}")
        print(f"  Loaded epoch: {loaded_epoch}")
        print(f"  Mean |action|: {mean_action_l1:.6f}")
        print(f"  Mean value: {mean_value:.6f}")
        return 0

    except AtomError as e:
        log_error(e)
        print(f"ATOM Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        error = AtomError(f"Unexpected error: {e}")
        log_error(error)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main_config(args: argparse.Namespace) -> int:
    """Run configuration management command."""
    try:
        if args.action == "show":
            config = load_configuration(args)
            print("Current Configuration:")
            print("=" * 50)
            for section_name, section in config.dict().items():
                print(f"\n{section_name.upper()}:")
                if isinstance(section, dict):
                    for key, value in section.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {section}")

        elif args.action == "validate":
            config = load_configuration(args)
            print("✅ Configuration is valid")
            # Additional validation could be added here

        elif args.action == "template":
            # Create a template configuration
            template_config = AtomConfig()
            output_path = args.output or Path("config_template.yaml")
            template_config.to_yaml(output_path)
            print(f"📝 Template configuration saved to: {output_path}")

        return 0

    except AtomError as e:
        print(f"ATOM Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def _dispatch_command(args: argparse.Namespace) -> int:
    """Route parsed arguments to the corresponding command handler."""
    command_handlers = {
        "train": main_train,
        "teacher": main_teacher,
        "dream": main_dream,
        "eval": main_eval,
        "config": main_config,
        "run": main_run,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


def _main_with_argv(argv: Optional[list[str]] = None) -> int:
    """Main CLI execution path with optional argv override for script wrappers."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return _dispatch_command(args)


def _parse_entrypoint_args(command: str, raw_argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse script-entrypoint args while allowing global options anywhere."""
    argv = list(sys.argv[1:] if raw_argv is None else raw_argv)

    global_parser = argparse.ArgumentParser(add_help=False)
    _add_global_cli_options(global_parser)
    global_args, remaining = global_parser.parse_known_args(argv)

    parser = create_parser()
    args = parser.parse_args([command, *remaining])

    for field_name in ("config", "experiment", "seed", "verbose", "debug"):
        value = getattr(global_args, field_name)
        if value != global_parser.get_default(field_name):
            setattr(args, field_name, value)

    return args


def main() -> int:
    """Main CLI entry point."""
    return _main_with_argv()


def main_train_entry() -> int:
    """Console-script entry point for atom-train."""
    return _dispatch_command(_parse_entrypoint_args("train"))


def main_teacher_entry() -> int:
    """Console-script entry point for atom-teacher."""
    return _dispatch_command(_parse_entrypoint_args("teacher"))


def main_dream_entry() -> int:
    """Console-script entry point for atom-dream."""
    return _dispatch_command(_parse_entrypoint_args("dream"))


def main_eval_entry() -> int:
    """Console-script entry point for atom-eval."""
    return _dispatch_command(_parse_entrypoint_args("eval"))


if __name__ == "__main__":
    sys.exit(main())
