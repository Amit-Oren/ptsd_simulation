import argparse
import yaml
from datetime import datetime
from pathlib import Path
from config.logging_config import setup_logging
from agents.base_agent import set_experiment_dir
from simulation.runner import SimulationRunner


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PTSD simulation.")
    parser.add_argument("--config", type=str, default="config/simulation_config.yaml",
                        help="Path to simulation config YAML (default: config/simulation_config.yaml)")
    parser.add_argument("--patients", type=int, nargs="+", default=None,
                        help="Specific patient IDs to run (overrides num_patients)")
    parser.add_argument("--num-patients", type=int, default=None,
                        help="Number of patients to randomly sample")
    parser.add_argument("--turns", type=int, default=None,
                        help="Number of conversation turns")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--obs-index", type=int, default=None,
                        help="Observation index to use")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Max validation retries")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM temperature (0.0–2.0)")
    parser.add_argument("--pipeline", type=str, default=None,
                        help="Pipeline preset: full, no_validation, no_analysis, no_validation_no_analysis")
    return parser.parse_args()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/output") / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(experiment_dir)
    set_experiment_dir(experiment_dir)

    args = parse_args()
    cfg = load_config(args.config)

    sim = cfg["simulation"]

    runner = SimulationRunner(
        num_patients  = args.num_patients  or sim["num_patients"],
        num_turns     = args.turns         or sim["num_turns"],
        seed          = args.seed          or sim["seed"],
        obs_index     = args.obs_index     if args.obs_index is not None else sim["obs_index"],
        max_retries   = args.max_retries   or sim["max_validation_retries"],
        temperature   = args.temperature   if args.temperature is not None else sim["temperature"],
        patient_ids   = args.patients or sim.get("patient_ids"),
        pipeline        = args.pipeline or sim.get("pipeline", "full"),
        persona_context = sim.get("persona_context", False),
        models          = cfg["models"],
        agent_classes = cfg["agents"],
        experiment_dir = experiment_dir,
    )
    runner.run()
