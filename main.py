import argparse
import yaml
from datetime import datetime
from pathlib import Path
from config.logging_config import setup_logging
from simulation.run_context import set_experiment_dir
from simulation.runner import SimulationRunner


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PTSD simulation.")
    parser.add_argument("--config", type=str, default="config/simulation_config.yaml",
                        help="Path to simulation config YAML (default: config/simulation_config.yaml)")
    parser.add_argument("--num-patients", type=int, default=None,
                        help="Number of synthetic patients to generate")
    parser.add_argument("--sessions", type=int, default=None,
                        help="Number of sessions per patient")
    parser.add_argument("--turns", type=int, default=None,
                        help="Number of conversation turns per session")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Max validation retries")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM temperature (0.0–2.0)")
    parser.add_argument("--pipeline", type=str, default=None,
                        help="Pipeline preset: full, full_moderated, no_validation, vignette_only, ...")
    parser.add_argument("--n-items", type=int, default=None,
                        help="Number of self-report items per node")
    parser.add_argument("--node-prob", type=float, default=None,
                        help="Probability of activating each cognitive model node (0.0–1.0)")
    parser.add_argument("--edge-prob", type=float, default=None,
                        help="Probability of activating each edge between active nodes (0.0–1.0)")
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
        num_sessions  = args.sessions      or sim["num_sessions"],
        num_turns     = args.turns         or sim["num_turns"],
        seed          = args.seed          or sim["seed"],
        max_retries   = args.max_retries   or sim["max_validation_retries"],
        temperature   = args.temperature   if args.temperature is not None else sim["temperature"],
        pipeline      = args.pipeline or sim.get("pipeline", "full"),
        n_items       = args.n_items       or sim.get("n_items", 3),
        node_prob     = args.node_prob     if args.node_prob  is not None else sim.get("node_prob", 0.7),
        edge_prob     = args.edge_prob     if args.edge_prob  is not None else sim.get("edge_prob", 0.5),
        models        = cfg["models"],
        agent_classes = cfg["agents"],
        experiment_dir = experiment_dir,
    )
    runner.run()
