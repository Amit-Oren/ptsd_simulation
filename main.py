from simulation.runner import SimulationRunner
from data.input import get_available_ids

if __name__ == "__main__":
    first_patient = get_available_ids()[0]

    runner = SimulationRunner(
        num_patients=1,
        num_turns=10,
        model="gpt-4o-mini",
        seed=1414,
        patient_ids=[first_patient],
        obs_index=0,
    )
    runner.run()
