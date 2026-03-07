from simulation.runner import SimulationRunner

if __name__ == "__main__":
    runner = SimulationRunner(
        num_patients=1,
        num_turns=5,
        model="gpt-4o-mini",
        seed=666
        
        # For testing specific patients and conversation patterns
        
        # subnums=[101],
        # row_index=0,
        # conversation_pattern={"emotion": "anxious", "intensity": "severe",
        #                   "expressiveness": "low", "disclosure": "low",
        #                   "politeness": "formal", "assertiveness": "high"}
    )
    runner.run()
