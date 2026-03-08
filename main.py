from simulation.runner import SimulationRunner

if __name__ == "__main__":
    runner = SimulationRunner(
        num_patients=3,
        num_turns=5,
        model="gpt-4o-mini",
        seed=1414
        
        # # For testing specific patients and conversation patterns
        
        # subnums=[7],
        # row_index=2,
        # conversation_pattern={"emotion": "angry", "intensity": "severe",
        #                   "expressiveness": "low", "disclosure": "low",
        #                   "politeness": "casual", "assertiveness": "high"}
    )
    runner.run()
