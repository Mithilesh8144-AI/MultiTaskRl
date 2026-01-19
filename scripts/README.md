# Scripts

Utility scripts for testing and analysis.

## Structure

```
scripts/
├── tests/                           # Test and verification scripts
│   ├── test_brc.py                 # Test BRC implementation
│   ├── test_environments.py        # Test all 3 environment variants
│   ├── test_wind_strength.py       # Test wind power parameters
│   └── verify_shared_dqn.py        # Verify Shared DQN agent
└── analysis/                        # Analysis and plotting scripts
    └── generate_comparison_plots.py # Generate publication-ready plots
```

## Test Scripts

### test_brc.py
Quick test to verify BRC implementation works correctly.
- Tests agent creation
- Tests forward pass
- Tests replay buffer
- Tests distributional outputs (21-atom categorical)

```bash
python scripts/tests/test_brc.py
```

### test_environments.py
Tests all 3 Lunar Lander environment variants work correctly.
- Standard environment
- Windy environment (lateral wind)
- Heavy environment (increased gravity)

```bash
python scripts/tests/test_environments.py
```

### test_wind_strength.py
Finds reasonable wind power for Windy LunarLander.
- Tests different wind power values
- Reports episode lengths and rewards

```bash
python scripts/tests/test_wind_strength.py
```

### verify_shared_dqn.py
Verification script to test Shared DQN agent on each task separately.
- Verifies correct environment assignment
- Tests task-aware vs task-blind modes
- Validates Q-value predictions

```bash
python scripts/tests/verify_shared_dqn.py
```

## Analysis Scripts

### generate_comparison_plots.py
Generates side-by-side comparison plots for all methods.
- Creates publication-ready visualizations
- Highlights key differences between methods
- Saves plots to `results/analysis/`

```bash
python scripts/analysis/generate_comparison_plots.py
```

## Notes

All test scripts can be run directly from the project root:
```bash
python scripts/tests/<script_name>.py
```

Analysis scripts output to `results/analysis/` directory.
