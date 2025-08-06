# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

laser-polio is a spatial polio transmission model built on the LASER framework. It models poliovirus transmission, vaccination campaigns, and disease dynamics across geographic regions (primarily Nigeria and West Africa). The model uses agent-based modeling with SEIR (Susceptible-Exposed-Infected-Recovered) compartments.

## Development Commands

### Installation & Environment
```bash
# Install dependencies using uv (recommended)
uv pip install -e .

# Upgrade laser-core to latest version
uv pip install --upgrade laser-core
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
pytest --cov --cov-report=term-missing --cov-report=xml -vv tests
```

### Linting & Formatting
```bash
# Run linting and auto-fix issues
uv run ruff check --fix

# Format code (ruff is configured in pyproject.toml)
uv run ruff format
```

### Running Examples
```bash
# Basic Nigeria demo
python examples/demo_nigeria.py

# Other available demos
python examples/demo_zamfara.py
python examples/demo_west_africa.py
python examples/demo_africa.py
```

## Code Architecture

### Core Components (src/laser_polio/)

The simulation consists of several ABM (Agent-Based Model) components that run in a specific order each timestep:

1. **VitalDynamics_ABM**: Manages births, deaths, and aging
2. **DiseaseState_ABM**: Handles SEIR state transitions and disease progression
3. **RI_ABM**: Routine immunization (ongoing vaccination programs)
4. **SIA_ABM**: Supplementary immunization activities (vaccination campaigns)
5. **Transmission_ABM**: Virus transmission between individuals and nodes

**CRITICAL ORDER**: Components must run in the exact order above to ensure accurate results. This matches Starsim's order of operations.

### Key Files

- **model.py**: Contains all ABM component classes and core simulation logic
- **pars.py**: Parameter definitions and default values
- **utils.py**: Helper functions for dates, dot_names (e.g., AFRO:NIGERIA:ZAMFARA:ANKA), and data processing
- **distributions.py**: Distributions class for specifying parameter distributions (e.g., `lp.normal(mean=3, std=1)`)
- **plotting.py**: Visualization functions for simulation results
- **run_sim.py**: High-level simulation runner

### Calibration System (calib/)

- **calibrate.py**: Main calibration script using Optuna optimization
- **calib_configs/**: YAML configuration files for different calibration scenarios
- **model_configs/**: YAML configuration files for different model setups
- **cloud/**: Scripts for running calibration on Azure Kubernetes Service

### Data Structure

The model uses curated datasets located in `data/`:
- Population data (WorldPop estimates)
- Age pyramids by country
- Birth rates, vaccination rates, case data
- Distance matrices for migration modeling
- Shapefiles for geographic mapping

## Key Design Patterns

### Performance Optimization
- Use numba for computationally intensive operations
- Avoid two-stage query-and-apply patterns (query and act directly)
- LaserFrame data structure optimized for agent-based operations

### Timer Management
Disease progression uses timer-based state transitions. Timers are decremented at the beginning of each timestep, so `dur_exp` is reduced by 1 during initialization to account for this.

### Geographic Modeling
- Uses dot_name format: `AFRO:NIGERIA:ZAMFARA:ANKA` for hierarchical geography
- Supports gravity and radiation models for migration
- Administrative level 2 (adm2) resolution for most operations

## Common Development Tasks

### Adding New Parameters
1. Define in `pars.py` with appropriate defaults
2. Add distribution specification if needed in parameter configs
3. Update relevant component classes to use the parameter

### Running Calibration
```bash
# Local calibration example
python calib/calibrate.py --config calib/calib_configs/r0_k.yaml --model_config calib/model_configs/config_zamfara.yaml

# Docker-based calibration
python calib/run_calib_docker_local.py
```

### Memory Profiling
```bash
python scripts/profiling/memory_usage_over_t.py
```

## Testing Strategy

Tests cover:
- Component initialization and basic functionality
- Disease state transitions and timing
- Transmission mechanics
- Intervention effects (RI, SIA)
- Data loading and processing

## Configuration Files

- **pyproject.toml**: Python package configuration, dependencies, ruff settings
- **pytest.ini**: Test configuration with strict warnings
- **pyrightconfig.json**: Type checking disabled (numba compatibility)

## Performance Notes

- Model designed for populations of 100k-1M agents
- Uses sparse matrices for migration when possible
- Logging configured to timestamped files in `logs/` directory
- Results saved to timestamped directories in `results/`
