# laser_polio
This is a spatial polio transmission model built on the LASER framework.

|||
|-|-|
|**docs**|[![Documentation Status](https://img.shields.io/readthedocs/laser-polio.svg)](https://docs.idmod.org/projects/laser-polio/en/latest/)|
|**tests**|[![GitHub Actions Build Status](https://github.com/InstituteforDiseaseModeling/laser-polio/actions/workflows/github-actions.yml/badge.svg)](https://github.com/InstituteforDiseaseModeling/laser-polio/actions) [![Code Coverage](https://codecov.io/gh/InstituteforDiseaseModeling/laser-polio/branch/main/graphs/badge.svg?branch=main)](https://app.codecov.io/github/InstituteforDiseaseModeling/laser-polio)|
|**package**|[![PyPI Package Latest Release](https://img.shields.io/pypi/v/laser-polio.svg)](https://pypi.org/project/laser-polio) [![PyPI Wheel](https://img.shields.io/pypi/wheel/laser-polio.svg)](https://pypi.org/project/laser-polio) [![Supported Versions](https://img.shields.io/pypi/pyversions/laser-polio.svg)](https://pypi.org/project/laser-polio) [![Supported Implementations](https://img.shields.io/pypi/implementation/laser-polio.svg)](https://pypi.org/project/laser-polio) [![Commits since latest release](https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-polio/v0.1.0.svg)](https://github.com/InstituteforDiseaseModeling/laser-polio/compare/v0.1.0...main)|

## Installation
The recommended approach is to use uv to setup your venv and install the package.

Install uv & setup your venv:
```
pip install uv
uv venv --python 3.12
```

Download the repo, then install with uv:
```
uv pip install -e .
```

## Usage
You can run a simple demo with `examples/demo_nigeria.py`

## Repo organization
All core model code is located in the `src\laser_polio` subfolder. The majority of the model architecture resides in `model.py` which contains classes like `SEIR_ABM`, `DiseaseState_ABM`, `Transmission_ABM`, `VitalDynamics_ABM`, `RI_ABM`, and `SIA_ABM`. These classes contain methods for running the sim, tracking infection status, transmitting, managing births and deaths, and applying vaccination interventions. The `distributions.py` file contains a `Distributions` class which facilitates specification of distributions in the pars (e.g., lp.normal(mean=3, std=1), see `examples/demo_nigeria.py`). The  `utils.py` file has a variety of helper functions to handle dates, dot_names (e.g., AFRO:NIGERIA:ZAMFARA:ANKA), and process data. The `src/laser_polio/archive/seir_mpm.py` file contains an experimental meta-population model which will be developed at a future date.

The contents of the other folders is as follows:
- The **data** folder contains curated datasets. The raw versions of the data along with the curation scripts can be found in `data\curation_scripts`.
- The **docs** folder contains information about model design and architecture.
- The **scripts** folder contains code for running and calibrating the model, profiling, and demos.
- The **tests** folder contains code for testing model functionality and benchmarking.

## Order of operations and time
The sim records initial conditions on day 0. As such, sim.results objects will be 1 longer than the specified simulation duration (sim.par.dur). On day 0, the results are logged and the clock is advanced without running any components (e.g., step() not run).

It is strongly suggested (read REQUIRED) that the order for running components is: VitalDynamics_ABM, DiseaseState_ABM, RI_ABM, SIA_ABM, Transmission_ABM. Deviating from it will likely cause inaccurate model results. This design was chosen to match the order of operations in Starsim.

In the DiseaseState_ABM component, individuals will progress through different disease states (SEIR) by checking their timers (e.g., exposure_timer, infection_timer). If those timers are zero at the beginning of a timestep (e.g., when running step() for DiseaseState_ABM), individuals will progress to the next state. Since transmission is the last component to run, the timers for newly exposed individuals are not decremented that day. To address this, we subtract one from dur_exp during initialization. In step_nb() in DiseaseState_ABM, exposed individuals must be updated and their timers decremented prior to updating infected individuals. Deviating from that order of operations will prevent the timers for newly infected individuals from being decremented and they'll receive and extra day of infectivity.

## Tests
Tests can be run with `python -m pytest tests/`

## Linting
Linting and formatting can be run with ruff `uv run ruff check --fix`
