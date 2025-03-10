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
You can run a simple demo with `scripts/demos/demo_nigeria.py`

## Repo organization
All core model code is located in the `src\laser_polio` subfolder. The majority of the model architecture resides in `seir_abm.py` which contains classes like `SEIR_ABM`, `DiseaseState_ABM`, `Transmission_ABM`, `VitalDynamics_ABM`, `RI_ABM`, and `SIA_ABM`. These classes contain methods for running the sim, tracking infection status, transmitting, managing births and deaths, and applying vaccination interventions. The `distributions.py` file contains a `Distributions` class which facilitates specification of distributions in the pars (see `scripts/demos/demo_nigeria.py`) so that in the model you can call the distribution from pars and the only input it requires is the number of draws. The  `utils.py` file has a variety of helper functions. The `seir_mpm.py` file contains an experimental meta-population model which will be developed at a future date.

The contents of the other folders is as follows:
- The data folder contains curated files needed for modeling along with the raw versions of the data and curation scripts.
- The docs folder contains information about model design and architecture.
- The scripts folder contains code for running and calibrating the model, profiling, and demos. 
- The tests folder contains code for testing model functionality and benchmarking. 

## Tests
Tests can be run `python -m pytest tests/`
