import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from laser_core.propertyset import PropertySet
from tqdm import tqdm

from laser_polio.mods import ages
from laser_polio.mods import fertility
from laser_polio.mods import init_prev
from laser_polio.mods import intrahost
from laser_polio.mods import mortality
from laser_polio.mods import ri
from laser_polio.mods import sia
from laser_polio.mods import transmission
from laser_polio.mymodel import Model

# combine pdfs into single
from laser_polio.utils import combine_pdfs

from . import final_reports

# ## Parameters
#
# We need some parameters now. We will use `PropertySet` rather than a raw dictionary for the "syntactic sugar" of referencing `params.ticks` rather than `params["ticks"]` each time.
#
# Also, we will set the parameters separately as `meta_params` and `measles_params` but combine them into one parameter set for future use. We _could_ create `model.params = PropertySet({"meta":meta_params, "measles":measles_params})` and then reference them "by path" in the subsequent code, e.g., `params.meta.ticks` and `params.measles.inf_mean`.

meta_params = PropertySet(
    {
        "ticks": int(365 * 5),
        # "cbr": 40,  # Nigeria 2015 according to (somewhat random internet source): https://fred.stlouisfed.org/series/SPDYNCBRTINNGA
        "output": Path.cwd() / "outputs",
        "prevalence": 0.0025,  # 2.5% prevalence
        # "eula_age": 5
    }
)
# parameter?

measles_params = PropertySet(
    {
        "exp_mean": np.float32(3.0),
        "exp_std": np.float32(1.0),
        "inf_mean": np.float32(6.0),
        "inf_std": np.float32(2.0),
        # "r_naught": np.float32(14.0),
        "r_naught": np.float32(7.0),
        "seasonality_factor": np.float32(0.125),
        "seasonality_phase": np.float32(182),
        "ri_coverage": np.float32(0.75),
    }
)

network_params = PropertySet(
    {
        "a": np.float32(1.0),  # population 1 power factor
        "b": np.float32(1.0),  # population 2 power factor
        "c": np.float32(2.0),  # distance power factor
        "k": np.float32(137.0),  # gravity constant
        "max_frac": np.float32(0.5),  # maximum fraction of population that can move in a single tick
    }
)


# Set up argparse to parse the input directory
parser = argparse.ArgumentParser(description="Specify the input directory.")
parser.add_argument("input_dir", nargs="?", default=".", help="Path to the input directory (default is current directory)")
parser.add_argument("--viz", action="store_true", help="Enable visualization (default is OFF)")
parser.add_argument("--to_pdf", action="store_true", help="Works with --viz. Write to pdf instead of displaying to screen.")

# Parse the arguments
args = parser.parse_args()
params = PropertySet(meta_params, measles_params, network_params)  # type: ignore
params.beta = 1.4  # model.params.r_naught / model.params.inf_mean # type: ignore
# Assign the input directory to model.params.input_dir
params.input_dir = args.input_dir
params.viz = args.viz  # This will be True if --viz is specified, False otherwise
params.to_pdf = args.to_pdf  # This will be True if --viz is specified, False otherwise

##### params code ends here

# We're going to create the human/agent population from the capacity (expansion slots based on births)
# It will be in model.population

# Encapsulate this in a 'factory' method in Model

model = Model.get(params)

# params/config done, model created from pop data, now do components
# =========================================================================


ages.init(model)
mortality.init(model)
fertility.init(model)
manifest = model.manifest
sia.init(model, manifest)
transmission.init(model, manifest)
ri.init(model)

if model.params.to_pdf:
    combine_pdfs()


# consider `step_functions` rather than `phases` for the following
model.phases = [
    model.propagate_population,
    fertility.step,  # type: ignore
    ages.step,
    mortality.step,  # type: ignore
    intrahost.step,  # type: ignore
    # intrahost.step2, # type: ignore
    transmission.step,  # type: ignore
    # ri.step, # type: ignore
    # immunity.step, # type: ignore
    # sia.step, # type: ignore
]


# ## Running the Simulation
#
# We iterate over the specified number of ticks, keeping track, in `metrics`, of the time spent in each phase at each tick.

model.metrics = []
for tick in tqdm(range(model.params.ticks)):
    """
    if tick == 1:
        model.save( filename="laser_cache/burnin_polio.h5" )

    """
    # """
    if tick == 40:  # outbreak/seeding
        init_prev.init(model)
    # """
    metrics = [tick]

    for phase in model.phases:
        tstart = datetime.now(tz=None)  # noqa: DTZ005
        phase(model, tick)
        tfinish = datetime.now(tz=None)  # noqa: DTZ005
        delta = tfinish - tstart
        metrics.append(delta.seconds * 1_000_000 + delta.microseconds)  # delta is a timedelta object, let's convert it to microseconds
    model.metrics.append(metrics)


# ## Final Population
#
# Let's take a quick look at the final population size accounting for births over the course of the simulation. This does _not_ account for non-disease deaths so we are looking at the maximum number of unique agents over the simulation.

print(f"{model.population.count=:,} (vs. requested capacity {model.population.capacity=:,})")

# ## Timing Metrics Part I
#
# Let's convert the timing information to a DataFrame and peek at the first few entries.


final_reports.report(model)
