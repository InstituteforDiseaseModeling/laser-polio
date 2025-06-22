import numpy as np

import laser_polio as lp
from laser_polio.pars import PropertySet

pars = PropertySet(
    {
        "start_date": lp.date("2020-01-01"),
        "dur": 1,
        "n_ppl": np.array([1000, 500]),  # Two nodes with populations
        "cbr": np.array([30, 25]),  # Birth rate per 1000/year
        "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        "init_immun": 0.0,  # 20% initially immune
        "init_prev": 0.0,  # 5% initially infected
        "t_to_paralysis": lp.lognormal(mean=12.5, sigma=3.5),
    }
)
sim = lp.SEIR_ABM(pars)
t_to_paralysis = sim.pars.t_to_paralysis(10000)
print(f"Mean time to paralysis: {np.mean(t_to_paralysis)}")
print(f"Median time to paralysis: {np.median(t_to_paralysis)}")
print(f"Std time to paralysis: {np.std(t_to_paralysis)}")
print(f"Min time to paralysis: {np.min(t_to_paralysis)}")
print(f"Max time to paralysis: {np.max(t_to_paralysis)}")
print(f"95% CI time to paralysis: {np.percentile(t_to_paralysis, [2.5, 97.5])}")

print("Done.")
