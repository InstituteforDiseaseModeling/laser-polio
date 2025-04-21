# Priorities


CALIBRATION
- Use static image renderers for plotly for optuna figs: https://plotly.com/python/renderers/#:~:text=similar%20for%20Nteract.-,Static%20Image%20Renderers,-A%20set%20of
- Fix the shape mismatch in calibration arrays due to wrapping into next year
- Seed infections after start date
- Run calibration (with all the latest changes) on the cluster
- Use more pars for Nigeria
- Targets:
    - Stretch: age distribution
- Levers:
    - Stretch: R0 scalar for N Nigeria
    - Stretch: risk_mult_var or corr_risk_inf

NEW FEATURES
- Rethink distance matrix - could we reduce precision to reduce memory? Or would jut uploading lats and longs be faster?
- Add scalar for N Nigeria
- Enable vx transmission (& add genome R0 multiplier, Sabin = 1/4; nOPV2 = 1/8)
- Add age pyramid by country
- Save results & specify frequency
- Reactive SIAs (2 campaigns per OB)
- Add chronically missed pop. Maybe use a individual prob of participating in SIA?

DEBUGGING
- Ask AI why my model might deviate from KM
- Look for other checks on transmission probability in addition to Kermack and McKendrick relation
- Add transmission tests with run_sim() using real data
- Plot all data inputs for visual checks
- Plot expected births?
- Update the birth and death plot to summarize by country.
- Test full models with real data

# Refinement

TESTING
- Use run_sim for testing.
- Is there a way to only load data & initialize sims once during calibration? How much speedup could we get?
- John G recommends Finite Radiation model as default assumption
- Work with John G to put bounds on gravity model pars??
- Calib question: Is there any appetite for making a broadly usable calibration bootstrapping function? For example, paralytic cases are a rare (1/2000) subset of Infections. So after/during calibration, we could resample the infection counts and get a bunch of new paralysis counts essentially for free.
- Curate the surveillance delays
- Add surveillance delays to reactive SIAs
- Add rule for blackouts (e.g., limiting number of campaigns / year) of maybe 1-2 years
- Use KM's gravity model scaling approach
- Export pars as pkl
- Re-org the data folder to have timestamped files? Or time-stamped folders?
- Check that the SIA schedule dot_names are in my shapes

CALIBRATION
- Calibrate the m (scalar) parameter on the R0 random effect
- Calib question: Is there any appetite for making a broadly usable calibration bootstrapping function? For example, paralytic cases are a rare (1/2000) subset of Infections. So after/during calibration, we could resample the infection counts and get a bunch of new paralysis counts essentially for free.
- In post(?), resample I count to get a variety of paralysis counts
- Calibration parameter:
    - maybe scalar on nOPV2 efficacy
    - m (scalar) parameter on R0 random effects

NEW FEATURES
- Add CBR by country-year
- Is there a way to only load data & initialize sims once during calibration? How much speedup could we get?
- Curate the surveillance delays
- Add surveillance delays to reactive SIAs
- Add rule for blackouts (e.g., limiting number of campaigns / year) of maybe 1-2 years
- Count number of Sabin2 or nOPV2 transmissions
- Count number of exportations for calibration
- Enable different RI rates over time
- Add EMOD style seasonality
- Look into age-specific death rates

MIGRATION
- John G recommends Finite Radiation model as default assumption
- Work with John G to put bounds on gravity model pars??
- Use KM's gravity model scaling approach
- Switch to radiation model (easier to explain cuz the numbers are %within vs %without)
- Do we need sub-adm2 resolution? And if so, how do we handle the distance matrix to minimize file size? Consider making values nan if over some threshold?

QUALITY OF LIFE
- Export pars as pkl
- Re-org the data folder to have timestamped files? Or time-stamped folders?

CLEANUP
- Get Hil & Kurt to add links to code in curation_scripts README
- Change terminology from SIA efficacy to SIA coverage spatial heterogeneity
- Rename variables to distinguish between exposure and infection
- Drop ABM term from components

STRETCH
- Add correlation in vx coverage so it's not random???
- Age-specific R0???
