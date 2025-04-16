# Priorities

DEBUGGING
- Plot all data inputs for visual checks
- Check transmission function calculations. Why do we need R0 so high!?
- Check transmission probability with real data. Why do we need R0 so high!?
- Plot expected births?
- Update the birth and death plot to summarize by country.
- Test full models with real data

CALIBRATION
- Try comparing observed paralysis counts to infections / 2000
- Use more pars for Nigeria
- Likelihood fn???
- Targets:
    - Stretch: age distribution
- Levers:
    - Stretch: R0 scalar for N Nigeria
    - Stretch: risk_mult_var or corr_risk_inf

CLEANUP
- Change terminology from SIA efficacy to SIA coverage spatial heterogeneity
- Rename variables to distinguish between exposure and infection
- Drop ABM term from components

NEW FEATURES
- Rethink distance matrix - could we reduce precision to reduce memory? Or would jut uploading lats and longs be faster?
- Add ability to seed infections at specific times & places
    - Use Kurt's approach for when/where to seed infections: BIRINIWA day 37, 2018 & SHINKAFI day 329, 2020
- Add scalar for N Nigeria
- Enable vx transmission (& add genome R0 multiplier, Sabin = 1/4; nOPV2 = 1/8)
- Set a random number seed
- Add age pyramid by country
- Import/seed infections throughout the sim after initialization
- Save results & specify frequency
- Reactive SIAs (2 campaigns per OB)
- Add chronically missed pop. Maybe use a individual prob of participating in SIA?


# Refinement
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
- Switch to radiation model (easier to explain cuz the numbers are %within vs %without)
- Count number of Sabin2 or nOPV2 transmissions
- Count number of exportations for calibration
- Enable different RI rates over time
- Do we need sub-adm2 resolution? And if so, how do we handle the distance matrix to minimize file size? Consider making values nan if over some threshold?
- Add EMOD style seasonality
- Double check that I'm using the ri_eff and sia_prob values correctly - do I need to multiply sia_prob by vx_eff?
- Look into age-specific death rates
- Write pars to disk
- Add CBR by country-year
- Calibrate the m (scalar) parameter on the R0 random effect
- Add correlation in vx coverage so it's not random???
- In post(?), resample I count to get a variety of paralysis counts
- Calibration parameter:
    - maybe scalar on nOPV2 efficacy
    - m (scalar) parameter on R0 random effects
- Age-specific R0???
- Get Hil & Kurt to add links to code in curation_scripts README
