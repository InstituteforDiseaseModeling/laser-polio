# PRIORITIES

@Steve
- Plot monthly cases likelihood component against phase
- Consider age distribution of cases as a calibration target, which would help parameterize R0
- Double check that the seasonality phase is correct based on latest calibration results
- Get cursor
- Calibrate with missed pop
- Calibrate with individ hetero
- Plot top 10 trial
- Try calibrating the timing of seeds???
- Try a universal scalar for init_immunity
- Try calibrating to first 3 years, then run a sim for another 3 y
- Sweep over radiation k and make a scatterplot like I did for gravity
- Sweep over hetero with best pars from calib
- Try calibrating the r0_scalar parameters in run_sim (24 and 0.2)
- Try calibrating the sia rand effects center and scale values
- Plot out 'bad' calib trials, get a smattering of options for each r0, both high and low likelihoods
- Try adding distance from origin as a calib target
- Fix test_sia_schedule() - number of recovered inidividuals does not match expected value
- Adjust vx coverage for non-missed agents by prob/(1-missed_frac)

CALIBRATION
- Refine how regional groupings are made for N/S Nigeria
- Update README with usage instructions

# REFINEMENT

ESSENTIAL NEW FEATURES
- Enable vx transmission (& add genome R0 multiplier, Sabin = 1/4; nOPV2 = 1/8)
- Add age pyramid by country
- Reactive SIAs (2 campaigns per OB)
- Add scalar for N Nigeria
- Add CBR by country-year
- Curate the surveillance delays
- Add surveillance delays to reactive SIAs
- Add delays to paralysis (and new_exposed) detection times
- Enabling RI with specific vaccines & dates
- Fix death rates

STRETCH NEW FEATURES
- Add rule for blackouts (e.g., limiting number of campaigns / year) of maybe 1-2 years
- Count number of Sabin2 or nOPV2 transmissions
- Count number of exportations for calibration
- Enable different RI rates over time
- Add EMOD style seasonality
- Look into age-specific death rates
- Save results & specify frequency
- Add IPV to RI

CALIBRATION
- Chat with Jeremy about databricks
- Calibrate the m (scalar) parameter on the R0 random effect
- Calibration parameter:
    - maybe scalar on nOPV2 efficacy
    - m (scalar) parameter on R0 random effects
- Targets:
    - Stretch: age distribution
- Levers:
    - Stretch: R0 scalar for N Nigeria
    - Stretch: risk_mult_var or corr_risk_inf
- Record lp version
- Record pkg versions

DEBUGGING
- Check why RI seems to stop after certain date
- Plot all data inputs for visual checks
- Plot expected births?
- Update the birth and death plot to summarize by country.

MIGRATION
- John G recommends Finite Radiation model as default assumption
- Work with John G to put bounds on gravity model pars??
- Use KM's gravity model scaling approach
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
- Try running calibration on a variety of resources/methods (aks, COMPS, databricks) & write up report

TESTING
- Add transmission tests with run_sim() using real data
