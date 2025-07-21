# --- PRIORITIES ---
- Check the scenarios that KF/HL put together. Make sure they're feasible with our setup. 
- Add ability to initialize sim from a finished sim (e.g., post-calibration)
- Add age pyramid by country (cuz we're getting close to running beyond Nigeria!)
- Make synthetic calibration with JB - try just KANO? Or a small geography? SK will noodle on it, write up script for making synthetic data, and send to JB. 
- Try adding --no-cache to the docker image
- Try using PIM scalars again
- Sweep over zero inflation by strain

CALIBRATION
- Run sim for top 1-10 calibrations
- Try a missed pop frac scalar based on X (immunity, underwt, sia_coverage)
- Calibrate with missed pop
- Sweep over radiation k and make a scatterplot like I did for gravity
- Sweep over hetero with best pars from calib
- Plot out 'bad' calib trials, get a smattering of options for each r0, both high and low likelihoods

NEW FEATURES
- Enable RI for nOPV2
- Add age pyramid by country
- Reactive SIAs
- Perf improvements
- Try alternate migration models like discrete radiation (JG recommends)
- Add scalar for N Nigeria
- Add CBR by country-year
- Curate the surveillance delays
- Add surveillance delays to reactive SIAs
- Enabling RI with specific vaccines & dates
- Fix death rates


# --- REFINEMENT/STRETCH ---

 NEW FEATURES
- Revisit how we initialize the population. Should we just initialize a laserframe for under 15 year olds? 
- Add rule for blackouts (e.g., limiting number of campaigns / year) of maybe 1-2 years
- Count number of Sabin2 or nOPV2 transmissions
- Count number of exportations for calibration
- Enable different RI rates over time
- Add EMOD style seasonality
- Look into age-specific death rates
- Save results & specify frequency
- Try adding pop density as another r0_scalar
- Adjust vx coverage for non-missed agents by prob/(1-missed_frac)
- Add correlation in vx coverage so it's not random???
- Age-specific R0???
- Consider age distribution of cases as a calibration target, which would help parameterize R0

CALIBRATION
- Dirchlet multi for likelihood on counts: https://github.com/starsimhub/starsim/blob/a253336142f499d0afc93693614830bce9c30a6d/starsim/calib_components.py#L431
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
- Refine how regional groupings are made for N/S Nigeria

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
- Update README with usage instructions
- Try running calibration on a variety of resources/methods (aks, COMPS, databricks) & write up report

TESTING
- Add transmission tests with run_sim() using real data
