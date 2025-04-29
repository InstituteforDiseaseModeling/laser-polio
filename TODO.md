# PRIORITIES

@Steve
- Stochastic tests
- New model features

CALIBRATION
- print number of jobs that are about to start
- how many jobs can I run? what's the optimal setup?
- fix report_calib_aks.py - not working says unknown databas optunaDatabase
- update how calib results path is passed from cloud vs to calibrate - don't think it's working. Can test it when they're unequal.
- Seed infections after start date
- Refine how regional groupings are made for N/S Nigeria
- Update README with usage instructions
- Save best trial results & figs

CLOUD
- Talk to JB about mapping calib directory during Docker run (kind of like a network drive)
- might be able to connect aks to kubernetes app in VS code so I can interact with each node
- think there's an aks python library that might provide a link or option for inspecting nodes to see which are running
- web interface for aks -> could allow us to login & browse what's running from online, but probably too many settings for me


# REFINEMENT

ESSENTIAL NEW FEATURES
- Enable vx transmission (& add genome R0 multiplier, Sabin = 1/4; nOPV2 = 1/8)
- Add age pyramid by country
- Reactive SIAs (2 campaigns per OB)
- Add scalar for N Nigeria
- Add CBR by country-year
- Curate the surveillance delays
- Add surveillance delays to reactive SIAs
- Add chronically missed pop. Maybe use a individual prob of participating in SIA?
- Add delays to paralysis (and new_exposed) detection times
- Enabling RI with specific vaccines & dates
- Fix death rates

STRETCH NEW FEATURES
- Rethink distance matrix - could we reduce precision to reduce memory? Or would jut uploading lats and longs be faster?
- Is there a way to only load data & initialize sims once during calibration? How much speedup could we get?
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
    - max_migr_frac - it's not the max, it's the sum of the network rows!
- Targets:
    - Stretch: age distribution
- Levers:
    - Stretch: R0 scalar for N Nigeria
    - Stretch: risk_mult_var or corr_risk_inf
- Record lp version
- Record pkg versions

DEBUGGING
- Ask AI why my model might deviate from KM
- Look for other checks on transmission probability in addition to Kermack and McKendrick relation
- Add transmission tests with run_sim() using real data
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

TESTING
- Make stochastic tests more robust
- Use run_sim for testing.
- Is there a way to only load data & initialize sims once during calibration? How much speedup could we get?
- John G recommends Finite Radiation model as default assumption
- Work with John G to put bounds on gravity model pars??
- Curate the surveillance delays
- Add surveillance delays to reactive SIAs
- Add rule for blackouts (e.g., limiting number of campaigns / year) of maybe 1-2 years
- Use KM's gravity model scaling approach
- Export pars as pkl
- Re-org the data folder to have timestamped files? Or time-stamped folders?
- Check that the SIA schedule dot_names are in my shapes
