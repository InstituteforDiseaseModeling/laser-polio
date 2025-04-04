# Priorities
- Change terminology from SIA efficacy to SIA coverage spatial heterogeneity
- ping Arie & hil about Reff REs in N Nigeria
Objectives:
Total cases
Cases by year -> month
Cases in N vs S Nigeria
Levers: 
R0
R0 scalar for N Nigeria
Gravity model coefficient (k)
Seasonality pars
Stretch: risk_mult_var or corr_risk_inf


- Need to use the immunity_ri_nOPV2 column and use that as including efficacy
- Add underwt fraction back in???
- Move translation of r0_scalars to pre-calib
- Remember to move RE processing out of sim and into data curation
- Check transmission probability with real data. Why do we need R0 so high!?
- Test full models with real data
- Drop ABM term from components
- Export pars as pkl
- Rename variables to distinguish between exposure and infection
- Enable vx transmission (& add genome R0 multiplier, Sabin = 1/4; nOPV2 = 1/8)
- Set a random number seed
- Use KM's gravity model scaling approach
- Update the birth and death plot to summarize by country.
- Plot expected births?
- Calibration
- Save results & specify frequency
- Reactive SIAs
- Check out the optuna sampler options: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html. What did Starsim use? 
- Re-org the data folder to have timestamped files? Or time-stamped folders? 
- Check that the SIA schedule dot_names are in my shapes

# Refinement
- Switch to radiation model (easier to explain cuz the numbers are %within vs %without)
- Count number of Sabin2 or nOPV2 transmissions
- Count number of exportations for calibration
- Enable different RI rates over time
- Do we need sub-adm2 resolution? And if so, how do we handle the distance matrix to minimize file size? Consider making values nan if over some threshold?
- Add EMOD style seasonality
- Fork polio-immunity-mapping repo
- Double check that I'm using the ri_eff and sia_prob values correctly - do I need to multiply sia_prob by vx_eff?
- Get total pop data, not just <5
- Investigate extra dot_names in the pop dataset
- Look into age-specific death rates
- Import/seed infections throughout the sim after OPV use?
- Write pars to disk
- Add partial susceptibility & paralysis protection
- Add distributions for duration of each state
- Add in default pars and allow user pars to overwrite them
- Add CBR by country-year
- Add age pyramid by country
- Calibrate the m (scalar) parameter on the R0 random effect
- Add correlation in vx coverage so it's not random???
- In post(?), resample I count to get a variety of paralysis counts
- Calibration parameter:
    - maybe scalar on nOPV2 efficacy
    - m (scalar) parameter on R0 random effects
- Age-specific R0???
