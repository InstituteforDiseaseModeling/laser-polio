# Priorities
- Check the RI & SIA figures - they're plotting strange results
- Merge my interventions branch

- Try running calibration by myself - see the docs

- Curate SIA data from Hil: Kurt_sharing\step04b_sia_randeffect\20241220_regression
- Change terminology from SIA efficacy to SIA coverage spatial heterogeneity
- Need to use the immunity_ri_nOPV2 column and use that as including efficacy
- Remember to move RE processing out of sim and into data curation

- Curate R0 file from Hil: Kurt_sharing\step04b_sia_randeffect\20241220_regression
- Try scaling R0. E.g. per convo with HL & KF: R0_i = exp{m * (b_i - mean(b))/sd(b) + log R0), so the random effect b_i is standardized to mean zero and std dev = 1, then rescaled with m and centered. For example if you thought that R0 is usually centered at 10 but ranges from 5 to 20, then you would pick m to scale within those bounds.
- Check transmission probability with real data. Why do we need R0 so high!?
- Test full models with real data
- Drop ABM term from components
- Export pars as pkl
- Rename variables to distinguish between exposure and infection
- Enable vx transmission
- Set a random number seed
- Use KM's gravity model scaling approach
- Update the birth and death plot to summarize by country.
- Plot expected births?
- Calibration
- Save results & specify frequency
- Reactive SIAs

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
