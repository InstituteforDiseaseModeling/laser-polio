# Priorities
- Testing
    - DiseaseState_abm
    - Transmission_abm
    - RI_abm
    - SIA_abm
- Add 1 to the durs b/c we're shortchanging them due to order of operations???
- Starsim order of operations (https://github.com/starsimhub/starsim/blob/main/starsim/loop.py)
    1. Start_step (multipliers like seasonality, rel_sus/RR_sus, rel_trans)
    2. Demographics
    3. State changes
    4. Interventions
    5. Transmission
    6. Log results
- Add default pars, then replace if user specifies pars
- Set a random number seed
- Use KM's gravity model scaling approach
- Check the RI & SIA figures - they're plotting strange results
- Update the birth and death plot to summarize by country
- Calibration
- Add step size to components (e.g., like vital dynamics)
- Save results & specify frequency
- Reactive SIAs

# Refinement
- Add EMOD style seasonality
- Fork polio-immunity-mapping repo
- Double check that I'm using the ri_eff and sia_prob values correctly - do I need to multiply sia_prob by vx_eff?
- Get total pop data, not just <5
- Investigate extra dot_names in the pop dataset
- Look into age-specific death rates
- Setup EULAs - currently are only age based, needs to be immunity based
- Import/seed infections throughout the sim after OPV use?
- Write pars to disk
- Add partial susceptibility & paralysis protection
- Add distributions for duration of each state
- Add in default pars and allow user pars to overwrite them
- Add CBR by country-year
- Add age pyramid by country
- Calculate distance between gps coordinates
