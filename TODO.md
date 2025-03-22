# Priorities
- Track down what happened to the priority ordering
    # Set the order in which components should be run during step()
    PRIORITY_ORDER = [
        "VitalDynamics_ABM",
        "DiseaseState_ABM",
        "RI_ABM",
        "SIA_ABM",
        "Transmission_ABM"
    ]
- Add vx efficacy by type
- Rename variables to distinguish between exposure and infection
- Testing
    - RI_abm
    - SIA_abm
    - Full models with real data
- Set a random number seed
- Use KM's gravity model scaling approach
- Switch to gpkg files. Combine all the shapes from PIM & save as a single simplified gpkg file. 
- Enable different RI rates over time
- Check the RI & SIA figures - they're plotting strange results
- Update the birth and death plot to summarize by country
- Calibration
- Add step size to components (e.g., like vital dynamics)
- Save results & specify frequency
- Reactive SIAs

# Refinement
- Do we need sub-adm2 resolution? And if so, how do we handle the distance matrix to minimize file size? Consider making values nan if over some threshold?
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
