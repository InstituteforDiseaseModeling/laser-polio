# Purpose
This folder contains curated datasets in root, plus the raw data and curation scripts in the subfolders.

# Data sources
The shapefiles & population data are sourced Dropbox/Kurt_sharing.

| Dataset | Source | Basis | Status |
|---------|---------|--------|---|
| Age distribution | [UN World Population Prospects](https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F02_1_POPULATION_5-YEAR_AGE_GROUPS_BOTH_SEXES.xlsx) 5-year age groups for both sexes | UNWPP | Done |
| Birth rates (CBR) | Kurt (source code: ???) | UNWPP | Done |
| Case data (paralysis) | ??? | POLIS | TODO |
| Death rates | ??? | ??? | TODO |
| Individual risk corr | [Rasters of U5 underweight fraction curated by Kurt](https://bmgf.sharepoint.com/sites/Measles/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FMeasles%2FShared%20Documents%2FTeam%20Documents%2FArchive%2FCoverages%2FIHME%2FCGF%5FWORLD%5F2020%5F08%5F31&viewid=088c215a%2D73e3%2D4ef7%2D9801%2Df2e003a79b7f&ct=1740691057970&or=OWA%2DNT%2DMail&ga=1) | [IHME](https://ghdx.healthdata.org/record/ihme-data/global-child-growth-failure-geospatial-estimates-2000-2019) | Done |
| Initial immunity | [polio-immunity-mapping](https://github.com/InstituteforDiseaseModeling/polio-immunity-mapping) Open R/immunity_calc/immunity_calc_eag.R, update the age bins in the 'calculate immunity' step ~line 197, and run the script using a terminal to specify coverage in the args, e.g., `Rscript R/immunity_calc/immunity_calc_eag.R scn=cvd2 coverage=0.5` -> scn/cvd2/results/immunity_age_groups_0.5coverage.rds | Polio immunity mapper | Done |
| Population counts |  Kurt (source code: ???) | WorldPop total population estimates | Done |
| R_eff random effects | Hil (source code: ???) | Regression model | Done |
| Routine immunization rates | Hil (source code: ???) | Regression model | Done |
| Shapefiles | Hil (source code: ???) | POLIS | Done |
| SIA calendar, historic | [polio-immunity-mapping](https://github.com/InstituteforDiseaseModeling/polio-immunity-mapping) `dvc repro` -> scn/cvd2/results/sia_district_rows.csv | POLIS | Done |
| SIA calendar, prospective | ??? | IDM | TODO |
| SIA efficacy | Hil (source code: ???) | Regression model | Done |

---


# Workflow
0. Obtain the latest shapefiles, population counts, dpt3, and sia calendar from the polio-immunity-mapping repo using `dvc pull` and copy them into their respective spots in laser-polio/data/curation_scripts. For more details on this workflow, see the section below titled 'polio-immunity-mapping repo'.
1. Run shapes/curate_shapes.R to curate the POLIS shapefiles for Africa.
2. Run shapes/calc_dist_matrix.py to calculate the distances (in km) between all admin2 units in Africa.
3. Run pop/curate_pop.py to curate the population data.
4. Run ri/curate_ri.py to curate the ri efficacy data.
5. Run sia/curate_sia.py to curate the SIA efficacy data.
6. Run compile_curated_data.py to combine the CBR, pop, RI, and SIA random effects into one dataset.

7. TODO: Curate the initial immunity conditions
8. TODO: Curate the age distribution data
9. TODO: Curate the case data
7. TODO: Validate the data & make sure the dot_names are all in the same order.


## TODO
- [ ] Initial immunity conditions
- [ ] Age distributions by country - use UNWPP, subnational as a stretch goal
- [ ] Death rates by age? - use UNWPP?
- [ ] Future SIA schedules
- [ ] Correlate of risk - Underweight fraction from IHME?
- [ ] Case data
- [ ] Downsample the admin2 shp


---


# Polio immunity mapping repo
Several datasets are sourced from the [polio-immunity-mapping](https://github.com/InstituteforDiseaseModeling/polio-immunity-mapping) repo. Instructions for generating the latest version of this dataset using the polio-immunity-mapping repo are below.

## If you've already run the polio-immunity-mapping repo:
1. Go to the repo, open a terminal (powershell, cmd, etc.), enter `dvc pull`. The file is located in data_local/dpt_district_summaries.csv.

## If this is your first time using the polio-immunity-mapping repo:
1. Clone the polio-immunity-mapping repo.
2. Download the dvc connection file [here](https://bmgf-my.sharepoint.com/:f:/g/personal/dejan_lukacevic_gatesfoundation_org/Eh_bnBEdFAEVEtLwu9qtxiwBfGi4JHSfBbvU2C0MV3to4w "https://bmgf-my.sharepoint.com/:f:/g/personal/dejan_lukacevic_gatesfoundation_org/eh_bnbedfaevetlwu9qtxiwbfgi4jhsfbbvu2c0mv3to4w") and save it to the repo in the data_local folder in the root. File access granted by Arie or Dejan.
3. Run `./pipeline/configure_dvc.cmd` to set dvc 'remote' and pull the data
4. Add a remote using the following script in a terminal:
	```
	# Add a remote
	dvc remote add -d -f --local polio-remote azure://data
	dvc remote modify    --local polio-remote connection_string "...paste conn string from a file..."

	# Confirm it was added
	dvc remote list --local

	# Pull data
	dvc pull
	```
 5. If dvc pull is throwing errors, you can try `dvc pull --force`
