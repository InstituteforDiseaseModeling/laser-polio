# Source
The population data (dpt_district_summaries.csv) are based on UN Worldpop estimates for individuals <5 years of age. These data were pulled from the [polio-immunity-mapping](https://github.com/InstituteforDiseaseModeling/polio-immunity-mapping) repo. Instructions for generating the latest version of this dataset using the polio-immunity-mapping repo are below.

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
