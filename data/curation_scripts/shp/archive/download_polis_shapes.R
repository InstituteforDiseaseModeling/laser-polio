### Download the POLIO ADMIN shapefile from ARC GIS

#devtools::install_github("R-ArcGIS/arcgislayers", dependencies = TRUE)
library(arcgislayers)
library(ggplot2)
library(sf)
library(janitor)

# URL for WHO polio shapefiles: https://services.arcgis.com/5T5nSi527N4F7luB/arcgis/rest/services/POLIO_ADMINISTRATIVE_BOUNDARIES/FeatureServer/
# layers:
# 0 = disputed borders
# 1 = Disputed areas
# 2 = ADM4
# 3 = ADM3
# 4 = ADM2
# 5 = ADM1
# 6 = ADM0
# Add these to the end of the URL


# Admin level 0
furl0 <- "https://services.arcgis.com/5T5nSi527N4F7luB/arcgis/rest/services/POLIO_ADMINISTRATIVE_BOUNDARIES/FeatureServer/6"
fl0 = arc_open(furl0)  # Open connection
emro_africa_countries <- c('EG', 'LY', 'SD', 'TN', 'MA', 'SO', 'DJ')  # List of African EMRO countries
emro_filter <- paste0("ISO_2_CODE IN ('", paste(emro_africa_countries, collapse = "', '"), "')")  # Convert the list into an SQL-friendly format
where_clause <- paste0("WHO_REGION = 'AFRO' OR ", emro_filter) # Construct the WHERE clause to include AFRO region plus selected EMRO countries
shp0 <- arc_select(fl0, where = where_clause)  # Download the feature layer
shp0 <- clean_names(shp0)  # Clean names
shp0 <- shp0[as.Date(shp0$enddate)>=Sys.Date(),] # Filter to current regions

# Admin level 2
furl2 <- "https://services.arcgis.com/5T5nSi527N4F7luB/arcgis/rest/services/POLIO_ADMINISTRATIVE_BOUNDARIES/FeatureServer/4"
fl2 = arc_open(furl2)  # Open connection
emro_africa_countries <- c('EG', 'LY', 'SD', 'TN', 'MA', 'SO', 'DJ')  # List of African EMRO countries
emro_filter <- paste0("ISO_2_CODE IN ('", paste(emro_africa_countries, collapse = "', '"), "')")  # Convert the list into an SQL-friendly format
where_clause <- paste0("WHO_REGION = 'AFRO' OR ", emro_filter) # Construct the WHERE clause to include AFRO region plus selected EMRO countries
shp2 <- arc_select(fl2, where = where_clause)  # Download the freature layer
shp2 <- clean_names(shp2)  # Clean names
shp2 <- shp2[as.Date(shp2$enddate)>=Sys.Date(),] # Filter to current regions
head(shp2)


# Plot the shapefile with alpha set to 0.5
ggplot(data = shp0) +
  geom_sf(alpha = 0.5) +
  theme_minimal() +
  labs(title = "WHO Polio Administrative Boundaries - ADM0",
       caption = "Source: WHO Polio Administrative Boundaries")
ggsave("data/curation_scripts/shapes/map_who_polio_adm0.pdf", width = 8.5, height = 11)

# Plot the shapefile with alpha set to 0.5
ggplot(data = shp2) +
  geom_sf(alpha = 0.5) +
  theme_minimal() +
  labs(title = "WHO Polio Administrative Boundaries - ADM2",
       caption = "Source: WHO Polio Administrative Boundaries")
ggsave("data/curation_scripts/shapes/map_who_polio_adm2.pdf", width = 8.5, height = 11)

# Save the shapefile as a GeoJSON
st_write(shp0, "data/curation_scripts/shapes/polis/polis_adm0_africa.geojson", driver = "GeoJSON", delete_dsn = TRUE)
st_write(shp2, "data/curation_scripts/shapes/polis/polis_adm2_africa.geojson", driver = "GeoJSON", delete_dsn = TRUE)

# Export as a Shapefile (.shp)
st_write(shp0, "data/curation_scripts/shapes/polis/polis_adm0_africa.shp", driver = "ESRI Shapefile", delete_dsn = TRUE)
st_write(shp2, "data/curation_scripts/shapes/polis/polis_adm2_africa.shp", driver = "ESRI Shapefile", delete_dsn = TRUE)

# Filter the shapefile to each unique ISO3 code and export
unique_iso3_codes <- unique(shp2$iso_3_code)
for (iso in unique_iso3_codes) {
  shp2_filtered <- shp2[shp2$iso_3_code == iso, ]
  output_path <- paste0("data/curation_scripts/shapes/polis/polis_adm2_", iso, ".shp")
  st_write(shp2_filtered, output_path, driver = "ESRI Shapefile", delete_dsn = TRUE)
}
