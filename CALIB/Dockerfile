# Use Rocky Linux as base
FROM rockylinux:9

RUN dnf install -y gcc gcc-g++ python3-devel

# Install Python 3.12 and dependencies
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy your script into the container
RUN pip install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio
RUN pip install optuna pymysql
COPY objective.py .
COPY worker.py .
COPY mysql_storage.py .
COPY demo_nigeria.py ./laser.py

# Copy additional files into the data directory inside the container
COPY data/init_immunity_0.5coverage_january.h5 /app/data/
COPY data/distance_matrix_africa_adm2.h5 /app/data/
COPY ./data/Nigeria_age_pyramid_2024.csv /app/data/
COPY ./data/sia_scenario_1.csv /app/data/
COPY ./data/compiled_cbr_pop_ri_sia_underwt_africa.csv /app/data/
COPY ./data/age_africa.csv /app/data/
COPY ./data/sia_historic_schedule.csv /app/data/
COPY ./data/shp_names_africa_adm2.csv /app/data/

# Run the script when the container starts
#CMD ["python3", "laser.py", "-p", "shared/params.json", "-o", "shared/simulation_results.csv"]
CMD ["python3", "worker.py"]
