# Use Rocky Linux as base
FROM rockylinux:9

# Update and install dependencies
RUN dnf update -y && \
    dnf install -y epel-release && \
    dnf install -y \
    python3 \
    python3-devel \
    mariadb-connector-c-devel \
    gcc \
    gcc-c++ \
    make \
    wget \
    screen \
    procps \
    vim \
#    git \
#hdf5-devel \
    && dnf clean all

# Install Python 3.12 and upgrade pip
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies
RUN pip install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio
RUN pip install optuna mysqlclient
RUN pip install pipdeptree && pipdeptree -p laser-polio > /app/laser_polio_deps.txt

# Copy your application and data files
COPY calib/logic.py .
COPY calib/calibrate.py .
COPY calib/calib_db.py .
#COPY calib/laser.py ./laser.py
COPY calib/setup_sim.py ./calib/setup_sim.py
COPY calib/cloud/check_study.sh .
COPY calib/calib_configs/calib_pars_r0.yaml ./calib/calib_configs/calib_pars_r0.yaml
COPY calib/model_configs/config_zamfara.yaml ./calib/model_configs/config_zamfara.yaml

# Make sure shell scripts are executable
RUN chmod a+x ./check_study.sh

# Copy data into the container
COPY data/init_immunity_0.5coverage_january.h5 /app/data/
COPY data/distance_matrix_africa_adm2.h5 /app/data/
COPY ./data/Nigeria_age_pyramid_2024.csv /app/data/
COPY ./data/sia_scenario_1.csv /app/data/
COPY ./data/compiled_cbr_pop_ri_sia_underwt_africa.csv /app/data/
COPY ./data/age_africa.csv /app/data/
COPY ./data/sia_historic_schedule.csv /app/data/
COPY ./data/shp_names_africa_adm2.csv /app/data/
COPY examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv /app/examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv
COPY examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_r14.csv /app/examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_r14.csv

ENV POLIO_ROOT=/app

# Clean up unnecessary dependencies and caches
RUN pip cache purge

# Run the script when the container starts
ENTRYPOINT ["python3", "calibrate.py"]
