# Use Rocky Linux as base
FROM rockylinux:9

#RUN dnf -y update
#RUN dnf install -y https://dev.mysql.com/get/mysql80-community-release-el9-1.noarch.rpm
#RUN dnf install -y gcc gcc-g++ python3-devel mysql-community-devel

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
    git \
    && dnf clean all

# Install Python 3.12 and dependencies
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy your script into the container
RUN pip install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio
RUN pip install optuna mysqlclient
COPY calib/logic.py .
COPY calib/calibrate.py .
COPY calib/calib_db.py .
COPY calib/demo_zamfara.py ./calib/demo_zamfara.py
COPY calib/cloud/check_study.sh .
COPY calib/calib_params/calib_better.yaml ./calib_pars.yaml
COPY calib/configs/config_base_wakanda.yaml ./config.yaml

RUN chmod a+x ./check_study.sh

# Copy additional files into the data directory inside the container
COPY data/init_immunity_0.5coverage_january.h5 /app/data/
COPY data/distance_matrix_africa_adm2.h5 /app/data/
COPY ./data/Nigeria_age_pyramid_2024.csv /app/data/
COPY ./data/sia_scenario_1.csv /app/data/
COPY ./data/compiled_cbr_pop_ri_sia_underwt_africa.csv /app/data/
COPY ./data/age_africa.csv /app/data/
COPY ./data/sia_historic_schedule.csv /app/data/
COPY ./data/shp_names_africa_adm2.csv /app/data/
COPY examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv /app/examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv

ENV POLIO_ROOT=/app

# Run the script when the container starts
#CMD ["python3", "laser.py", "-p", "shared/params.json", "-o", "shared/simulation_results.csv"]
ENTRYPOINT ["python3", "calibrate.py"]
