# Start with Rocky Linux 9
FROM rockylinux:9.3

# Enable IUS and EPEL
RUN dnf install -y epel-release && \
    dnf update -y

# Install Python 3.12 and tools from IUS
RUN dnf install -y \
    python3.12 \
    python3.12-devel \
    python3.12-pip \
    mariadb-connector-c-devel \
    gcc \
    gcc-c++ \
    make \
    wget \
    screen \
    procps \
    vim \
    git && \
    dnf clean all

# Set default python3 and pip3 to use 3.10 explicitly
RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.12 100

# Confirm
RUN python3 --version && pip3 --version

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --upgrade pip
# Put latest laser-polio wheel link below
RUN dnf install -y git
RUN pip3 install https://github.com/InstituteforDiseaseModeling/laser-polio/releases/download/v0.1.2/laser_polio-0.1.2-py3-none-any.whl
# RUN pip3 install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio
RUN pip3 install optuna mysqlclient
RUN pip3 install pipdeptree && pipdeptree -p laser-polio > /app/laser_polio_deps.txt

# Copy your application and data files (same as before)
COPY calib/ .
COPY calib/calib_configs/ ./calib/calib_configs/
COPY calib/model_configs/ ./calib/model_configs/
RUN chmod a+x cloud/check_study.sh
COPY ./data/ /app/data/

ENV POLIO_ROOT=/app
ENV NUMBA_CPU_NAME=generic
ENV HEADLESS=1

# Final cleanup
RUN pip3 cache purge

# Entrypoint
ENTRYPOINT ["python3", "calibrate.py"]
