FROM idm-docker-staging.packages.idmod.org/laser/laser-polio-base:latest

# Copy your application and data files (same as before)
COPY calib/ .
COPY calib/calib_configs/ ./calib/calib_configs/
COPY calib/model_configs/ ./calib/model_configs/
RUN chmod a+x cloud/check_study.sh
COPY ./data/ /app/data/

ENV POLIO_ROOT=/app
ENV NUMBA_CPU_NAME=generic
ENV HEADLESS=1

# Use the downloaded version to install the corresponding package
RUN pip3 install -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio

# Final cleanup
RUN pip3 cache purge

RUN pip3 install pipdeptree && pipdeptree -p laser-polio > /app/laser_polio_deps.txt

# Entrypoint
ENTRYPOINT ["python3", "calibrate.py"]
