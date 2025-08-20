FROM idm-docker-staging.packages.idmod.org/laser/laser-polio-base:latest

WORKDIR /app

# Set environment variables early
ENV POLIO_ROOT=/app
ENV NUMBA_CPU_NAME=generic
ENV HEADLESS=1


# Install laser-polio (use cache nullification hack suggested by robot)
# Add '--build-arg CACHE_BREAKER=$(date +%s)' to docker build line
ARG CACHE_BREAKER=1
RUN pip install --upgrade -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio

# Copy application code and configuration files
COPY calib/ ./calib/
COPY ./data/ /app/data/

# Ensure script permissions after code is copied
RUN chmod a+x calib/cloud/check_study.sh

# Analyze dependencies (after laser-polio is installed)
RUN pip3 install pipdeptree && pipdeptree -p laser-polio > /app/laser_polio_deps.txt

# Final cleanup to reduce image size
RUN pip3 cache purge

# Run a real, minimal simulation
RUN python3 -c "\
from laser_polio.run_sim import run_sim; \
run_sim(regions=['ZAMFARA'], start_year=2020, n_days=1, init_prev=0.001, r0=10, \
        background_seeding=False, use_pim_scalars=False, run=True, save_data=False, verbose=0)"

# Entrypoint
ENTRYPOINT ["python3", "calib/calibrate.py"]
