FROM idm-docker-staging.packages.idmod.org/laser/laser-polio-base:latest

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

# Get latest release tag and write it to lp_version
RUN curl -s https://api.github.com/repos/InstituteforDiseaseModeling/laser-polio/releases/latest \
    | sed -n 's/.*"tag_name": "\(.*\)",/\1/p' > lp_version
# Use the downloaded version to install the corresponding package
RUN pip install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio==$(cat lp_version)

# RUN curl -s https://api.github.com/repos/InstituteforDiseaseModeling/laser-polio/releases/latest \
#     | sed -n 's/.*"tag_name": "\(.*\)",/\1/p' > lp_version
# RUN pip install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio

# Final cleanup
RUN pip3 cache purge

# Entrypoint
ENTRYPOINT ["python3", "calibrate.py"]
