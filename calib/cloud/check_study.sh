#!/bin/sh
STUDY_NAME=$1

optuna studies --storage "mysql://${MYSQL_USER}:${MYSQL_PASSWORD}@mysql:3306/${MYSQL_DB}" | grep "${STUDY_NAME}" > /dev/null
echo $?
