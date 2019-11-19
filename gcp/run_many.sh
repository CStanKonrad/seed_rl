#!/bin/bash

set -e

if [ "$#" -ne "5" ]
then
    echo "JOB_NAME_PREFIX BUCKET_NAME AI_PLATFORM_CONFIG_FILE ENV_CONFIG_FILE NUM_RUNS"
    exit 1
fi

JOB_NAME_PREFIX="$1"
BUCKET_NAME="$2"
AI_PLATFORM_CONFIG_FILE="$3"
ENV_CONFIG_FILE="$4"
NUM_RUNS=$5

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for((i=0;i<NUM_RUNS;i++))
do
    echo "Running $i"
    $DIR/run_exp.sh "${JOB_NAME_PREFIX}_${i}" "$BUCKET_NAME" "$AI_PLATFORM_CONFIG_FILE" "$ENV_CONFIG_FILE" &
done


