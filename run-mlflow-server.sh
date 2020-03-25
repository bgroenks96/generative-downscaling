#!/bin/sh

pip install google-cloud-storage
pip install --upgrade mlflow
export GOOGLE_APPLICATION_CREDENTIALS="/home/jovyan/generative-downscaling/gcs.secret.json"
mlflow server \
    --backend-store-uri "./mlruns" \
    --default-artifact-root "gs://generative-downscaling-artifact-store/artifacts" \
    --host 0.0.0.0
