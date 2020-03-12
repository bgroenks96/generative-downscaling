#!/bin/sh

# BASELINE - BCSD
# mlflow run . -e bcsd --experiment-name bcsd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=southeast_us -P var=MAXT --no-conda
# mlflow run . -e bcsd --experiment-name bcsd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=pacific_nw -P var=MAXT --no-conda
# mlflow run . -e bcsd --experiment-name bcsd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=southeast_us -P var=PRCP --no-conda
# mlflow run . -e bcsd --experiment-name bcsd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=pacific_nw -P var=PRCP --no-conda

# BASELINE - Bano-Medina CNN-10
mlflow run . -e bmd --experiment-name bmd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=southeast_us -P var=MAXT --no-conda
mlflow run . -e bmd --experiment-name bmd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=pacific_nw -P var=MAXT --no-conda
mlflow run . -e bmd --experiment-name bmd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=southeast_us -P var=PRCP --no-conda
mlflow run . -e bmd --experiment-name bmd-final -P data_lr=erai/daily-1deg -P scale=4 -P region=pacific_nw -P var=PRCP --no-conda

# Glow-JFLVM
mlflow run . -e glow-jflvm --experiment-name glow-jflvm-final -P data_lr=erai/daily-1deg -P scale=4 -P region=southeast_us -P var=MAXT -P layers=3 -P depth=8 -P mode=test \
    -P epochs=50 --no-conda
mlflow run . -e glow-jflvm --experiment-name glow-jflvm-final -P data_lr=erai/daily-1deg -P scale=4 -P region=pacific_nw -P var=MAXT -P layers=3 -P depth=8 -P mode=test \
    -P epochs=50 --no-conda
mlflow run . -e glow-jflvm --experiment-name glow-jflvm-final -P data_lr=erai/daily-1deg -P scale=4 -P region=southeast_us -P var=PRCP -P layers=3 -P depth=4 -P mode=test \
    -P epochs=50 --no-conda
mlflow run . -e glow-jflvm --experiment-name glow-jflvm-final -P data_lr=erai/daily-1deg -P scale=4 -P region=pacific_nw -P var=PRCP -P layers=3 -P depth=4 -P mode=test \
    -P epochs=50 --no-conda
