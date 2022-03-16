
# setup environment
module purge
module load StdEnv/2020
module load python/3.8.12
export PYTHONUNBUFFERED=1
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt