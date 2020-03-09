import mlflow
import shutil

deleted_runs = mlflow.search_runs(experiment_ids=['0','1','2','3','4'], run_view_type=mlflow.entities.ViewType.DELETED_ONLY)
for i, run in deleted_runs.iterrows():
    run_id, exp_id = run['run_id'], run['experiment_id']
    print(f'Deleting run {run_id} in experiment {exp_id}')
    shutil.rmtree(f'./mlruns/{exp_id}/{run_id}')
