import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from prefect import flow,task, task_runners


# Specify Public URL of EC2 instance where the MLflow tracking server is running
TRACKING_URI = r"file://C:/DFKI/Rwanda_GWP/code/mlops/mlruns"

mlflow.set_tracking_uri(TRACKING_URI) 
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")

#specify name of experiment (will be created if it does not exist)
mlflow.set_experiment("mlflow-exp")

@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default=r"C:\DFKI\Rwanda_GWP\code\mlops\data",
    help="Location where the processed NYC taxi trip data was saved"
)
@flow(task_runner=task_runners.SequentialTaskRunner(), name="training_flow")
def run_train(data_path: str):
    mlflow.sklearn.autolog()
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse",rmse)
        mlflow.sklearn.log_model(rf,artifact_path="sklearn_model")

if __name__ == '__main__':
    run_train()
