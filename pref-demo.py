import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from prefect.logging import get_run_logger
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact


@flow(log_prints=True)
def run_full_pipeline():
    """Flow: do just a little bit of ML."""
    df_meteo = read_meteo()
    df_panel = read_panels()
    df_pred = merge_meteo_panel(df_meteo, df_panel).pipe(do_ml_thing)
    show_chart(df_pred)    


@task
def read_meteo():
    """Task 1: Fetch the statistics for a GitHub repo"""
    return (
        pl.read_csv("data/history.csv")
            .with_columns(
                date=pl.col("date").str.to_date(), 
            )
        )


@task
def read_panels():
    return (
        pl.read_csv("data/generated.csv")
            .with_columns(
                date=pl.col("date").str.to_date(format="%m/%d/%Y"), 
                kWh=pl.col("kWh").str.replace(",", "").cast(pl.Int32)/1000
            )
    )

@task
def merge_meteo_panel(df_meteo, df_panel):
    return df_panel.join(df_meteo, left_on="date", right_on="date").drop_nulls()


@task
def do_ml_thing(df_merged):
    y = df_merged["kWh"]
    X = df_merged.drop("date", "kWh")

    logger = get_run_logger()
    logger.info(f"{X.shape=}")
    logger.info(f"{y.shape=}")

    preds = cross_val_predict(HistGradientBoostingRegressor(), X, y, cv=10)

    return df_merged.with_columns(preds=preds)


@task
def show_chart(df_pred):
    chart = df_pred.plot.scatter("preds", "kWh").properties(title="predicted vs. actual")
    
    error = float(np.mean(np.abs(df_pred["kWh"].to_numpy() - df_pred["preds"].to_numpy())))

    create_markdown_artifact(
        key="chart",
        markdown=f"""
# Great markdown report

I would personally prefer good old HTML.

## Stats

But this is the mean absolute error. 

{error}
        """, 
        description="Pretty chart"
    )

# Run the flow
if __name__ == "__main__":
    run_full_pipeline()