# /// script
# dependencies = [
#     "altair==5.4.1",
#     "marimo",
#     "pandas==2.2.3",
#     "polars==1.14.0",
#     "scikit-learn==1.5.2",
#     "skore==0.4.1",
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl 
    import altair as alt
    return alt, mo, pl


@app.cell
def _(pl):
    df_meteo = (
        pl.read_csv("https://raw.githubusercontent.com/koaning/solah/refs/heads/main/data/history.csv")
        .with_columns(
            date=pl.col("date").str.to_date(), 
        )
    )
    return (df_meteo,)


@app.cell
def _(pl):
    df_generated = (
        pl.read_csv("https://raw.githubusercontent.com/koaning/solah/refs/heads/main/data/generated.csv")
            .with_columns(
                date=pl.col("date").str.to_date(format="%m/%d/%Y"), 
                kWh=pl.col("kWh").str.replace(",", "").cast(pl.Int32)/1000
            )
    )
    return (df_generated,)


@app.cell
def _(df_meteo, mo):
    cols = [n for n in df_meteo.columns if n != "date"]

    radio_col = mo.ui.radio(options=cols, value="sunshine_duration")
    return cols, radio_col


@app.cell
def _(df_generated, df_meteo):
    df_merged = df_generated.join(df_meteo, left_on="date", right_on="date").drop_nulls()
    return (df_merged,)


@app.cell
def _(X, df_meteo, mo, models, radio_mod, y):
    models[radio_mod.value].fit(X, y)

    df_to_predict = df_meteo.drop_nulls()

    out = (
        df_to_predict
        .with_columns(pred=models[radio_mod.value].predict(df_to_predict.drop("date")))
        .tail(1)
        .to_dicts()
    )[0]

    mo.md(f"""
    ## Day ahead prediction

    For **{out['date']}** we seem to predict **{out['pred']:.1f} kWh** of energy production. 
    """)
    return df_to_predict, out


@app.cell
def _(df_merged, mo):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict

    models = {
        "ridge": Ridge(), 
        "histboost": HistGradientBoostingRegressor()
    }

    radio_mod = mo.ui.radio(options=list(models.keys()), value="ridge")

    y = df_merged["kWh"]
    X = df_merged.drop("date", "kWh")
    return (
        HistGradientBoostingRegressor,
        Ridge,
        X,
        cross_val_predict,
        models,
        radio_mod,
        y,
    )


@app.cell
def _(X, cross_val_predict, models, radio_mod, y):
    preds = cross_val_predict(models[radio_mod.value], X, y, cv=5)
    return (preds,)


@app.cell
def _(df_merged, mo, pl, preds, radio_mod):
    df_pred = df_merged.with_columns(preds=preds)

    mo.hstack([
        radio_mod,
        df_pred.plot.scatter("preds", "kWh").properties(title="predicted vs. actual"), 
        df_pred.with_columns(err=pl.col("preds") - pl.col("kWh")).plot.scatter("date", "err").properties(title="error over time")
    ])
    return (df_pred,)


@app.cell
def _(mo):
    mo.md("""
    ## Cross validated metrics

    If you are curious to explore the data more in depth, you can do so by exploring below.
    """)
    return


@app.cell
def _(df_merged):
    df_merged
    return


if __name__ == "__main__":
    app.run()
