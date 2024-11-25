# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.4.1",
#     "marimo",
#     "polars==1.14.0",
#     "scikit-learn==1.5.2",
# ]
# ///

import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import polars as pl 
    import altair as alt
    return alt, mo, pl


@app.cell
def __(pl):
    df_meteo = (
        pl.read_csv("data/history.csv")
        .with_columns(
            date=pl.col("date").str.to_date(), 
        )
    )

    df_meteo.plot.line("date", "sunshine_duration")
    return (df_meteo,)


@app.cell
def __(pl):
    df_generated = (
        pl.read_csv("data/generated.csv")
            .with_columns(
                date=pl.col("date").str.to_date(format="%m/%d/%Y"), 
                kWh=pl.col("kWh").str.replace(",", "").cast(pl.Int32)/1000
            )
    )

    df_generated.plot.line("date", "kWh")
    return (df_generated,)


@app.cell
def __(df_meteo, mo):
    cols = [n for n in df_meteo.columns if n != "date"]

    radio_col = mo.ui.radio(options=cols, value="sunshine_duration")
    return cols, radio_col


@app.cell
def __(df_generated, df_meteo):
    df_merged = df_generated.join(df_meteo, left_on="date", right_on="date").drop_nulls()
    return (df_merged,)


@app.cell
def __(df_merged, mo, radio_col):
    mo.hstack([radio_col, df_merged.plot.scatter("date", radio_col.value), df_merged.plot.scatter(radio_col.value, "kWh")])
    return


@app.cell
def __(df_merged):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor

    y = df_merged["kWh"]
    X = df_merged.drop("date", "kWh")

    preds = Ridge().fit(X, y).predict(X)
    return HistGradientBoostingRegressor, Ridge, X, preds, y


@app.cell
def __(df_merged, preds):
    df_merged.with_columns(preds=preds).plot.scatter("preds", "kWh")
    return


@app.cell
def __(df_merged):
    df_merged.write_csv("data/merged.csv")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
