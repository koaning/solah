# /// script
# dependencies = [
#     "altair==5.4.1",
#     "marimo[lsp]",
#     "numpy==2.2.5",
#     "pandas==2.2.3",
#     "polars==1.14.0",
#     "pyobsplot==0.5.3.2",
#     "scikit-learn==1.5.2",
#     "pyarrow==18.1.0",
#     "srsly==2.5.1",
#     "requests==2.32.3",
# ]
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(df_merged, mo, p1, p2):
    stats = df_merged.tail(1).to_dicts()[0]

    mo.vstack([
        mo.hstack([
            mo.stat(value=stats["date"], label="Date", bordered=True),
            mo.stat(value=f"{stats['pred']:.1f} kWh", label="Predicted power output", bordered=True),
        ], widths="equal", gap=1),
        mo.hstack([
            mo.vstack([mo.md("## kWh over time"), p1]), 
            mo.vstack([mo.md("## General errors"), p2]),
        ])
    ])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl 
    import altair as alt
    import datetime as dt 
    return (pl,)


@app.cell
def _(pl):
    df_pred = (
        pl.read_csv("https://raw.githubusercontent.com/koaning/solah/refs/heads/main/data/merged.csv")
        .with_columns(
            date=pl.col("date").str.to_date(), 
        )
    )
    return (df_pred,)


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
def _(df_generated, df_pred, pl):
    df_merged = (
        df_pred.join(df_generated, left_on="date", right_on="date").with_columns(err=pl.col("kWh") - pl.col("pred"))
    )
    return (df_merged,)


@app.cell
def _(mo):
    window_slider = mo.ui.slider(7, 31, 1, label="Window width", value=14)
    window_slider
    return (window_slider,)


@app.cell(hide_code=True)
def _(df_merged, mo, window_slider):
    from pyobsplot import Plot

    p1 = Plot.plot({
        "grid": True,
        "marks": [
            Plot.dot(df_merged,{"x": "date", "y": "kWh", "opacity": 0.4}),
            Plot.lineY(df_merged, 
               Plot.windowY(
                   {"k": window_slider.value}, 
                   {"x": "date", "y": "kWh", "stroke": "steelblue", "strokeWidth": 3}
            ))
        ],
        "height": 500, 
        "style": {
            "background-color": "#181C1A"
        } if mo.app_meta().theme == "dark" else {}
    }, theme=mo.app_meta().theme)

    p2 = Plot.plot({
        "grid": True,
        "marks": [
            Plot.dot(df_merged, {"x": "pred", "y": "kWh", "opacity": 0.3}),
            Plot.density(df_merged, {"x": "pred", "y": "kWh", "stroke": "gray", "opacity": 0.2}),
        ],
        "height": 500, 
        "style": {
            "background-color": "#181C1A"
        } if mo.app_meta().theme == "dark" else {}
    }, theme=mo.app_meta().theme)
    return p1, p2


if __name__ == "__main__":
    app.run()
