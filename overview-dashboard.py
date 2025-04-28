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
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo, out, p1, p2, p3, score_vals, sliders):
    mo.vstack([
        mo.hstack([
            mo.stat(value=out['date'], label="Date", bordered=True),
            mo.stat(value=f"{out['pred']:.1f} kWh ± {(score_vals[-1] - score_vals[0])/2:.2f}", label="Predicted power output", bordered=True),
            sliders,
        ], widths="equal", gap=1),
        mo.hstack([
            mo.vstack([mo.md("## kWh over time"), p1]), 
            mo.vstack([mo.md("## General errors"), p2]),
            mo.vstack([mo.md("## Local error distribution"), p3]),
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
    return alt, pl


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
def _(mo):
    window_slider = mo.ui.slider(7, 31, 1, label="Window width", value=14)
    err_slider = mo.ui.slider(0.1, 3, 0.01, label="Error smoothing", value=0.2)

    sliders = mo.md("""
    {window_slider}

    {err_slider}
    """).batch(window_slider=window_slider, err_slider=err_slider)
    return err_slider, sliders, window_slider


@app.cell(hide_code=True)
def _(df_density, df_err, df_generated, df_quantiles, mo, sliders):
    from pyobsplot import Plot

    p1 = Plot.plot({
        "grid": True,
        "marks": [
            Plot.dot(df_generated,{"x": "date", "y": "kWh", "opacity": 0.4}),
            Plot.lineY(df_generated, 
               Plot.windowY(
                   {"k": sliders.value["window_slider"]}, 
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
            Plot.dot(df_err, {"x": "preds", "y": "kWh", "opacity": "dist"}),
            Plot.density(df_err, {"x": "preds", "y": "kWh", "stroke": "gray", "opacity": 0.2}),
        ],
        "height": 500, 
        "style": {
            "background-color": "#181C1A"
        } if mo.app_meta().theme == "dark" else {}
    }, theme=mo.app_meta().theme)

    p3 = Plot.plot({
        "grid": True,
        "marks": [
            Plot.areaY(df_density, {"x": "preds", "y": "score", "fillOpacity": 0.3}),
            Plot.lineY(df_density, {"x": "preds", "y": "score"}),
            Plot.ruleX(df_quantiles, {"x": "value", "strokeOpacity": 0.8, "strokeDashOffset": "100"}),
            Plot.text(df_quantiles, dict(
                  x="value",
                  text="name",
                  href="href",
                  target="_blank",
                  rotate=-90,
                  dx=-3,
                  frameAnchor="top-right",
                  lineAnchor="bottom",
                  fontVariant="tabular-nums",
                  fill="currentColor",
                ))
        ],
        "height": 500, 
        "style": {
            "background-color": "#181C1A"
        } if mo.app_meta().theme == "dark" else {}
    }, theme=mo.app_meta().theme)
    return Plot, p1, p2, p3


@app.cell(hide_code=True)
def _(X, df_meteo, models, radio_mod, y):
    models[radio_mod.value].fit(X, y)

    df_to_predict = df_meteo.drop_nulls()

    df_with_pred = (
        df_to_predict
        .with_columns(
            pred=models["ridge"].predict(df_to_predict.drop("date")),
        )
    )

    out = df_with_pred.tail(1).to_dicts()[0]
    return df_to_predict, df_with_pred, out


@app.cell(hide_code=True)
def _(df_merged, mo):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    models = {
        "ridge": Ridge(), 
    }

    radio_mod = mo.ui.radio(options=list(models.keys()), value="ridge")

    y = df_merged["kWh"]
    X = df_merged.drop("date", "kWh")
    return Ridge, X, cross_val_predict, models, radio_mod, y


@app.cell(hide_code=True)
def _(X, cross_val_predict, models, radio_mod, y):
    preds = cross_val_predict(models[radio_mod.value], X, y, cv=5)
    return (preds,)


@app.cell(hide_code=True)
def _(df_err, np, pl, xs):
    from sklearn.neighbors import KernelDensity

    kernel = KernelDensity().fit(
        np.array(df_err["preds"]).reshape(-1, 1), 
        sample_weight=df_err["dist"]
    )

    samples = np.exp(kernel.score_samples(xs.reshape(-1, 1)))
    sorted_samples = samples.cumsum()
    sorted_samples /= sorted_samples.max()

    df_density = pl.DataFrame({
        "score": samples,
        "preds": xs
    })

    quantiles = [0.1, 0.9]
    quan_vals = [np.argmin((sorted_samples - q)**2) for q in quantiles]
    score_vals = [xs[q] for q in quan_vals]

    df_quantiles = pl.DataFrame([
        {"quantile": q, "value": qv, "name": f"{q * 100:.0f}% quantile"} for q, qv in zip(quantiles, score_vals)
    ])
    return (
        KernelDensity,
        df_density,
        df_quantiles,
        kernel,
        quan_vals,
        quantiles,
        samples,
        score_vals,
        sorted_samples,
    )


@app.cell(hide_code=True)
def _(df_pred, out, pl, sliders):
    import numpy as np

    df_err = df_pred.with_columns(
        err=pl.col("preds") - pl.col("kWh"), 
        dist=np.exp(-((pl.lit(out["pred"]) - pl.col("preds"))/sliders.value["err_slider"])**2)
    )

    weighted_mean = np.average(df_err["preds"], weights=df_err["dist"])
    weighted_std = np.sqrt(np.average((weighted_mean - df_err["preds"])**2, weights=df_err["dist"]))

    xs = np.linspace(0, 23, 200)
    return df_err, np, weighted_mean, weighted_std, xs


@app.cell(hide_code=True)
def _(df_merged, preds):
    df_pred = df_merged.with_columns(preds=preds)
    return (df_pred,)


if __name__ == "__main__":
    app.run()
