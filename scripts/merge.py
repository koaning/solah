import pandas as pd 

# Good grief this stuff is nasty. Should've moved to polars. 
df_hist = pd.read_csv("data/history.csv").assign(date=lambda d: pd.to_datetime(pd.to_datetime(d['date']).dt.strftime("%Y-%m-%d")))
df_annot = pd.read_csv("data/annots.tsv", sep="\t")

df_annot_clean = (
    df_annot
      .assign(date=lambda d: pd.to_datetime("2024-04-01") + pd.to_timedelta(d['date'] - 45352, unit='D'))
      .assign(date=lambda d: pd.to_datetime(pd.to_datetime(d['date']).dt.strftime("%Y-%m-%d")))
      .assign(taken=lambda d: d['levering_normal_tarif'] + d['levering_low_tarif'],
              supplied=lambda d: d['teruglevering_normal_tarif'] + d['teruglevering_low_tarif'])
      [['date', 'taken', 'supplied']]
)

df_hist.merge(df_annot_clean, on='date', how='left').to_csv("data/merged.csv", index=False)