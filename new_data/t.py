import pandas as pd

db = pd.read_csv("LNPDB_vitro_del_processed.csv")

new = db[db["Experiment_value"].isna()][["Experiment_ID"]]

new.to_csv("new.csv", index=False)
