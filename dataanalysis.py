import pandas as pd

data=pd.read_csv(r"Data/metadata.csv")
print(data.head())

print(data.isna().sum())