import pandas as pd

# Jan 2023 — ~3M Uber/Lyft rides
url_jan = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet"
url_feb = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-02.parquet"

df_jan = pd.read_parquet(url_jan)
df_feb = pd.read_parquet(url_feb)
df = pd.concat([df_jan, df_feb], ignore_index=True)

# Download taxi zone lookup (zone IDs → neighborhood names)
zones = pd.read_csv("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv")
print(df.shape, df.columns.tolist())