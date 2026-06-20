# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Data Ingestion
# MAGIC
# MAGIC Ingests raw CSV/API data into Delta tables under the `bronze` schema.
# MAGIC Sources:
# MAGIC - India tractor sales (Dataful.in / Ministry of Agriculture)
# MAGIC - US AEM monthly retail sales
# MAGIC - UN Comtrade global trade (HS code 8701)
# MAGIC - FRED agricultural machinery price index

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, LongType
)
import requests
import pandas as pd

spark = SparkSession.builder.appName("TractorIndustry-Bronze").getOrCreate()

# COMMAND ----------

# MAGIC %md ## 1. Create Schemas

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze COMMENT 'Raw ingested data — no transformations'")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver COMMENT 'Cleaned and enriched data'")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold   COMMENT 'Business KPIs and aggregations for BI'")

print("Schemas ready: bronze / silver / gold")

# COMMAND ----------

# MAGIC %md ## 2. India Tractor Sales — Dataful.in (Ministry of Agriculture)
# MAGIC
# MAGIC Download CSV from: https://dataful.in/datasets/3412/
# MAGIC Upload to DBFS: /FileStore/tractor/raw/india_tractor_sales.csv

# COMMAND ----------

INDIA_SCHEMA = StructType([
    StructField("fiscal_year",   StringType(),  True),
    StructField("state",         StringType(),  True),
    StructField("vehicle_type",  StringType(),  True),
    StructField("value",         IntegerType(), True),
    StructField("units",         StringType(),  True),
    StructField("notes",         StringType(),  True),
])

df_india_raw = (
    spark.read
    .option("header", "true")
    .schema(INDIA_SCHEMA)
    .csv("dbfs:/FileStore/tractor/raw/india_tractor_sales.csv")
)

df_india_raw.write.format("delta").mode("overwrite").saveAsTable("bronze.india_tractor_sales")
print(f"bronze.india_tractor_sales: {df_india_raw.count()} rows")
df_india_raw.show(5)

# COMMAND ----------

# MAGIC %md ## 3. US AEM Sales — Manually extracted from AEM monthly PDFs
# MAGIC
# MAGIC Source: https://www.aem.org/market-share-statistics/us-ag-tractor-and-combine-reports
# MAGIC Data manually extracted into CSV with columns:
# MAGIC   year, month, hp_category, drive_type, units_sold

# COMMAND ----------

US_AEM_SCHEMA = StructType([
    StructField("year",         IntegerType(), True),
    StructField("month",        IntegerType(), True),
    StructField("hp_category",  StringType(),  True),
    StructField("drive_type",   StringType(),  True),
    StructField("units_sold",   IntegerType(), True),
])

df_us_raw = (
    spark.read
    .option("header", "true")
    .schema(US_AEM_SCHEMA)
    .csv("dbfs:/FileStore/tractor/raw/us_aem_sales.csv")
)

df_us_raw.write.format("delta").mode("overwrite").saveAsTable("bronze.us_aem_sales")
print(f"bronze.us_aem_sales: {df_us_raw.count()} rows")
df_us_raw.show(5)

# COMMAND ----------

# MAGIC %md ## 4. UN Comtrade — Global Trade Data (HS 8701)
# MAGIC
# MAGIC Free API: https://comtradeapi.un.org/
# MAGIC Register for a free API key at the link above.

# COMMAND ----------

COMTRADE_API_KEY = dbutils.secrets.get(scope="tractor", key="comtrade_api_key")

def fetch_comtrade(year: int, flow_code: str = "X") -> pd.DataFrame:
    """
    Fetch tractor export (X) or import (M) data for a given year.
    HS code 8701 = Tractors (other than pedestrian-controlled tractors).
    """
    url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
    params = {
        "reporterCode": "",
        "period":       year,
        "flowCode":     flow_code,
        "cmdCode":      "8701",
        "subscription-key": COMTRADE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return pd.DataFrame(data)

frames = []
for yr in range(2015, 2025):
    try:
        frames.append(fetch_comtrade(yr, "X"))
        print(f"  Fetched exports {yr}")
    except Exception as e:
        print(f"  Skipped {yr}: {e}")

df_comtrade_pd = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

if not df_comtrade_pd.empty:
    df_comtrade = spark.createDataFrame(df_comtrade_pd)
    df_comtrade.write.format("delta").mode("overwrite").saveAsTable("bronze.comtrade_exports")
    print(f"bronze.comtrade_exports: {df_comtrade.count()} rows")

# COMMAND ----------

# MAGIC %md ## 5. FRED API — Agricultural Machinery Price Index (WPU114)

# COMMAND ----------

FRED_API_KEY = dbutils.secrets.get(scope="tractor", key="fred_api_key")

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":       series_id,
        "api_key":         FRED_API_KEY,
        "file_type":       "json",
        "observation_start": "2000-01-01",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return pd.DataFrame(obs)[["date", "value"]].rename(columns={"value": "index_value"})

df_fred_pd = fetch_fred_series("WPU114")   # Farm machinery & equipment PPI
df_fred_pd["series_id"] = "WPU114"
df_fred_pd["series_name"] = "Farm Machinery & Equipment PPI"

df_fred = spark.createDataFrame(df_fred_pd)
df_fred.write.format("delta").mode("overwrite").saveAsTable("bronze.fred_price_index")
print(f"bronze.fred_price_index: {df_fred.count()} rows")
df_fred.show(5)

# COMMAND ----------

# MAGIC %md ## 6. Verify All Bronze Tables

# COMMAND ----------

tables = spark.sql("SHOW TABLES IN bronze").collect()
for t in tables:
    cnt = spark.sql(f"SELECT COUNT(*) AS n FROM bronze.{t.tableName}").collect()[0]["n"]
    print(f"  bronze.{t.tableName}: {cnt:,} rows")
