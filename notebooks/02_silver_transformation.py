# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Cleaning, Standardisation & Enrichment
# MAGIC
# MAGIC Reads Bronze Delta tables → applies:
# MAGIC - Column renaming & type casting
# MAGIC - Fiscal year → calendar year parsing
# MAGIC - Null handling
# MAGIC - Crop-season labelling (India Kharif / Rabi)
# MAGIC - Unified `silver.global_tractor_sales` combining India + US

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md ## 1. India — Parse Fiscal Year & Filter Tractors Only

# COMMAND ----------

df_india_bronze = spark.read.table("bronze.india_tractor_sales")

df_india_silver = (
    df_india_bronze
    # Keep only tractors (exclude power tillers)
    .filter(F.col("vehicle_type") == "Tractors")
    # Parse "2023-24" → calendar year 2023 (start year of FY)
    .withColumn("calendar_year",
        F.split(F.col("fiscal_year"), "-").getItem(0).cast("int"))
    .withColumn("fiscal_year_label",
        F.concat(F.lit("FY"), F.substring(F.col("fiscal_year"), 3, 2),
                 F.lit("-"), F.substring(F.col("fiscal_year"), 6, 2)))
    # Value is in thousands — convert to absolute units
    .withColumn("units_sold_thousands", F.col("value").cast("int"))
    .withColumn("units_sold", (F.col("value") * 1000).cast("long"))
    .withColumn("country", F.lit("India"))
    .select(
        "country", "calendar_year", "fiscal_year_label",
        "units_sold", "units_sold_thousands", "notes"
    )
    .orderBy("calendar_year")
)

df_india_silver.write.format("delta").mode("overwrite").saveAsTable("silver.india_annual_sales")
print(f"silver.india_annual_sales: {df_india_silver.count()} rows")
df_india_silver.show(20)

# COMMAND ----------

# MAGIC %md ## 2. US — Standardise HP Categories & Add Year Totals

# COMMAND ----------

HP_MAP = {
    "< 40 HP":   "sub40HP",
    "40 < 100 HP": "mid40to100HP",
    "100+ HP":   "plus100HP",
}

df_us_bronze = spark.read.table("bronze.us_aem_sales")

df_us_silver = (
    df_us_bronze
    .withColumn("hp_std", F.trim(F.col("hp_category")))
    .replace(HP_MAP, subset=["hp_std"])
    .withColumn("country", F.lit("USA"))
    .withColumn("date",
        F.to_date(F.concat_ws("-", F.col("year"), F.col("month"), F.lit("01"))))
    .select(
        "country", "year", "month", "date",
        F.col("hp_std").alias("hp_category"),
        "drive_type", "units_sold"
    )
)

df_us_silver.write.format("delta").mode("overwrite").saveAsTable("silver.us_monthly_sales")
print(f"silver.us_monthly_sales: {df_us_silver.count()} rows")
df_us_silver.show(10)

# COMMAND ----------

# MAGIC %md ## 3. India Monthly Seasonality — Label Kharif / Rabi Seasons

# COMMAND ----------

# India fiscal year months: Apr(4)=month1 ... Mar(3)=month12
# Kharif sowing:  June–September  (months 6, 7, 8, 9)
# Rabi sowing:    October–February (months 10, 11, 12, 1, 2)
# Harvest/Slack:  March–May       (months 3, 4, 5)

def season_label(month_col):
    return (
        F.when(F.col(month_col).isin([6, 7, 8, 9]),    F.lit("Kharif (Jun–Sep)"))
         .when(F.col(month_col).isin([10, 11, 12, 1, 2]), F.lit("Rabi (Oct–Feb)"))
         .otherwise(F.lit("Harvest / Slack (Mar–May)"))
    )

# Estimated monthly proportions derived from TAMA historical patterns
# (Actual monthly data can be loaded from TAMA if available)
monthly_weights = [
    (4,  6.4),   # Apr
    (5,  9.0),   # May
    (6,  10.6),  # Jun
    (7,  12.6),  # Jul  ← Kharif peak
    (8,  9.6),   # Aug
    (9,  8.0),   # Sep
    (10, 6.2),   # Oct
    (11, 10.2),  # Nov  ← Rabi peak
    (12, 9.6),   # Dec
    (1,  7.9),   # Jan
    (2,  4.8),   # Feb
    (3,  5.1),   # Mar
]

schema_monthly = ["month_num", "pct_share"]
df_weights = spark.createDataFrame(monthly_weights, schema_monthly)

# Cross join with annual sales to derive monthly estimates
df_india_annual = spark.read.table("silver.india_annual_sales")

df_india_monthly = (
    df_india_annual
    .crossJoin(df_weights)
    .withColumn("estimated_units",
        (F.col("units_sold") * F.col("pct_share") / 100).cast("long"))
    .withColumn("season", season_label("month_num"))
    .withColumn("month_name",
        F.date_format(
            F.to_date(F.concat_ws("-", F.col("calendar_year"), F.col("month_num"), F.lit("1"))),
            "MMM"
        )
    )
    .select(
        "country", "calendar_year", "fiscal_year_label",
        "month_num", "month_name", "season",
        "estimated_units", "pct_share"
    )
    .orderBy("calendar_year", "month_num")
)

df_india_monthly.write.format("delta").mode("overwrite").saveAsTable("silver.india_monthly_sales")
print(f"silver.india_monthly_sales: {df_india_monthly.count()} rows")

# COMMAND ----------

# MAGIC %md ## 4. UN Comtrade — Clean & Standardise Trade Data

# COMMAND ----------

df_comtrade_bronze = spark.read.table("bronze.comtrade_exports")

# Select and rename key fields from Comtrade API response
df_comtrade_silver = (
    df_comtrade_bronze
    .select(
        F.col("period").cast("int").alias("year"),
        F.col("reporterDesc").alias("exporter_country"),
        F.col("reporterCode").alias("exporter_code"),
        F.col("partnerDesc").alias("partner_country"),
        F.col("primaryValue").cast("double").alias("trade_value_usd"),
        F.col("netWgt").cast("double").alias("net_weight_kg"),
        F.col("qty").cast("double").alias("quantity_units"),
        F.col("cmdCode").alias("hs_code"),
        F.col("flowCode").alias("flow_type"),
    )
    .filter(F.col("trade_value_usd").isNotNull())
    .filter(F.col("trade_value_usd") > 0)
)

df_comtrade_silver.write.format("delta").mode("overwrite").saveAsTable("silver.global_trade")
print(f"silver.global_trade: {df_comtrade_silver.count()} rows")

# COMMAND ----------

# MAGIC %md ## 5. FRED Price Index — Parse Dates & Remove Missing Values

# COMMAND ----------

df_fred_bronze = spark.read.table("bronze.fred_price_index")

df_fred_silver = (
    df_fred_bronze
    .withColumn("obs_date", F.to_date(F.col("date"), "yyyy-MM-dd"))
    .withColumn("year",     F.year("obs_date"))
    .withColumn("month",    F.month("obs_date"))
    .withColumn("index_val", F.col("index_value").cast("double"))
    .filter(F.col("index_val").isNotNull())
    .filter(F.col("index_val") > 0)
    .select("series_id", "series_name", "obs_date", "year", "month", "index_val")
    .orderBy("obs_date")
)

df_fred_silver.write.format("delta").mode("overwrite").saveAsTable("silver.price_index")
print(f"silver.price_index: {df_fred_silver.count()} rows")

# COMMAND ----------

# MAGIC %md ## 6. Verify Silver Tables

# COMMAND ----------

for tbl in ["india_annual_sales", "india_monthly_sales", "us_monthly_sales",
            "global_trade", "price_index"]:
    cnt = spark.sql(f"SELECT COUNT(*) AS n FROM silver.{tbl}").collect()[0]["n"]
    print(f"  silver.{tbl}: {cnt:,} rows")
