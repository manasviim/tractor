# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Business KPIs & Aggregations
# MAGIC
# MAGIC Produces analytics-ready tables consumed by Looker and the Vercel web app.
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |---|---|
# MAGIC | gold.market_summary | Global volume by year, top countries |
# MAGIC | gold.india_seasonality | Monthly sales avg with season labels |
# MAGIC | gold.hp_segment_trend | US sales split by HP category & year |
# MAGIC | gold.price_vs_sales | Equipment PPI vs US unit sales (correlation) |
# MAGIC | gold.top_exporters | Top 15 countries by export value (latest year) |
# MAGIC | gold.supply_chain_flow | Manufacturer → Dealer → Farmer value-chain metrics |

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md ## 1. Market Summary — Global & Key Countries

# COMMAND ----------

# India annual
df_india = (
    spark.read.table("silver.india_annual_sales")
    .select(
        F.col("calendar_year").alias("year"),
        F.col("country"),
        F.col("units_sold"),
        F.lit(None).cast("double").alias("market_value_usd"),
    )
)

# US annual aggregation
df_us_annual = (
    spark.read.table("silver.us_monthly_sales")
    .groupBy("country", "year")
    .agg(F.sum("units_sold").alias("units_sold"))
    .withColumn("market_value_usd", F.lit(None).cast("double"))
)

# Combine
df_market = df_india.unionByName(df_us_annual)

# YoY growth
w = Window.partitionBy("country").orderBy("year")
df_market_summary = (
    df_market
    .withColumn("prev_units", F.lag("units_sold", 1).over(w))
    .withColumn("yoy_pct",
        F.round(
            (F.col("units_sold") - F.col("prev_units")) / F.col("prev_units") * 100,
            2
        )
    )
    .withColumn("units_sold_thousands",
        F.round(F.col("units_sold") / 1000, 1))
    .orderBy("country", "year")
)

df_market_summary.write.format("delta").mode("overwrite").saveAsTable("gold.market_summary")
print(f"gold.market_summary: {df_market_summary.count()} rows")
df_market_summary.show(20)

# COMMAND ----------

# MAGIC %md ## 2. India Seasonality — Average Monthly Share Across All Years

# COMMAND ----------

df_india_monthly = spark.read.table("silver.india_monthly_sales")

df_seasonality = (
    df_india_monthly
    .groupBy("month_num", "month_name", "season")
    .agg(
        F.round(F.avg("pct_share"), 2).alias("avg_pct_share"),
        F.round(F.avg("estimated_units") / 1000, 1).alias("avg_units_thousands"),
        F.count("calendar_year").alias("years_of_data"),
    )
    .orderBy("month_num")
)

# Most recent year detail
df_latest_monthly = (
    df_india_monthly
    .filter(F.col("calendar_year") == F.lit(2023))  # FY2023-24
    .select(
        "month_num", "month_name", "season",
        F.round(F.col("estimated_units") / 1000, 1).alias("fy2324_units_thousands"),
        "pct_share"
    )
)

df_seasonality_full = df_seasonality.join(df_latest_monthly, on=["month_num", "month_name", "season"], how="left")
df_seasonality_full.write.format("delta").mode("overwrite").saveAsTable("gold.india_seasonality")
print(f"gold.india_seasonality: {df_seasonality_full.count()} rows")
df_seasonality_full.show()

# COMMAND ----------

# MAGIC %md ## 3. HP Segment Trend — US Market by Horsepower Category

# COMMAND ----------

df_us = spark.read.table("silver.us_monthly_sales")

df_hp_trend = (
    df_us
    .groupBy("year", "hp_category", "country")
    .agg(F.sum("units_sold").alias("units_sold"))
    .withColumn("segment_label",
        F.when(F.col("hp_category") == "sub40HP",         F.lit("< 40 HP"))
         .when(F.col("hp_category") == "mid40to100HP",    F.lit("40–100 HP"))
         .when(F.col("hp_category") == "plus100HP",       F.lit("100+ HP"))
         .otherwise(F.col("hp_category"))
    )
    .orderBy("year", "hp_category")
)

# Add share within year
w_year = Window.partitionBy("year")
df_hp_trend = df_hp_trend.withColumn(
    "year_total", F.sum("units_sold").over(w_year)
).withColumn(
    "pct_of_year", F.round(F.col("units_sold") / F.col("year_total") * 100, 1)
)

df_hp_trend.write.format("delta").mode("overwrite").saveAsTable("gold.hp_segment_trend")
print(f"gold.hp_segment_trend: {df_hp_trend.count()} rows")
df_hp_trend.show(20)

# COMMAND ----------

# MAGIC %md ## 4. Price vs Sales Correlation (US Market)

# COMMAND ----------

df_price = (
    spark.read.table("silver.price_index")
    .filter(F.col("month") == 1)  # Jan observation as annual proxy
    .select("year", F.round("index_val", 2).alias("ppi_value"))
)

df_us_annual_total = (
    spark.read.table("silver.us_monthly_sales")
    .groupBy("year")
    .agg(F.sum("units_sold").alias("total_units"))
)

df_price_sales = (
    df_price
    .join(df_us_annual_total, on="year", how="inner")
    .withColumn("units_thousands", F.round(F.col("total_units") / 1000, 1))
    .orderBy("year")
)

df_price_sales.write.format("delta").mode("overwrite").saveAsTable("gold.price_vs_sales")
print(f"gold.price_vs_sales: {df_price_sales.count()} rows")
df_price_sales.show()

# Compute Pearson correlation
corr = df_price_sales.stat.corr("ppi_value", "total_units")
print(f"\nPearson correlation (PPI vs Units Sold): {corr:.4f}")
spark.createDataFrame(
    [{"metric": "pearson_corr_ppi_vs_units", "value": round(corr, 4)}]
).write.format("delta").mode("overwrite").saveAsTable("gold.price_correlation_summary")

# COMMAND ----------

# MAGIC %md ## 5. Top Exporters — Latest Year from UN Comtrade

# COMMAND ----------

df_trade = spark.read.table("silver.global_trade")

latest_year = df_trade.agg(F.max("year").alias("max_year")).collect()[0]["max_year"]

df_top_exporters = (
    df_trade
    .filter(F.col("year") == latest_year)
    .filter(F.col("flow_type") == "X")
    .groupBy("exporter_country", "exporter_code")
    .agg(
        F.round(F.sum("trade_value_usd") / 1e9, 3).alias("export_value_usd_bn"),
        F.sum("quantity_units").alias("total_units"),
    )
    .orderBy(F.desc("export_value_usd_bn"))
    .limit(15)
    .withColumn("rank", F.monotonically_increasing_id() + 1)
    .withColumn("data_year", F.lit(latest_year))
)

df_top_exporters.write.format("delta").mode("overwrite").saveAsTable("gold.top_exporters")
print(f"gold.top_exporters: {df_top_exporters.count()} rows (year={latest_year})")
df_top_exporters.show()

# COMMAND ----------

# MAGIC %md ## 6. Supply Chain Flow Metrics (Derived)

# COMMAND ----------

# Approximate value-chain metrics based on industry benchmarks
supply_chain_data = [
    ("Raw Materials",      "Steel, Rubber, Electronics", 100, 18),
    ("Component Mfg",      "Engines, Gearboxes, Hydraulics", 118, 22),
    ("Assembly (OEM)",     "Mahindra, John Deere, TAFE, CNH", 140, 35),
    ("Dealer Network",     "Authorised Dealerships (14,000+ in India)", 175, 20),
    ("End Customer",       "Farmer / Contractor", 210, None),
]

schema_sc = ["stage", "players", "index_value_100", "margin_pct"]
df_supply_chain = spark.createDataFrame(supply_chain_data, schema_sc)
df_supply_chain.write.format("delta").mode("overwrite").saveAsTable("gold.supply_chain_flow")
print("gold.supply_chain_flow: created")
df_supply_chain.show()

# COMMAND ----------

# MAGIC %md ## 7. Final Gold Table Inventory

# COMMAND ----------

gold_tables = ["market_summary", "india_seasonality", "hp_segment_trend",
               "price_vs_sales", "top_exporters", "supply_chain_flow"]

print("=== Gold Layer Summary ===")
for tbl in gold_tables:
    cnt = spark.sql(f"SELECT COUNT(*) AS n FROM gold.{tbl}").collect()[0]["n"]
    print(f"  gold.{tbl}: {cnt:,} rows  ✓")

print("\nAll Gold tables ready for Looker & Vercel serving.")
