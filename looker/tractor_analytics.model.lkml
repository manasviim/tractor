# tractor_analytics.model.lkml
# ---------------------------------------------------------------------------
# Looker model — Tractor Industry Analytics
# Connection: Databricks SQL Warehouse
# ---------------------------------------------------------------------------

connection: "databricks_sql"

# Include all view files in this project
include: "/views/*.view.lkml"

# ---------------------------------------------------------------------------
# Explores
# ---------------------------------------------------------------------------

explore: market_summary {
  label: "Global Market Summary"
  description: "Year-over-year tractor sales volume by country"

  join: india_seasonality {
    type:         left_outer
    sql_on:       ${market_summary.country} = 'India'
                  AND ${market_summary.year} = ${india_seasonality.calendar_year} ;;
    relationship: one_to_many
  }
}

explore: india_seasonality {
  label: "India Seasonality"
  description: "Monthly tractor sales pattern tied to Kharif / Rabi crop seasons"
}

explore: hp_segment_trend {
  label: "US HP Segment Trend"
  description: "US tractor retail sales broken down by horsepower category"
}

explore: price_vs_sales {
  label: "Price vs Sales Correlation"
  description: "US farm-equipment PPI index vs unit sales — shows demand elasticity"
}

explore: top_exporters {
  label: "Global Trade — Top Exporters"
  description: "Top 15 countries by tractor export value (UN Comtrade, HS 8701)"
}

explore: supply_chain_flow {
  label: "Supply Chain Value Flow"
  description: "Margin stack from raw materials to end customer"
}
