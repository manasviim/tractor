# views/price_vs_sales.view.lkml
# ---------------------------------------------------------------------------
# US farm-equipment PPI vs tractor unit sales
# Source: gold.price_vs_sales (Databricks)
# ---------------------------------------------------------------------------

view: price_vs_sales {
  sql_table_name: gold.price_vs_sales ;;

  dimension: year {
    primary_key:  yes
    type:         number
    sql:          ${TABLE}.year ;;
    label:        "Year"
    value_format: "0"
  }

  dimension: ppi_value {
    type:        number
    sql:         ${TABLE}.ppi_value ;;
    label:       "Farm Eq. PPI (Index)"
    description: "FRED WPU114 — Producer Price Index for Farm Machinery & Equipment (base Dec 2002 = 100)"
  }

  dimension: units_thousands {
    type:        number
    sql:         ${TABLE}.units_thousands ;;
    label:       "US Sales (000s)"
    value_format: "#,##0.0"
  }

  # ── Measures ────────────────────────────────────────────────────────────

  measure: avg_ppi {
    type:        average
    sql:         ${TABLE}.ppi_value ;;
    label:       "Avg PPI"
    value_format: "0.0"
  }

  measure: avg_us_sales_thousands {
    type:        average
    sql:         ${TABLE}.units_thousands ;;
    label:       "Avg US Sales (000s)"
    value_format: "#,##0.0"
  }

  measure: max_ppi {
    type:        max
    sql:         ${TABLE}.ppi_value ;;
    label:       "Peak PPI"
    value_format: "0.0"
  }

  measure: total_us_units {
    type:        sum
    sql:         ${TABLE}.total_units ;;
    label:       "Total US Units"
    value_format: "#,##0"
  }
}
