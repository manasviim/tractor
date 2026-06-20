# views/market_summary.view.lkml
# ---------------------------------------------------------------------------
# Global tractor market — annual units sold by country
# Source: gold.market_summary (Databricks)
# ---------------------------------------------------------------------------

view: market_summary {
  sql_table_name: gold.market_summary ;;

  # ── Dimensions ─────────────────────────────────────────────────────────

  dimension: pk {
    primary_key: yes
    hidden:      yes
    type:        string
    sql:         CONCAT(${country}, '_', ${year}) ;;
  }

  dimension: country {
    type:        string
    sql:         ${TABLE}.country ;;
    label:       "Country"
    description: "Reporting country (India, USA, etc.)"
  }

  dimension: year {
    type:        number
    sql:         ${TABLE}.year ;;
    label:       "Calendar Year"
    value_format: "0"
  }

  dimension: yoy_pct {
    type:        number
    sql:         ${TABLE}.yoy_pct ;;
    label:       "YoY Change (%)"
    value_format: "+0.0\%;-0.0\%;0.0\%"
    description: "Year-over-year percentage change in unit sales"
  }

  # ── Measures ────────────────────────────────────────────────────────────

  measure: total_units_sold {
    type:        sum
    sql:         ${TABLE}.units_sold ;;
    label:       "Total Units Sold"
    value_format: "#,##0"
    drill_fields: [country, year, total_units_sold]
  }

  measure: total_units_sold_thousands {
    type:        sum
    sql:         ${TABLE}.units_sold_thousands ;;
    label:       "Units Sold (000s)"
    value_format: "#,##0.0K"
  }

  measure: avg_yoy_growth {
    type:        average
    sql:         ${TABLE}.yoy_pct ;;
    label:       "Avg YoY Growth (%)"
    value_format: "0.0\%"
  }

  measure: country_count {
    type:  count_distinct
    sql:   ${TABLE}.country ;;
    label: "# Countries"
  }
}
