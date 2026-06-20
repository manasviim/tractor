# views/india_seasonality.view.lkml
# ---------------------------------------------------------------------------
# India monthly sales seasonality with Kharif / Rabi season labels
# Source: gold.india_seasonality (Databricks)
# ---------------------------------------------------------------------------

view: india_seasonality {
  sql_table_name: gold.india_seasonality ;;

  dimension: pk {
    primary_key: yes
    hidden:      yes
    type:        string
    sql:         CONCAT(${month_num}, '_', ${season}) ;;
  }

  dimension: month_num {
    type:        number
    sql:         ${TABLE}.month_num ;;
    label:       "Month Number"
    hidden:      yes
  }

  dimension: month_name {
    type:        string
    sql:         ${TABLE}.month_name ;;
    label:       "Month"
    order_by_field: month_num
  }

  dimension: season {
    type:        string
    sql:         ${TABLE}.season ;;
    label:       "Crop Season"
    description: "Kharif (Jun–Sep), Rabi (Oct–Feb), or Harvest/Slack"

    tags: ["season"]
  }

  dimension: calendar_year {
    type:        number
    sql:         ${TABLE}.calendar_year ;;
    label:       "Calendar Year"
    hidden:      yes
  }

  # ── Measures ────────────────────────────────────────────────────────────

  measure: avg_monthly_share {
    type:        average
    sql:         ${TABLE}.avg_pct_share ;;
    label:       "Avg Monthly Share (%)"
    value_format: "0.0\%"
    description: "Average % of annual sales in this month across all years"
  }

  measure: avg_units_thousands {
    type:        average
    sql:         ${TABLE}.avg_units_thousands ;;
    label:       "Avg Units (000s)"
    value_format: "#,##0.0"
  }

  measure: fy2324_units_thousands {
    type:        sum
    sql:         ${TABLE}.fy2324_units_thousands ;;
    label:       "FY23-24 Units (000s)"
    value_format: "#,##0.0"
    description: "Estimated units sold in FY 2023-24 for this month"
  }
}
