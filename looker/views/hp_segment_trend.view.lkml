# views/hp_segment_trend.view.lkml
# ---------------------------------------------------------------------------
# US tractor retail sales by horsepower segment (AEM data)
# Source: gold.hp_segment_trend (Databricks)
# ---------------------------------------------------------------------------

view: hp_segment_trend {
  sql_table_name: gold.hp_segment_trend ;;

  dimension: pk {
    primary_key: yes
    hidden:      yes
    type:        string
    sql:         CONCAT(${year}, '_', ${hp_category}) ;;
  }

  dimension: year {
    type:        number
    sql:         ${TABLE}.year ;;
    label:       "Year"
    value_format: "0"
  }

  dimension: hp_category {
    type:        string
    sql:         ${TABLE}.hp_category ;;
    label:       "HP Category (raw)"
    hidden:      yes
  }

  dimension: segment_label {
    type:        string
    sql:         ${TABLE}.segment_label ;;
    label:       "HP Segment"
    description: "< 40 HP | 40–100 HP | 100+ HP"
    order_by_field: hp_category
  }

  dimension: pct_of_year {
    type:        number
    sql:         ${TABLE}.pct_of_year ;;
    label:       "% of Annual Sales"
    value_format: "0.0\%"
  }

  # ── Measures ────────────────────────────────────────────────────────────

  measure: total_units {
    type:        sum
    sql:         ${TABLE}.units_sold ;;
    label:       "Units Sold"
    value_format: "#,##0"
    drill_fields: [year, segment_label, total_units]
  }

  measure: avg_annual_units {
    type:        average
    sql:         ${TABLE}.units_sold ;;
    label:       "Avg Annual Units"
    value_format: "#,##0"
  }

  measure: avg_segment_share {
    type:        average
    sql:         ${TABLE}.pct_of_year ;;
    label:       "Avg Segment Share (%)"
    value_format: "0.0\%"
  }
}
