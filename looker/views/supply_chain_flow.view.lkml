# views/supply_chain_flow.view.lkml
# ---------------------------------------------------------------------------
# Tractor supply chain value-add stages (Manufacturer → Dealer → Farmer)
# Source: gold.supply_chain_flow (Databricks)
# ---------------------------------------------------------------------------

view: supply_chain_flow {
  sql_table_name: gold.supply_chain_flow ;;

  dimension: stage {
    primary_key: yes
    type:        string
    sql:         ${TABLE}.stage ;;
    label:       "Value Chain Stage"
  }

  dimension: players {
    type:        string
    sql:         ${TABLE}.players ;;
    label:       "Key Players"
  }

  dimension: index_value_100 {
    type:        number
    sql:         ${TABLE}.index_value_100 ;;
    label:       "Price Index (Raw Mat. = 100)"
    description: "Cumulative value addition index from raw materials (100) to customer"
    value_format: "0"
  }

  dimension: margin_pct {
    type:        number
    sql:         ${TABLE}.margin_pct ;;
    label:       "Stage Margin (%)"
    value_format: "0\%"
  }

  # ── Measures ────────────────────────────────────────────────────────────

  measure: total_value_add {
    type:        sum
    sql:         ${TABLE}.margin_pct ;;
    label:       "Total Channel Margin (%)"
    value_format: "0\%"
  }

  measure: stage_count {
    type:  count
    label: "# Stages"
  }
}
