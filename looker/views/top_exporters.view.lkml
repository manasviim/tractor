# views/top_exporters.view.lkml
# ---------------------------------------------------------------------------
# Top 15 tractor exporting countries (UN Comtrade, HS 8701)
# Source: gold.top_exporters (Databricks)
# ---------------------------------------------------------------------------

view: top_exporters {
  sql_table_name: gold.top_exporters ;;

  dimension: pk {
    primary_key: yes
    hidden:      yes
    type:        string
    sql:         CONCAT(${exporter_country}, '_', ${data_year}) ;;
  }

  dimension: exporter_country {
    type:        string
    sql:         ${TABLE}.exporter_country ;;
    label:       "Exporter Country"
    map_layer_name: countries
    tags:        ["country"]
  }

  dimension: exporter_code {
    type:        string
    sql:         ${TABLE}.exporter_code ;;
    label:       "Country Code"
    hidden:      yes
  }

  dimension: rank {
    type:        number
    sql:         ${TABLE}.rank ;;
    label:       "Export Rank"
    value_format: "0"
  }

  dimension: data_year {
    type:        number
    sql:         ${TABLE}.data_year ;;
    label:       "Data Year"
    value_format: "0"
  }

  # ── Measures ────────────────────────────────────────────────────────────

  measure: export_value_usd_bn {
    type:        sum
    sql:         ${TABLE}.export_value_usd_bn ;;
    label:       "Export Value ($B)"
    value_format: "$#,##0.000B"
    drill_fields: [exporter_country, export_value_usd_bn]
  }

  measure: total_units_exported {
    type:        sum
    sql:         ${TABLE}.total_units ;;
    label:       "Units Exported"
    value_format: "#,##0"
  }
}
