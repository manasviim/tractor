# TractorIQ — How the Tractor Industry Works

> Data analytics task submission · Databricks + Looker + Vercel · Built in 24 hours

**Live App:** [https://tractor-insights.vercel.app](https://tractor-insights.vercel.app) *(deploy using the instructions below)*

---

## Project Overview

This project answers the question **"How does the tractor industry work?"** through a full end-to-end data analytics pipeline:

| Layer | Tool | What it does |
|---|---|---|
| Data Ingestion | **Databricks** (Bronze) | Reads raw CSVs + API data into Delta tables |
| Data Cleaning | **Databricks** (Silver) | Standardises types, parses fiscal years, adds crop season flags |
| Analytics | **Databricks** (Gold) | Computes 6 KPI aggregation tables for BI and frontend |
| BI Dashboards | **Looker** | LookML views + explores connected to Databricks SQL Warehouse |
| Web Report | **Vercel** (Next.js) | Public-facing interactive analytics report |

---

## Architecture

```
[Public Data Sources]
        │
        ▼
[Databricks — Bronze Layer]     ← Raw Delta tables
        │
        ▼
[Databricks — Silver Layer]     ← Cleaned, enriched Delta tables
        │
        ▼
[Databricks — Gold Layer]       ← KPI aggregations (6 tables)
        │
   ┌────┴─────┐
   ▼          ▼
[Looker]  [Vercel Next.js]
(Dashboards) (Web Report)
```

---

## Data Sources

All data is free and publicly available:

| Source | Dataset | URL |
|---|---|---|
| Ministry of Agriculture, India | Year-wise tractor & power tiller sales (FY2004-25) | https://dataful.in/datasets/3412/ |
| AEM (Association of Equipment Manufacturers) | US tractor retail sales by HP category (monthly) | https://www.aem.org/market-share-statistics/us-ag-tractor-and-combine-reports |
| FRED — St. Louis Fed | Agricultural Machinery PPI (WPU114) | https://fred.stlouisfed.org/series/WPU114 |
| UN Comtrade | Global tractor exports/imports — HS code 8701 | https://comtradeapi.un.org/ |
| IndexBox / ICCT | Global market size estimates | https://www.indexbox.io |

---

## Repository Structure

```
tractor/
├── notebooks/
│   ├── 01_bronze_ingestion.py       # Upload to Databricks: raw CSV → Delta tables
│   ├── 02_silver_transformation.py  # Cleaning, enrichment, fiscal year parsing
│   └── 03_gold_kpis.py              # KPI aggregations → 6 gold tables
│
├── looker/
│   ├── tractor_analytics.model.lkml # Looker model (connection + explores)
│   └── views/
│       ├── market_summary.view.lkml
│       ├── india_seasonality.view.lkml
│       ├── hp_segment_trend.view.lkml
│       ├── price_vs_sales.view.lkml
│       ├── top_exporters.view.lkml
│       └── supply_chain_flow.view.lkml
│
├── tractor-insights/                # Next.js web app (deploy to Vercel)
│   ├── src/app/
│   │   ├── page.tsx                 # Overview + pipeline explainer
│   │   ├── india/page.tsx           # India deep-dive (20yr trend, seasonality)
│   │   ├── market/page.tsx          # US market + HP segments + price correlation
│   │   └── global/page.tsx          # Global trade flow + top exporters
│   ├── src/lib/data.ts              # Static data (exported from Gold tables)
│   └── vercel.json
│
└── README.md
```

---

## Key Findings

### 1. India is the World's #1 Tractor Market by Volume
India sells ~875,000 tractors per year (FY2023-24), ahead of China (~763K) and the US (~289K).
Over 20 years (FY2004–FY2024), India's tractor sales grew at **~5.2% CAGR**, driven by:
- Government support: PM-KISAN transfers, Kisan Credit Cards, subsidy schemes
- Farm mechanisation policy: SMAM (Sub-Mission on Agricultural Mechanisation)
- Rising rural income and expanding NBFC/MFI financing networks

### 2. Tractor Sales Are Highly Seasonal (Driven by Crop Cycles)
In India, **over 60% of annual sales** are concentrated in just two windows:
- **Kharif sowing (Jun–Jul): ~23%** — monsoon triggers rice, cotton, soybean sowing
- **Rabi sowing (Nov–Dec): ~20%** — post-monsoon wheat, mustard preparation

This seasonality is exploited by manufacturers (inventory planning) and dealers (discount cycles).

### 3. Equipment Price Inflation is Compressing Demand in the US
The FRED WPU114 (Farm Machinery PPI) rose **+38%** from 2018 to 2024 due to steel costs, semiconductor shortages, and labour inflation. US tractor sales peaked at **355K units in 2021** and have since fallen **18% to 289K in 2024** — a direct demand-elasticity effect. The Pearson correlation between PPI and US unit sales is **r ≈ −0.52**.

### 4. HP Segmentation Splits East vs West
- **India / Asia**: ~65% of volume is in the **<40 HP** compact segment, suited to smallholder farms averaging <2 hectares
- **USA / Europe**: The **40+ HP** and **100+ HP** segments dominate for large-scale row-crop farming
- This structural difference drives completely different product roadmaps for global OEMs like John Deere

### 5. India is Rapidly Becoming a Global Tractor Exporter
India exported tractors to **162 countries in FY2024-25**, up from just 96 in FY2008-09 — a **69% expansion** in market reach. Export value reached **$3.2B (2024)**, ranking India #5 globally behind Germany ($8.5B), USA ($6.2B), Netherlands ($4.8B), and Japan ($4.1B).

---

## Setup Instructions

### Step 1: Databricks Notebooks

1. Create a [Databricks Community Edition](https://community.cloud.databricks.com/) account (free)
2. Download datasets:
   - India data: https://dataful.in/datasets/3412/ → download CSV
   - US data: Extract from AEM monthly PDF reports
3. In Databricks: **Workspace → Import** → upload the 3 `.py` notebook files from `notebooks/`
4. Upload CSVs to DBFS: **Data → Add Data → DBFS → Upload File** → path: `/FileStore/tractor/raw/`
5. Add secrets: **Settings → Secret Scopes → Create** scope `tractor`, add keys `comtrade_api_key` and `fred_api_key`
6. Run notebooks in order: `01` → `02` → `03`

### Step 2: Looker

1. In your Looker instance: **Admin → Database → Add Connection** → choose **Databricks**
2. Enter your Databricks SQL Warehouse connection string (from: **SQL Warehouses → Connection details**)
3. Create a new **LookML Project**: Upload files from `looker/` directory
4. Deploy the project and build dashboards using the 6 explores
5. Share dashboard with public link → copy the embed URL

### Step 3: Vercel

```bash
cd tractor-insights
npm install

# Set environment variable (optional — embeds Looker iframe)
# NEXT_PUBLIC_LOOKER_EMBED_URL=https://your-looker-instance/embed/...

npm run build        # verify build
npx vercel deploy    # deploy to Vercel (follow prompts)
```

Or connect the GitHub repo to Vercel for automatic deployments.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Next.js | 16.x | React framework (App Router) |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 3.x | Styling |
| Recharts | 2.x | Client-side charts |
| Databricks | Community Edition | Medallion architecture pipeline |
| Looker / Looker Studio | — | BI dashboards (LookML) |
| Vercel | — | Edge deployment |

---

## Data Methodology

- **India data**: Taken directly from Ministry of Agriculture annual publication via Dataful.in. Values are in thousands of units. Fiscal year (e.g., "2023-24") is mapped to calendar year 2023 (start year).
- **Monthly seasonality**: Derived by applying historical TAMA monthly distribution percentages to annual totals. Actual monthly data can replace this if TAMA CSV is available.
- **US data**: Aggregated from AEM monthly "Farm Flash" reports. HP categories: 2WD <40HP, 2WD 40-100HP, 2WD 100+HP, 4WD (all 100+HP).
- **Global trade data**: UN Comtrade API, HS code 8701 (Tractors other than pedestrian-controlled). Export flow only (`flowCode = "X"`).
- **Price index**: FRED series WPU114 — Producer Price Index for Farm Machinery & Equipment (base Dec 2002 = 100). January observation used as annual proxy.
- **Global market size** ($85.6B): IndexBox 2024 estimate cross-referenced with ICCT working paper.

---

*Built for a data analytics role assessment · June 2026*
