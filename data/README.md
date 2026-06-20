# Data Directory — Tractor Industry Analytics

## Structure

```
data/
├── raw/           ← Source files exactly as downloaded
└── processed/     ← Cleaned, enriched, standardized files (created by Silver notebooks)
```

---

## Raw Datasets

### 1. India Tractor Sales (2003–2025)
**File:** `india_tractor_sales_2004_2025.csv`  
**Source:** TMA (Tractor Manufacturers Association of India) — Annual press releases  
**URL:** https://tractormfrs.com / https://dataful.in/datasets/3412/  
**Columns:** `fiscal_year, year_start, year_end, total_units_sold, source`  
**Coverage:** FY 2003-04 to 2024-25 (22 years)  
**Notes:** India is the world's #1 tractor market by volume (~900K units/year)

### 2. India Monthly Seasonality
**File:** `india_tractor_monthly_seasonality.csv`  
**Source:** TMA seasonal patterns + Ministry of Agriculture crop calendars  
**Columns:** `month, month_name, avg_share_pct, season_label, notes`  
**Notes:** ~34% of sales happen in June–August (Kharif sowing season)

### 3. India Brand Market Share (2023)
**File:** `india_tractor_brand_market_share.csv`  
**Source:** TMA + ICRA / CRISIL industry reports (public)  
**Columns:** `brand, parent_company, country_of_origin, market_share_2023_pct, hp_range_focus, key_models`  
**Notes:** Mahindra dominates with ~42% share

### 4. US Tractor Sales by HP Category (2010–2025)
**File:** `us_aem_tractor_sales_by_hp.csv`  
**Source:** AEM (Association of Equipment Manufacturers) Farm Flash Reports  
**URL:** https://www.aem.org/market-share-statistics/us-ag-tractor-and-combine-reports  
**Columns:** `year, under_40hp_units, 40_to_100hp_units, over_100hp_units, 4wd_units, total_2wd_4wd`  
**Notes:** 2025 is YTD provisional; full subscription data at https://shop.aem.org/

### 5. AEM May 2026 Farm Flash Press Release
**File:** `aem_may2026_farm_flash.pdf`  
**Source:** AEM (free press release)  
**Notes:** Latest available monthly snapshot as of June 2026

### 6. Global Tractor Production by Country (2018–2023)
**File:** `global_tractor_production_by_country.csv`  
**Source:** TMA (India), China Tractor Association, AEM (US), VDMA (Germany), JAMA (Japan)  
**Columns:** `country, country_code, year, tractors_produced_units, tractors_exported_units, domestic_sales_estimate`

### 7. UN Comtrade — Tractor Trade (HS 8701, 2015–2022)
**File:** `comtrade_hs8701_tractors_2015_2022.csv`  
**Source:** UN Comtrade Public Preview API (no key required)  
**URL:** https://comtradeapi.un.org/  
**Columns:** `year, reporterCode, flowCode, partnerCode, netWgt_kg, fobValue_usd, primaryValue_usd, qty`  
**Countries:** Germany (276), USA (840), Japan (392), China (156), France (250), Russia (643), Italy (380), India (356)  
**Notes:** flowCode X = exports, M = imports; fobValue_usd in USD

**Country Code Reference:**
| Code | Country |
|------|---------|
| 276 | Germany |
| 840 | USA |
| 392 | Japan |
| 156 | China |
| 250 | France |
| 643 | Russia |
| 380 | Italy |
| 356 | India |

### 8. FRED — Agricultural Machinery Price Index (WPU114)
**File:** `fred_wpu114.csv`  
**Source:** Federal Reserve Bank of St. Louis (FRED)  
**Series:** WPU114 — PPI: Farm and garden machinery and equipment  
**URL:** https://fred.stlouisfed.org/series/WPU114  
**Coverage:** 1939–present (monthly)  
**Unit:** Index, 1982=100

### 9. FRED — Farm Equipment PPI (WPUFD41312)
**File:** `fred_farm_equipment_ppi.csv`  
**Source:** FRED  
**Series:** WPUFD41312 — PPI: Farm machinery and equipment  
**Coverage:** 1947–present (monthly)

### 10. FRED — Corn Price Index (WPU012202)
**File:** `fred_corn_price_index.csv`  
**Source:** FRED  
**Series:** WPU012202 — PPI: Corn  
**Coverage:** 1971–present (monthly)  
**Notes:** Key demand driver for tractor sales (corn price → farm income → tractor purchases)

### 11. FRED — Wheat Price Index (WPU012101)
**File:** `fred_wheat_price_index.csv`  
**Source:** FRED  
**Series:** WPU012101 — PPI: Wheat  
**Coverage:** 1974–present (monthly)

### 12. FRED — US Net Farm Income (A368RC1A027NBEA)
**File:** `fred_us_net_farm_income.csv`  
**Source:** FRED / BEA  
**Series:** A368RC1A027NBEA — Net value added by private farm sector (billions USD)  
**Coverage:** 1948–2024 (annual)

### 13. FAOSTAT — Arable Land by Country (1961–2022)
**File:** `faostat_arable_land_selected.csv`  
**Source:** FAO STAT — Inputs/Land Use domain  
**URL:** https://www.fao.org/faostat/en/#data/RL  
**Columns:** FAOSTAT normalized format (Area, Item, Element, Year, Unit, Value)  
**Coverage:** India, China, USA, Germany, France, Brazil, Japan, Pakistan, Bangladesh, World aggregate  
**Notes:** Full bulk ZIP also in `faostat_land/` subdirectory

---

## Files Still to Obtain

| File | Source | How |
|------|--------|-----|
| `india_monthly_sales_brandwise_2015_2023.csv` | Dataful.in dataset 3412 | Register + download from https://dataful.in/datasets/3412/ |
| `us_aem_monthly_2000_2025.csv` | AEM subscription | Purchase at https://shop.aem.org/ OR use USDA NASS free API |
| `comtrade_hs8701_2000_2014.csv` | UN Comtrade | Register API key at https://comtradeapi.un.org/ |

---

## Data Quality Notes

- India TMA figures are widely cited in media and ICRA/CRISIL research; ±2% margin of error for fiscal years
- US AEM HP breakdowns are estimated from AEM press releases and USDA Census of Agriculture
- Comtrade data: `isReported=false` means the figure is estimated/aggregated, not directly reported by the country
- FAOSTAT arable land is in 1000 hectares; multiply by 1000 for hectares
