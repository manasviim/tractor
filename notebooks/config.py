"""
Configuration for Tractor Industry Analytics Notebooks
After registration, add your API keys here.
"""

# ============================================================
# API Keys — Fill in after registration
# ============================================================

# FRED (St. Louis Fed) — Register at:
# https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = ""  # e.g. "abcdef1234567890abcdef1234567890"

# UN Comtrade — Register at:
# https://comtradeapi.un.org/
COMTRADE_API_KEY = ""  # e.g. "eyJhbGci..."

# USDA NASS Quick Stats — Register at:
# https://quickstats.nass.usda.gov/api
USDA_NASS_API_KEY = ""  # e.g. "A1B2C3D4-E5F6-..."

# ============================================================
# Paths (Databricks DBFS paths — update after uploading CSVs)
# ============================================================
DBFS_RAW_PATH = "/FileStore/tractor/raw"
DBFS_PROCESSED_PATH = "/FileStore/tractor/processed"

# Local paths (for development outside Databricks)
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

# ============================================================
# Data Files
# ============================================================
FILES = {
    "india_sales":       "india_tractor_sales_2004_2025.csv",
    "india_seasonality": "india_tractor_monthly_seasonality.csv",
    "india_brands":      "india_tractor_brand_market_share.csv",
    "us_aem_hp":         "us_aem_tractor_sales_by_hp.csv",
    "global_production": "global_tractor_production_by_country.csv",
    "comtrade":          "comtrade_hs8701_tractors_2015_2022.csv",
    "fred_ppi":          "fred_wpu114.csv",
    "fred_equip_ppi":    "fred_farm_equipment_ppi.csv",
    "fred_corn":         "fred_corn_price_index.csv",
    "fred_wheat":        "fred_wheat_price_index.csv",
    "fred_farm_income":  "fred_us_net_farm_income.csv",
    "faostat_land":      "faostat_arable_land_selected.csv",
}

# ============================================================
# Country Code Mapping (UN Comtrade)
# ============================================================
COUNTRY_CODES = {
    276: "Germany",
    840: "USA",
    392: "Japan",
    156: "China",
    250: "France",
    643: "Russia",
    380: "Italy",
    356: "India",
    528: "Netherlands",
    826: "United Kingdom",
    616: "Poland",
    792: "Turkey",
    36:  "Australia",
    76:  "Brazil",
    410: "South Korea",
}

# ============================================================
# HP Category Labels (AEM)
# ============================================================
HP_CATEGORIES = {
    "under_40hp":    "< 40 HP",
    "40_to_100hp":   "40–100 HP",
    "over_100hp":    "> 100 HP",
    "4wd":           "4WD",
}

# ============================================================
# Season Labels (India, fiscal year = April to March)
# ============================================================
INDIA_SEASONS = {
    "Kharif": {"months": [6, 7, 8, 9], "crops": ["Rice", "Maize", "Soybean", "Cotton", "Groundnut"]},
    "Rabi":   {"months": [10, 11, 12, 1, 2, 3], "crops": ["Wheat", "Mustard", "Chickpea", "Lentil"]},
    "Zaid":   {"months": [3, 4, 5], "crops": ["Watermelon", "Muskmelon", "Cucumber"]},
}
