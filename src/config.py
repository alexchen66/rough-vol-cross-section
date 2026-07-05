from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FEATURES = ROOT / "data" / "features"

# Universe
START_DATE = "2005-01-01"   # need 2yr of history before first rebalance
END_DATE   = "2023-12-31"
MIN_PRICE  = 5.0
MIN_DOLLAR_VOL_20D = 1_000_000   # $1M avg daily dollar volume
MIN_LISTING_DAYS   = 252

# Labels
HOLDING_PERIOD = 20   # trading days

# Walk-forward windows: (train_start, train_end, val_start, val_end, test_start, test_end)
WALK_FORWARD_WINDOWS = [
    ("2007-01-01", "2013-12-31", "2014-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
    ("2008-01-01", "2014-12-31", "2015-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2009-01-01", "2015-12-31", "2016-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
    ("2010-01-01", "2016-12-31", "2017-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2011-01-01", "2017-12-31", "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2012-01-01", "2018-12-31", "2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2013-01-01", "2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2014-01-01", "2020-12-31", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]
EMBARGO_DAYS = 20

# Features
PRICE_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
    "mom_12_1",
    "close_to_ma20", "close_to_ma60", "ma20_to_ma60",
    "price_position_252d",
    "close_to_52w_high",   # George & Hwang (2004): near 52w high → positive
    "max_ret_20d",         # Bali et al. (2011): lottery effect → negative
]
RISK_FEATURES = [
    "rv_20d", "rv_60d", "rv_120d",
    "downside_vol_60d",
    "beta_252d",
    "idio_vol_252d",
    "max_dd_60d", "max_dd_252d",
    "skew_60d",
    "amihud_20d",
    "dollar_vol_20d",
    "turnover_20d",
]
FUNDAMENTAL_FEATURES = [
    "book_to_market",
    "earnings_yield",
    "sales_to_price",
    "roe", "roa",
    "gross_profitability",
    "asset_growth",
    "revenue_growth",
    "debt_to_equity",
    "debt_to_assets",
    "sue",       # Standardized Unexpected Earnings — PEAD signal
    "accruals",  # Earnings quality: low accruals → higher future returns
]
ROUGH_VOL_FEATURES = [
    "hurst_126d", "hurst_252d",
    "vol_of_vol_60d", "vol_of_vol_126d",
    "idio_roughness_126d",
    "roughness_x_momentum",
    "roughness_x_illiquidity",
]
ANALYST_FEATURES = [
    "sue_ibes",      # WRDS-computed SUE: (actual - consensus) / dispersion
    "revision_1m",   # 1-month % change in consensus EPS estimate
    "revision_3m",   # 3-month % change in consensus EPS estimate
    "est_disp",      # analyst forecast dispersion (uncertainty)
    "num_analysts",  # analyst coverage count
]
ALL_FEATURES = (
    PRICE_FEATURES + RISK_FEATURES + FUNDAMENTAL_FEATURES
    + ROUGH_VOL_FEATURES + ANALYST_FEATURES
)

# Portfolio
LONG_ONLY_N = 50
TOP_PCT     = 0.10   # top/bottom decile for long-short

# Transaction costs to test (bps)
COST_BPS_GRID = [0, 5, 10, 20]
