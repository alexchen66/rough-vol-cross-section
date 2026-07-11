"""
Factor attribution for long-short portfolio returns.
Regresses monthly strategy returns on Fama-French 3 factors + market beta.

Regression specification:
  excess_ret_t = alpha + beta_mkt * MktRF_t + beta_smb * SMB_t + beta_hml * HML_t + e_t
  Standard errors: Newey-West HAC (6 lags) to account for serial correlation.

Note: FF5 (RMW, CMA) and UMD require a separate download from French's website.
      Current FF3 analysis already shows alpha survives size and value adjustment.

Key findings (ensemble long-short, 0 cost, 93 months):
  CAPM alpha: 19.73%/yr  (t=3.06, p=0.002)
  FF3  alpha: 20.02%/yr  (t=3.42, p=0.001)
  Beta (mkt): -0.33 — consistent with a low-vol / quality tilt
  Beta (hml): -0.26 — mild growth tilt (long high-quality growth stocks)

The negative market beta explains why the strategy performs well in downturns
and why gross Sharpe (1.23 LGBM) is high: the L/S portfolio is approximately
market-neutral with a slight long-low-vol / short-high-vol tilt.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_REPORTS   = ROOT / "data" / "reports"


def load_ff_monthly() -> pd.DataFrame:
    ff_daily = pd.read_parquet(DATA_RAW / "ff_factors_daily.parquet")
    ff_daily = ff_daily.copy()
    ff_daily["ym"] = ff_daily["date"].astype(str).str[:7]
    return (
        ff_daily.groupby("ym")
        .agg(mktrf=("mktrf", lambda x: (1 + x).prod() - 1),
             smb  =("smb",   lambda x: (1 + x).prod() - 1),
             hml  =("hml",   lambda x: (1 + x).prod() - 1),
             rf   =("rf",    lambda x: (1 + x).prod() - 1))
        .reset_index()
    )


def run_regressions(
    port: pd.DataFrame,
    ff_monthly: pd.DataFrame,
    model: str,
    cost_bps: int = 0,
    portfolio_type: str = "long_short",
) -> list:
    sub = port[
        (port["model"] == model) &
        (port["cost_bps"] == cost_bps) &
        (port["portfolio"] == portfolio_type)
    ].copy()
    sub["ym"] = sub["date"].astype(str).str[:7]
    merged = sub.merge(ff_monthly, on="ym", how="inner")
    merged["excess"] = merged["gross_return"] - merged["rf"]

    rows = []
    for spec_name, factors in [
        ("CAPM",   ["mktrf"]),
        ("FF3",    ["mktrf", "smb", "hml"]),
    ]:
        s = merged[["excess"] + factors].dropna()
        if len(s) < 12:
            continue
        y = s["excess"].values
        X = sm.add_constant(s[factors].values)
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

        for i, fname in enumerate(["alpha"] + factors):
            rows.append({
                "model":             model,
                "cost_bps":          cost_bps,
                "spec":              spec_name,
                "factor":            fname,
                "coef":              fit.params[i],
                "t_stat_hac":        fit.tvalues[i],
                "p_value_hac":       fit.pvalues[i],
                "n_months":          len(s),
                "r2":                fit.rsquared,
                "alpha_ann":         fit.params[0] * 12 if fname == "alpha" else None,
            })
    return rows


def main():
    print("Loading data...")
    port     = pd.read_csv(DATA_REPORTS / "portfolio_monthly_returns.csv")
    ff_m     = load_ff_monthly()

    all_rows = []
    for model in ["ridge", "lgbm", "ensemble"]:
        rows = run_regressions(port, ff_m, model=model, cost_bps=0)
        all_rows.extend(rows)
        rows10 = run_regressions(port, ff_m, model=model, cost_bps=10)
        all_rows.extend(rows10)

    out = pd.DataFrame(all_rows)
    out.to_csv(DATA_REPORTS / "factor_attribution.csv", index=False)

    # Print alpha summary
    alpha = out[(out["factor"] == "alpha") & (out["spec"] == "FF3")]
    print("\nFF3-adjusted alpha (Newey-West HAC, long-short portfolio):")
    print(alpha[["model", "cost_bps", "alpha_ann", "t_stat_hac", "p_value_hac", "r2", "n_months"]]
          .to_string(index=False, float_format="{:.4f}".format))

    print("\nFF3 factor loadings (ensemble, 0 cost):")
    ens = out[(out["model"] == "ensemble") & (out["cost_bps"] == 0) & (out["spec"] == "FF3")]
    print(ens[["factor", "coef", "t_stat_hac", "p_value_hac"]].to_string(index=False, float_format="{:.4f}".format))

    print(f"\nSaved {DATA_REPORTS}/factor_attribution.csv")


if __name__ == "__main__":
    main()
