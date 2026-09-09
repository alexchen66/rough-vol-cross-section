"""
Formal ablation + spanning tests for the roughness features.

Answers, with proper statistics, the question the backtests left open:
"adding roughness didn't seem to change anything — is that real?"

Four tests:
  1. Paired incremental IC test: full model vs no-rough model, same months,
     Newey-West (HAC) t-stat on the monthly IC differences.
  2. Standalone GK-roughness factor: rank IC and decile long-short with HAC t.
  3. Fama-MacBeth cross-sectional regression of forward returns on roughness
     with vol/liquidity controls (is the premium subsumed by known factors?).
  4. Spanning regression of the roughness long-short return on FF3 compounded
     over matching 20-day holding windows, plus turnover / cost survival.

Run:  python src/roughness_ablation_tests.py
Outputs: data/reports/roughness_ablation_summary.csv (+ console table)
"""
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES

SIG = "roughness_gk_126d"
CONTROLS = ["rv_20d", "beta_252d", "idio_vol_252d", "amihud_20d", "turnover_20d"]
HOLDING_DAYS = 20
HAC_LAGS = 6          # ~ holding period overlap at monthly rebalance
DECILE = 0.10


# ── helpers ──────────────────────────────────────────────────────────────────

def load_pq(path, columns=None):
    """Parquet loader; casts timestamp cols to 'YYYY-MM-DD' strings."""
    t = pq.read_table(path, columns=columns)
    for i, col in enumerate(t.schema.names):
        if pa.types.is_timestamp(t.schema.field(col).type):
            t = t.set_column(i, col, t.column(col).cast(pa.string()))
    df = t.to_pandas()
    if "date" in df.columns:
        df["date"] = df["date"].str[:10]
    return df


def hac_mean(x, lags=HAC_LAGS):
    """Newey-West mean, se, t for a (possibly autocorrelated) series."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(s / n)
    return m, se, m / se, n


def ols_hac(y, X, lags=HAC_LAGS):
    """OLS with Newey-West standard errors. Returns beta, se, t, r2."""
    Xm = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    e = y - Xm @ beta
    n, k = Xm.shape
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    S = np.zeros((k, k))
    for l in range(lags + 1):
        w = 1.0 if l == 0 else (1 - l / (lags + 1))
        for i in range(l, n):
            outer = np.outer(Xm[i] * e[i], Xm[i - l] * e[i - l])
            S += w * (outer if l == 0 else outer + outer.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, beta / se, r2


# ── Test 1: paired incremental IC ────────────────────────────────────────────

def paired_ic_test(reports_dir):
    full = pd.read_csv(reports_dir / "ic_by_rebalance_date_all_windows.csv")
    nor = pd.read_csv(reports_dir / "ic_by_rebalance_date_no_rough_all_windows.csv")
    rows = []
    for model in ["ridge", "lgbm"]:
        f = full[full.model == model].set_index("date")["spearman_rank_ic"]
        nr = nor[nor.model == model].set_index("date")["spearman_rank_ic"]
        common = f.index.intersection(nr.index)
        mf, _, tf, _ = hac_mean(f.loc[common])
        mn, _, tn, _ = hac_mean(nr.loc[common])
        md, _, td, n = hac_mean(f.loc[common] - nr.loc[common])
        rows.append(dict(test="paired_incremental_ic", model=model, n=n,
                         ic_full=mf, t_full=tf, ic_norough=mn, t_norough=tn,
                         ic_diff=md, t_diff=td))
    return pd.DataFrame(rows)


# ── Tests 2–4: standalone factor, Fama-MacBeth, spanning ────────────────────

def zscore(g):
    return (g - g.mean()) / g.std()


def factor_tests():
    rough = load_pq(DATA_FEATURES / "features_rough_vol_v2.parquet",
                    ["date", "permno", SIG])
    risk = load_pq(DATA_FEATURES / "features_risk.parquet",
                   ["date", "permno"] + CONTROLS)
    labels = load_pq(DATA_PROCESSED / "labels.parquet",
                     ["date", "permno", "r_fwd", "y_xs"])
    df = labels.merge(rough, on=["date", "permno"]).merge(risk, on=["date", "permno"])

    # -- standalone rank IC --
    def _ic(g):
        v = g[[SIG, "y_xs"]].dropna()
        return spearmanr(v[SIG], v["y_xs"])[0] if len(v) >= 100 else np.nan

    ic = df.groupby("date").apply(_ic, include_groups=False).dropna()
    m_ic, _, t_ic, n_ic = hac_mean(ic)

    # -- decile long-short --
    def _ls(g):
        v = g[[SIG, "r_fwd"]].dropna()
        if len(v) < 100:
            return np.nan
        k = max(1, int(len(v) * DECILE))
        v = v.sort_values(SIG)
        return v.tail(k)["r_fwd"].mean() - v.head(k)["r_fwd"].mean()

    ls = df.groupby("date").apply(_ls, include_groups=False).dropna()
    m_ls, _, t_ls, _ = hac_mean(ls)

    # -- Fama-MacBeth with controls --
    def _fm(g, xcols):
        sub = g[["y_xs"] + xcols].dropna()
        if len(sub) < 100:
            return None
        X = np.column_stack([np.ones(len(sub)), sub[xcols].apply(zscore).values])
        beta, *_ = np.linalg.lstsq(X, sub["y_xs"].values, rcond=None)
        return pd.Series(beta[1:], index=xcols)

    uni = df.groupby("date").apply(lambda g: _fm(g, [SIG]), include_groups=False).dropna()
    multi = df.groupby("date").apply(lambda g: _fm(g, [SIG] + CONTROLS), include_groups=False).dropna()
    m_u, _, t_u, _ = hac_mean(uni[SIG])
    m_m, _, t_m, _ = hac_mean(multi[SIG])

    # -- spanning regression on FF3 compounded over the holding window --
    ff = load_pq(DATA_RAW / "ff_factors_daily.parquet").sort_values("date").reset_index(drop=True)
    idx = {d: i for i, d in enumerate(ff["date"])}
    rows = []
    for d in ls.index:
        i = idx.get(d)
        if i is None or i + 1 + HOLDING_DAYS + 1 >= len(ff):
            continue
        w = ff.iloc[i + 2: i + 2 + HOLDING_DAYS]  # returns accrue after entry at t+1
        rows.append(dict(date=d,
                         mktrf=(1 + w.mktrf).prod() - 1,
                         smb=(1 + w.smb).prod() - 1,
                         hml=(1 + w.hml).prod() - 1))
    ffw = pd.DataFrame(rows).set_index("date")
    sp = ffw.join(ls.rename("ls")).dropna()
    b, se, t, r2 = ols_hac(sp["ls"].values, sp[["mktrf", "smb", "hml"]].values)

    # -- turnover / cost survival --
    prev_l = prev_s = None
    tos = []
    for d in sorted(ls.index):
        g = df[df.date == d].set_index("permno")
        v = g[[SIG]].dropna().sort_values(SIG)
        k = max(1, int(len(v) * DECILE))
        L, S = set(v.tail(k).index), set(v.head(k).index)
        if prev_l is not None:
            tos.append(1 - (len(L & prev_l) + len(S & prev_s)) / (len(L) + len(S)))
        prev_l, prev_s = L, S
    to = float(np.mean(tos))

    summary = [
        dict(test="standalone_rank_ic", value=m_ic, t=t_ic, n=n_ic),
        dict(test="decile_ls_monthly", value=m_ls, t=t_ls, n=len(ls)),
        dict(test="fm_lambda_univariate_bp", value=m_u * 1e4, t=t_u, n=len(uni)),
        dict(test="fm_lambda_with_controls_bp", value=m_m * 1e4, t=t_m, n=len(multi)),
        dict(test="ff3_alpha_monthly", value=b[0], t=t[0], n=len(sp)),
        dict(test="ff3_r2", value=r2, t=np.nan, n=len(sp)),
        dict(test="monthly_turnover_oneside", value=to, t=np.nan, n=len(tos)),
    ]
    for c_bps in [5, 10, 20]:
        net = ls.mean() - 2 * to * c_bps / 1e4
        summary.append(dict(test=f"ls_net_{c_bps}bps_ann", value=net * 12, t=np.nan, n=len(ls)))
    return pd.DataFrame(summary)


def main():
    reports = ROOT / "data" / "reports"
    t1 = paired_ic_test(reports)
    print("\n=== Test 1: paired incremental IC (full vs no-rough) ===")
    print(t1.round(4).to_string(index=False))

    t2 = factor_tests()
    print("\n=== Tests 2-4: standalone factor / Fama-MacBeth / spanning ===")
    print(t2.round(4).to_string(index=False))

    out = reports / "roughness_ablation_summary.csv"
    pd.concat([t1, t2]).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
