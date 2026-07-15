import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


PREDICTIONS = None
TARGETS = None
BASE_SHIFT = 1
MIN_BLOCK_SHIFT = 60


def init_worker(predictions, targets, base_shift, min_block_shift):
    global PREDICTIONS, TARGETS, BASE_SHIFT, MIN_BLOCK_SHIFT
    PREDICTIONS = predictions
    TARGETS = targets
    BASE_SHIFT = base_shift
    MIN_BLOCK_SHIFT = min_block_shift


def estimate_kappa_tstat(pred, ret):
    m = np.isfinite(pred) & np.isfinite(ret)
    pred = pred[m]
    ret = ret[m]
    if len(pred) < 20:
        return np.nan

    r = ret[1:]
    pred_t = pred[:-1]
    pred_t1 = pred[1:]

    def alm_model(x, kappa, intercept):
        p0, p1 = x
        return (
            np.log(1 - kappa * np.exp(p0))
            - np.log(1 - kappa * np.exp(p1))
            + intercept
        )

    try:
        popt, pcov = curve_fit(
            alm_model,
            (pred_t, pred_t1),
            r,
            p0=[0.5, 0.0],
            bounds=([0, -1], [1, 1]),
            maxfev=10000,
        )
        se = np.sqrt(np.diag(pcov))
        if not np.isfinite(se[0]) or se[0] < 1e-30:
            return np.nan
        return popt[0] / se[0]
    except Exception:
        return np.nan


def summarize_tstats(test_name, rep, seed, tstats):
    finite = np.isfinite(tstats)
    n_finite = int(finite.sum())
    sig = int((tstats[finite] > 1.96).sum())
    sig_trim50 = int(((tstats[finite] > 1.96) & (tstats[finite] <= 50)).sum())
    return {
        "test": test_name,
        "rep": rep,
        "seed": seed,
        "n_finite_tstats": n_finite,
        "sig_kappa_t_gt_1_96": sig,
        "sig_share": sig / n_finite if n_finite else np.nan,
        "n_t_gt_50": int((tstats[finite] > 50).sum()),
        "sig_kappa_t_gt_1_96_t_le_50": sig_trim50,
        "sig_share_t_le_50_den_all": sig_trim50 / n_finite if n_finite else np.nan,
        "median_kappa_tstat": float(np.nanmedian(tstats)),
        "p95_kappa_tstat": float(np.nanpercentile(tstats, 95)),
        "p99_kappa_tstat": float(np.nanpercentile(tstats, 99)),
        "max_kappa_tstat": float(np.nanmax(tstats)),
    }


def baseline_tstats():
    n_stocks = PREDICTIONS.shape[0]
    tstats = np.full(n_stocks, np.nan)
    for i in range(n_stocks):
        pred = PREDICTIONS[i, :-BASE_SHIFT]
        ret = TARGETS[i, BASE_SHIFT:]
        tstats[i] = estimate_kappa_tstat(pred, ret)
    return tstats


def firm_permutation_tstats(rng):
    n_stocks = PREDICTIONS.shape[0]
    perm = rng.permutation(n_stocks)
    tstats = np.full(n_stocks, np.nan)
    for i in range(n_stocks):
        pred = PREDICTIONS[perm[i], :-BASE_SHIFT]
        ret = TARGETS[i, BASE_SHIFT:]
        tstats[i] = estimate_kappa_tstat(pred, ret)
    return tstats


def date_permutation_tstats(rng):
    n_stocks, n_dates = PREDICTIONS.shape
    pred_perm = np.empty((n_stocks, n_dates - BASE_SHIFT), dtype=float)
    for t in range(n_dates - BASE_SHIFT):
        pred_perm[:, t] = PREDICTIONS[rng.permutation(n_stocks), t]

    tstats = np.full(n_stocks, np.nan)
    for i in range(n_stocks):
        ret = TARGETS[i, BASE_SHIFT:]
        tstats[i] = estimate_kappa_tstat(pred_perm[i], ret)
    return tstats


def circular_block_shift_tstats(rng):
    n_stocks, n_dates = PREDICTIONS.shape
    tstats = np.full(n_stocks, np.nan)
    usable_len = n_dates - BASE_SHIFT
    for i in range(n_stocks):
        shift = int(rng.integers(MIN_BLOCK_SHIFT, usable_len - MIN_BLOCK_SHIFT))
        pred = np.roll(PREDICTIONS[i, :usable_len], shift)
        ret = TARGETS[i, BASE_SHIFT:]
        tstats[i] = estimate_kappa_tstat(pred, ret)
    return tstats


def run_replicate(task):
    test_name, rep, seed = task
    rng = np.random.default_rng(seed)
    if test_name == "firm_permutation":
        tstats = firm_permutation_tstats(rng)
    elif test_name == "date_permutation":
        tstats = date_permutation_tstats(rng)
    elif test_name == "circular_block_shift":
        tstats = circular_block_shift_tstats(rng)
    else:
        raise ValueError(f"Unknown test: {test_name}")
    row = summarize_tstats(test_name, rep, seed, tstats)
    return row, tstats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--base-shift", type=int, default=1)
    parser.add_argument("--min-block-shift", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--estimation-dir",
        type=Path,
        default=Path("Results/Estimation/Cross_Sectional"),
    )
    args = parser.parse_args()

    with h5py.File(args.estimation_dir / "betas.h5", "r") as f:
        predictions = f["predictions"][:].astype(float)
        targets = f["targets"][:].astype(float)
        stocks = [s.decode() if isinstance(s, bytes) else str(s) for s in f["stocks"][:]]

    init_worker(predictions, targets, args.base_shift, args.min_block_shift)
    actual_tstats = baseline_tstats()
    actual_row = summarize_tstats("actual", -1, args.seed, actual_tstats)

    rng = np.random.default_rng(args.seed)
    tests = ["firm_permutation", "date_permutation", "circular_block_shift"]
    tasks = []
    for test_name in tests:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=args.n_reps)
        tasks.extend((test_name, rep, int(seeds[rep])) for rep in range(args.n_reps))

    rows = []
    tstat_rows = []
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(
            args.workers,
            initializer=init_worker,
            initargs=(predictions, targets, args.base_shift, args.min_block_shift),
        ) as pool:
            iterator = pool.imap_unordered(run_replicate, tasks)
            for row, tstats in iterator:
                rows.append(row)
                tstat_rows.append((row["test"], row["rep"], tstats))
    else:
        for task in tasks:
            row, tstats = run_replicate(task)
            rows.append(row)
            tstat_rows.append((row["test"], row["rep"], tstats))

    summary = pd.DataFrame(rows).sort_values(["test", "rep"])
    summary_path = args.estimation_dir / "placebo_battery_stage2.csv"
    summary.to_csv(summary_path, index=False)

    actual_path = args.estimation_dir / "placebo_battery_actual_stage2_tstats.csv"
    pd.DataFrame({"stock": stocks, "actual_kappa_tstat": actual_tstats}).to_csv(
        actual_path, index=False
    )

    adjusted_rows = []
    for test_name in tests:
        mats = [x[2] for x in tstat_rows if x[0] == test_name]
        mat = np.vstack(mats)
        pvals = np.full(actual_tstats.shape, np.nan)
        valid_actual = np.isfinite(actual_tstats)
        for j in np.where(valid_actual)[0]:
            placebo_j = mat[:, j]
            valid_placebo = np.isfinite(placebo_j)
            n_placebo = int(valid_placebo.sum())
            if n_placebo > 0:
                pvals[j] = (
                    np.sum(placebo_j[valid_placebo] >= actual_tstats[j]) + 1
                ) / (n_placebo + 1)
        valid_pvals = np.isfinite(pvals)
        adjusted_rows.append(
            {
                "test": f"{test_name}_firm_pvalues",
                "n_firms": int(valid_pvals.sum()),
                "n_p_le_0_05": int(np.sum(pvals[valid_pvals] <= 0.05)),
                "share_p_le_0_05": float(np.mean(pvals[valid_pvals] <= 0.05)),
                "median_p": float(np.nanmedian(pvals)),
                "p10": float(np.nanpercentile(pvals, 10)),
                "p90": float(np.nanpercentile(pvals, 90)),
            }
        )
        pd.DataFrame({"stock": stocks, "randomization_pvalue": pvals}).to_csv(
            args.estimation_dir / f"placebo_battery_{test_name}_firm_pvalues.csv",
            index=False,
        )

    table_rows = []
    for test_name in tests:
        g = summary[summary["test"] == test_name]
        actual_sig = actual_row["sig_kappa_t_gt_1_96"]
        actual_trim = actual_row["sig_kappa_t_gt_1_96_t_le_50"]
        table_rows.append(
            {
                "test": test_name,
                "actual_sig": actual_sig,
                "placebo_mean_sig": g["sig_kappa_t_gt_1_96"].mean(),
                "placebo_p50_sig": g["sig_kappa_t_gt_1_96"].quantile(0.50),
                "placebo_p95_sig": g["sig_kappa_t_gt_1_96"].quantile(0.95),
                "randomization_p_sig": (np.sum(g["sig_kappa_t_gt_1_96"] >= actual_sig) + 1)
                / (len(g) + 1),
                "actual_sig_trim50": actual_trim,
                "placebo_mean_sig_trim50": g["sig_kappa_t_gt_1_96_t_le_50"].mean(),
                "placebo_p95_sig_trim50": g["sig_kappa_t_gt_1_96_t_le_50"].quantile(0.95),
                "randomization_p_sig_trim50": (
                    np.sum(g["sig_kappa_t_gt_1_96_t_le_50"] >= actual_trim) + 1
                )
                / (len(g) + 1),
            }
        )

    table = pd.DataFrame(table_rows)
    table = pd.concat([table, pd.DataFrame(adjusted_rows)], ignore_index=True)
    table_path = args.estimation_dir / "placebo_battery_stage2_table.csv"
    table.to_csv(table_path, index=False)

    print(f"Saved {summary_path.resolve()}")
    print(f"Saved {table_path.resolve()}")
    print("Actual:")
    print(pd.DataFrame([actual_row]).to_string(index=False))
    print("Table:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
