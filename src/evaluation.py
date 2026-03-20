"""
evaluation.py
=============
Unified evaluation utilities for MLP, GNN, and Temporal GNN models.

Public API
----------
    evaluate_gnn(model, test_loader, device, compare_baselines=False, ...)
    evaluate_mlp(model, test_loader, device, compare_baselines=False, ...)
    evaluate_temporal_gnn(model, data_dict, train_years, val_years, test_years,
                          device, compare_baselines=False, ...)

Baselines (enabled when compare_baselines=True)
-----------------------------------------------
    Prev-score  : use each node's previous-year score as the prediction
                  (extracted from scalar[:, 3] = score_d_1).
    Overall mean: use the global mean of all *non-test* years as a constant
                  prediction (no data leakage).

Bootstrap
---------
    When compare_baselines=True each function also runs a paired bootstrap
    test against the overall-mean baseline.  The test set is resampled with
    replacement n_bootstrap times; for each resample the mean of the resampled
    targets is used as the (constant) mean-baseline prediction, while the model
    is evaluated on the same resample.  This produces a distribution of
    bootstrapped metric values for both model and mean baseline.

Backward-compatible aliases
---------------------------
    evaluation              = evaluate_gnn  (no baselines, same signature)
    evaluation_mlp          = evaluate_mlp  (no baselines, same signature)
    evaluate_with_baselines : wrapper keeping the old positional-argument style
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_metrics(predictions, targets):
    """Compute standard regression metrics.

    Returns
    -------
    dict with keys: mse, rmse, mae, mape, r2, median_ae
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    epsilon = 1e-8
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    median_ae = np.median(np.abs(predictions - targets))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "median_ae": float(median_ae),
    }


def _bootstrap_mean_baseline(
    model_preds,
    model_targets,
    n_bootstrap=1000,
    metric="mae",
    seed=42,
):
    """Bootstrap significance test for the overall-mean baseline.

    Resamples the test set with replacement ``n_bootstrap`` times.  For each
    resample the mean of the resampled *targets* is used as the (constant)
    mean-baseline prediction, while the model predictions are evaluated on
    the same resample.  This shows the distribution of the bootstrapped mean
    baseline compared to the model on matching data.

    Parameters
    ----------
    model_preds   : np.ndarray  – model predictions on the test set
    model_targets : np.ndarray  – ground-truth targets
    n_bootstrap   : int         – number of bootstrap iterations (default 1000)
    metric        : str         – 'mae', 'mse', 'rmse', or 'r2'
    seed          : int         – RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        observed_model_metric, observed_baseline_metric, observed_delta,
        boot_model_metrics, boot_baseline_metrics, boot_deltas,
        ci_lower, ci_upper, p_value, metric, n_bootstrap
    """
    rng = np.random.default_rng(seed)
    model_preds = np.asarray(model_preds, dtype=float)
    model_targets = np.asarray(model_targets, dtype=float)

    def _m(preds, tgts):
        if metric == "mae":
            return float(np.mean(np.abs(preds - tgts)))
        if metric == "mse":
            return float(np.mean((preds - tgts) ** 2))
        if metric == "rmse":
            return float(np.sqrt(np.mean((preds - tgts) ** 2)))
        if metric == "r2":
            ss_res = np.sum((tgts - preds) ** 2)
            ss_tot = np.sum((tgts - np.mean(tgts)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        raise ValueError(f"Unknown metric '{metric}'. Choose mae/mse/rmse/r2.")

    global_mean = float(np.mean(model_targets))
    obs_model = _m(model_preds, model_targets)
    obs_baseline = _m(np.full_like(model_targets, global_mean), model_targets)
    # positive delta = model is better (lower error / higher r2)
    obs_delta = obs_baseline - obs_model

    n = len(model_targets)
    boot_model_metrics = np.empty(n_bootstrap)
    boot_base_metrics = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_targets = model_targets[idx]
        b_preds = model_preds[idx]
        # Mean baseline: recompute mean from the resampled targets
        b_mean = float(np.mean(b_targets))
        boot_model_metrics[i] = _m(b_preds, b_targets)
        boot_base_metrics[i] = _m(np.full_like(b_targets, b_mean), b_targets)

    boot_deltas = boot_base_metrics - boot_model_metrics
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    # p-value = fraction of iterations where model does NOT outperform baseline
    if metric == "r2":
        # higher is better; model better when delta < 0
        p_value = float(np.mean(boot_deltas >= 0))
    else:
        # lower is better; model better when delta > 0
        p_value = float(np.mean(boot_deltas <= 0))

    return {
        "observed_model_metric": obs_model,
        "observed_baseline_metric": obs_baseline,
        "observed_delta": obs_delta,
        "boot_model_metrics": boot_model_metrics,
        "boot_baseline_metrics": boot_base_metrics,
        "boot_deltas": boot_deltas,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "metric": metric,
        "n_bootstrap": n_bootstrap,
    }


def _baseline_from_dict(data_dict, test_years):
    """Compute prev-score and overall-mean baselines from a year-keyed data_dict.

    Prev-score baseline
        Uses ``scalar[:, 3]`` (``score_d_1``) stored in each year's Data object,
        restricted to nodes where ``score_d_1 != 0`` (nodes with a known
        previous-year value).

    Overall-mean baseline
        Computes the global mean target over all *non-test* years (no data
        leakage) and uses it as a constant prediction for every test node.

    Returns
    -------
    ps_preds, ps_targets, om_preds, om_targets : np.ndarray (1-D each)
    """
    test_set = set(int(y) for y in test_years)

    train_arrays = [
        data_dict[y].y.cpu().numpy() for y in data_dict if int(y) not in test_set
    ]
    if not train_arrays:
        # Fallback: every year is a test year
        train_arrays = [data_dict[y].y.cpu().numpy() for y in data_dict]
    overall_mean = float(np.mean(np.concatenate(train_arrays)))

    ps_preds_l, ps_targets_l = [], []
    om_preds_l, om_targets_l = [], []

    for year in sorted(test_set):
        if year not in data_dict:
            continue
        current = data_dict[year]
        targets = current.y.cpu().numpy()
        prev_scores = current.scalar[:, 3].cpu().numpy()  # score_d_1

        # Overall-mean baseline: every test node
        om_preds_l.extend([overall_mean] * len(targets))
        om_targets_l.extend(targets.tolist())

        # Prev-score baseline: only nodes with a known previous score
        valid = prev_scores != 0.0
        ps_preds_l.extend(prev_scores[valid].tolist())
        ps_targets_l.extend(targets[valid].tolist())

    return (
        np.array(ps_preds_l, dtype=float),
        np.array(ps_targets_l, dtype=float),
        np.array(om_preds_l, dtype=float),
        np.array(om_targets_l, dtype=float),
    )


def _baseline_from_loader(all_scalars, all_targets, train_mean=None):
    """Compute prev-score and overall-mean baselines from MLP DataLoader outputs.

    Prev-score baseline
        ``all_scalars[:, 3]`` (``score_d_1``), nodes where value != 0.

    Overall-mean baseline
        Uses ``train_mean`` if provided; otherwise falls back to the test-set
        mean with a data-leakage warning.

    Parameters
    ----------
    all_scalars : np.ndarray, shape (N, n_scalar_features)
    all_targets : np.ndarray, shape (N,)  – already in original (non-scaled) space
    train_mean  : float or None – pre-computed mean of training targets

    Returns
    -------
    ps_preds, ps_targets, om_preds, om_targets : np.ndarray (1-D each)
    """
    prev_scores = all_scalars[:, 3]
    valid = prev_scores != 0.0
    ps_preds = prev_scores[valid].astype(float)
    ps_targets = all_targets[valid].astype(float)

    if train_mean is None:
        warnings.warn(
            "train_mean not provided for MLP overall-mean baseline. "
            "Falling back to the test-set mean — this leaks label information.",
            UserWarning,
            stacklevel=3,
        )
        mean_val = float(np.mean(all_targets))
    else:
        mean_val = float(train_mean)

    om_preds = np.full(len(all_targets), mean_val, dtype=float)
    om_targets = all_targets.astype(float)

    return ps_preds, ps_targets, om_preds, om_targets


# ── Print helpers ─────────────────────────────────────────────────────────────


def _print_metrics_block(label, metrics, n_preds):
    print(f"\n{'=' * 70}")
    print(label)
    print("=" * 70)
    print(f"Number of predictions:            {n_preds}")
    print(f"Mean Squared Error (MSE):         {metrics['mse']:.6f}")
    print(f"Root Mean Squared Error (RMSE):   {metrics['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE):        {metrics['mae']:.6f}")
    print(f"Median Absolute Error:            {metrics['median_ae']:.6f}")
    print(f"Mean Absolute Percentage Error:   {metrics['mape']:.2f}%")
    print(f"R² Score:                         {metrics['r2']:.6f}")


def _print_comparison_table(model_metrics, ps_metrics, om_metrics):
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY  (lower is better, except R²)")
    print("=" * 70)

    cols = [("Model", model_metrics)]
    if ps_metrics is not None:
        cols.append(("Prev Score", ps_metrics))
    if om_metrics is not None:
        cols.append(("Overall Mean", om_metrics))

    header = f"{'Metric':<26}" + "".join(f"{c[0]:<15}" for c in cols)
    print(header)
    print("-" * 70)

    for key, name in [
        ("mse", "MSE"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("r2", "R² (higher better)"),
        ("median_ae", "Median AE"),
    ]:
        row = f"{name:<26}" + "".join(f"{c[1][key]:<15.6f}" for c in cols)
        print(row)

    print("\n" + "=" * 70)
    print("MODEL IMPROVEMENT OVER BASELINES  (MAE / RMSE, positive = model better)")
    print("=" * 70)
    if ps_metrics is not None:
        mae_i = (
            (ps_metrics["mae"] - model_metrics["mae"]) / ps_metrics["mae"] * 100
            if ps_metrics["mae"] != 0
            else float("nan")
        )
        rmse_i = (
            (ps_metrics["rmse"] - model_metrics["rmse"]) / ps_metrics["rmse"] * 100
            if ps_metrics["rmse"] != 0
            else float("nan")
        )
        print(f"  vs Prev-Score:   MAE {mae_i:+.2f}%,  RMSE {rmse_i:+.2f}%")
    if om_metrics is not None:
        mae_i = (
            (om_metrics["mae"] - model_metrics["mae"]) / om_metrics["mae"] * 100
            if om_metrics["mae"] != 0
            else float("nan")
        )
        rmse_i = (
            (om_metrics["rmse"] - model_metrics["rmse"]) / om_metrics["rmse"] * 100
            if om_metrics["rmse"] != 0
            else float("nan")
        )
        print(f"  vs Overall Mean: MAE {mae_i:+.2f}%,  RMSE {rmse_i:+.2f}%")


def _print_bootstrap_result(result, label):
    print(f"\n{'=' * 70}")
    print(
        f"BOOTSTRAP TEST  (n={result['n_bootstrap']}, "
        f"metric={result['metric'].upper()})  —  Model vs {label}"
    )
    print("=" * 70)
    print(f"  Observed model metric:    {result['observed_model_metric']:.6f}")
    print(f"  Observed baseline metric: {result['observed_baseline_metric']:.6f}")
    print(
        f"  Observed delta:           {result['observed_delta']:+.6f}  (baseline − model)"
    )
    print(
        f"  95% CI for delta:         "
        f"[{result['ci_lower']:+.6f}, {result['ci_upper']:+.6f}]"
    )
    print(f"  p-value:                  {result['p_value']:.4f}")
    sig = "significant" if result["p_value"] < 0.05 else "NOT significant"
    print(f"  Conclusion (α=0.05):      improvement is {sig}")


# ── Plot helpers ──────────────────────────────────────────────────────────────


# Base font sizes; multiply by font_scale at call sites.
_BASE_TITLE = 13
_BASE_LABEL = 11
_BASE_TICK = 10
_BASE_LEGEND = 9


def _plot_simple(targets, predictions, metrics, title="Model", font_scale=1.0):
    """Scatter + residual plot (used when compare_baselines=False)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lims = [
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    ]
    axes[0].scatter(targets, predictions, alpha=0.5, s=10)
    axes[0].plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Values", fontsize=_BASE_LABEL * font_scale)
    axes[0].set_ylabel("Predicted Values", fontsize=_BASE_LABEL * font_scale)
    axes[0].set_title(
        f"{title}  (R\u00b2={metrics['r2']:.4f})", fontsize=_BASE_TITLE * font_scale
    )
    axes[0].legend(fontsize=_BASE_LEGEND * font_scale)
    axes[0].tick_params(labelsize=_BASE_TICK * font_scale)
    axes[0].grid(True, alpha=0.3)

    residuals = targets - predictions
    axes[1].scatter(predictions, residuals, alpha=0.5, s=10)
    axes[1].axhline(0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Values", fontsize=_BASE_LABEL * font_scale)
    axes[1].set_ylabel(
        "Residuals  (Actual − Predicted)", fontsize=_BASE_LABEL * font_scale
    )
    axes[1].set_title("Residual Plot", fontsize=_BASE_TITLE * font_scale)
    axes[1].tick_params(labelsize=_BASE_TICK * font_scale)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_comparison_grid(
    model_targets,
    model_preds,
    model_metrics,
    ps_targets,
    ps_preds,
    ps_metrics,
    om_targets,
    om_preds,
    om_metrics,
    bootstrap_result,
    font_scale=1.0,
):
    """2×2 comparison grid.

    [0,0] Model predicted vs actual scatter
    [0,1] Prev-score baseline scatter
    [1,0] Side-by-side error-metric bar chart (model vs both baselines)
    [1,1] Bootstrap histogram: model metric distribution vs mean-baseline distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ── [0,0] Model scatter ───────────────────────────────────────────────────
    lims = [
        min(model_targets.min(), model_preds.min()),
        max(model_targets.max(), model_preds.max()),
    ]
    axes[0, 0].scatter(model_targets, model_preds, alpha=0.5, s=10)
    axes[0, 0].plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    axes[0, 0].set_xlabel("Actual Values", fontsize=_BASE_LABEL * font_scale)
    axes[0, 0].set_ylabel("Predicted Values", fontsize=_BASE_LABEL * font_scale)
    axes[0, 0].set_title(
        f"Model  (R²={model_metrics['r2']:.4f})", fontsize=_BASE_TITLE * font_scale
    )
    axes[0, 0].legend(fontsize=_BASE_LEGEND * font_scale)
    axes[0, 0].tick_params(labelsize=_BASE_TICK * font_scale)
    axes[0, 0].grid(True, alpha=0.3)

    # ── [0,1] Prev-score baseline scatter ────────────────────────────────────
    if ps_metrics is not None and len(ps_targets) > 0:
        ps_lims = [
            min(ps_targets.min(), ps_preds.min()),
            max(ps_targets.max(), ps_preds.max()),
        ]
        axes[0, 1].scatter(
            ps_targets, ps_preds, alpha=0.5, s=10, color="orange", label="Prev Score"
        )
        axes[0, 1].plot(ps_lims, ps_lims, "r--", lw=2, label="Perfect Prediction")
        axes[0, 1].set_xlabel("Actual Values", fontsize=_BASE_LABEL * font_scale)
        axes[0, 1].set_ylabel("Predicted Values", fontsize=_BASE_LABEL * font_scale)
        axes[0, 1].set_title(
            f"Baseline: Prev Score  (R²={ps_metrics['r2']:.4f})",
            fontsize=_BASE_TITLE * font_scale,
        )
        axes[0, 1].legend(fontsize=_BASE_LEGEND * font_scale)
        axes[0, 1].tick_params(labelsize=_BASE_TICK * font_scale)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].set_visible(False)

    # ── [1,0] Error metric bar chart ─────────────────────────────────────────
    metric_names = ["MSE", "RMSE", "MAE", "Median AE"]
    metric_keys = ["mse", "rmse", "mae", "median_ae"]
    bar_data = [("Model", [model_metrics[k] for k in metric_keys], "steelblue")]
    if ps_metrics is not None:
        bar_data.append(("Prev Score", [ps_metrics[k] for k in metric_keys], "orange"))
    if om_metrics is not None:
        bar_data.append(
            ("Overall Mean", [om_metrics[k] for k in metric_keys], "purple")
        )

    x = np.arange(len(metric_names))
    n_bars = len(bar_data)
    width = 0.7 / n_bars
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    for (lbl, vals, col), off in zip(bar_data, offsets):
        axes[1, 0].bar(x + off, vals, width, label=lbl, color=col)

    axes[1, 0].set_xlabel("Metric", fontsize=_BASE_LABEL * font_scale)
    axes[1, 0].set_ylabel("Error Value", fontsize=_BASE_LABEL * font_scale)
    axes[1, 0].set_title(
        "Error Metrics Comparison  (Lower is Better)", fontsize=_BASE_TITLE * font_scale
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metric_names, fontsize=_BASE_TICK * font_scale)
    axes[1, 0].tick_params(labelsize=_BASE_TICK * font_scale)
    axes[1, 0].legend(fontsize=_BASE_LEGEND * font_scale)
    axes[1, 0].grid(True, alpha=0.3)

    # ── [1,1] Bootstrap histogram ─────────────────────────────────────────────
    if bootstrap_result is not None:
        r = bootstrap_result
        ax = axes[1, 1]
        ax.hist(
            r["boot_baseline_metrics"],
            bins=50,
            alpha=0.6,
            color="purple",
            label="Bootstrapped mean baseline",
        )
        ax.hist(
            r["boot_model_metrics"],
            bins=50,
            alpha=0.6,
            color="steelblue",
            label="Bootstrapped model",
        )
        ax.axvline(
            r["observed_baseline_metric"],
            color="purple",
            linestyle="--",
            lw=2,
            label=(
                f"Obs. baseline {r['metric'].upper()}"
                f"={r['observed_baseline_metric']:.4f}"
            ),
        )
        ax.axvline(
            r["observed_model_metric"],
            color="steelblue",
            linestyle="--",
            lw=2,
            label=(
                f"Obs. model {r['metric'].upper()}={r['observed_model_metric']:.4f}"
            ),
        )
        ax.set_xlabel(
            f"Bootstrapped {r['metric'].upper()}", fontsize=_BASE_LABEL * font_scale
        )
        ax.set_ylabel("Frequency", fontsize=_BASE_LABEL * font_scale)
        ax.set_title(
            f"Bootstrap: Model vs Overall-Mean Baseline\n"
            f"n={r['n_bootstrap']},  p={r['p_value']:.4f}",
            fontsize=_BASE_TITLE * font_scale,
        )
        ax.legend(fontsize=_BASE_LEGEND * font_scale)
        ax.tick_params(labelsize=_BASE_TICK * font_scale)
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.show()


def _plot_scatter(targets, predictions, metrics, title="Model", font_scale=1.0):
    """Standalone scatter of predicted vs actual.  Returns the figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    lims = [
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    ]
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Values", fontsize=_BASE_LABEL * font_scale)
    ax.set_ylabel("Predicted Values", fontsize=_BASE_LABEL * font_scale)
    ax.set_title(
        f"{title}  (R\u00b2={metrics['r2']:.4f})", fontsize=_BASE_TITLE * font_scale
    )
    ax.legend(fontsize=_BASE_LEGEND * font_scale)
    ax.tick_params(labelsize=_BASE_TICK * font_scale)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_metrics_bar(model_metrics, ps_metrics, om_metrics, font_scale=1.0):
    """Standalone grouped bar chart for error metric comparison.  Returns the figure."""
    metric_names = ["MSE", "RMSE", "MAE", "Median AE"]
    metric_keys = ["mse", "rmse", "mae", "median_ae"]
    bar_data = [("Model", [model_metrics[k] for k in metric_keys], "steelblue")]
    if ps_metrics is not None:
        bar_data.append(("Prev Score", [ps_metrics[k] for k in metric_keys], "orange"))
    if om_metrics is not None:
        bar_data.append(
            ("Overall Mean", [om_metrics[k] for k in metric_keys], "purple")
        )

    x = np.arange(len(metric_names))
    n_bars = len(bar_data)
    width = 0.7 / n_bars
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig, ax = plt.subplots(figsize=(8, 5))
    for (lbl, vals, col), off in zip(bar_data, offsets):
        ax.bar(x + off, vals, width, label=lbl, color=col)

    ax.set_xlabel("Metric", fontsize=_BASE_LABEL * font_scale)
    ax.set_ylabel("Error Value", fontsize=_BASE_LABEL * font_scale)
    ax.set_title(
        "Error Metrics Comparison  (Lower is Better)", fontsize=_BASE_TITLE * font_scale
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=_BASE_TICK * font_scale)
    ax.tick_params(labelsize=_BASE_TICK * font_scale)
    ax.legend(fontsize=_BASE_LEGEND * font_scale)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_bootstrap_standalone(bootstrap_result, font_scale=1.0):
    """Standalone bootstrap histogram.  Returns the figure."""
    r = bootstrap_result
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        r["boot_baseline_metrics"],
        bins=50,
        alpha=0.6,
        color="purple",
        label="Bootstrapped mean baseline",
    )
    ax.hist(
        r["boot_model_metrics"],
        bins=50,
        alpha=0.6,
        color="steelblue",
        label="Bootstrapped model",
    )
    ax.axvline(
        r["observed_baseline_metric"],
        color="purple",
        linestyle="--",
        lw=2,
        label=f"Obs. baseline {r['metric'].upper()}={r['observed_baseline_metric']:.4f}",
    )
    ax.axvline(
        r["observed_model_metric"],
        color="steelblue",
        linestyle="--",
        lw=2,
        label=f"Obs. model {r['metric'].upper()}={r['observed_model_metric']:.4f}",
    )
    ax.set_xlabel(
        f"Bootstrapped {r['metric'].upper()}", fontsize=_BASE_LABEL * font_scale
    )
    ax.set_ylabel("Frequency", fontsize=_BASE_LABEL * font_scale)
    ax.set_title(
        f"Bootstrap: Model vs Overall-Mean Baseline\n"
        f"n={r['n_bootstrap']},  p={r['p_value']:.4f}",
        fontsize=_BASE_TITLE * font_scale,
    )
    ax.legend(fontsize=_BASE_LEGEND * font_scale)
    ax.tick_params(labelsize=_BASE_TICK * font_scale)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _save_fig(fig, path):
    """Save *fig* to *path* at 150 dpi."""
    fig.savefig(path, bbox_inches="tight", dpi=150)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_gnn(
    model,
    test_loader,
    device,
    compare_baselines=False,
    data_dict=None,
    test_years=None,
    n_bootstrap=1000,
    export=False,
    export_dir=None,
    font_scale=1.0,
    printing=True,
):
    """Evaluate a standard GNN model on a test DataLoader.

    Parameters
    ----------
    model             : Trained GNN model.
    test_loader       : PyG DataLoader for the test set.
    device            : torch.device.
    compare_baselines : If True, compute prev-score and overall-mean baselines
                        and run a bootstrap significance test vs the overall-mean
                        baseline.  Requires ``data_dict`` and ``test_years``.
    data_dict         : dict mapping year → PyG Data object.
                        Required when compare_baselines=True.
    test_years        : list/set of test years.
                        Required when compare_baselines=True.
    n_bootstrap       : Number of bootstrap iterations (default 1000).
    printing          : Whether to print metrics and show plots.

    Returns
    -------
    dict with keys:
        'model'                 : {predictions, targets, years, metrics}
        'prev_score_baseline'   : {predictions, targets, metrics} | None
        'overall_mean_baseline' : {predictions, targets, metrics, bootstrap} | None
    """
    model.eval()
    raw_preds, raw_targets, raw_years = [], [], []

    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            out = model(data)
            raw_preds.append(out.view(-1).cpu())
            raw_targets.append(data.y.view(-1).cpu())
            if hasattr(data, "year"):
                raw_years.append(data.year.view(-1).cpu())

    model_preds = torch.cat(raw_preds).numpy()
    model_targets = torch.cat(raw_targets).numpy()
    year_arr = torch.cat(raw_years).numpy() if raw_years else np.array([])

    model_metrics = _compute_metrics(model_preds, model_targets)

    ps_preds = ps_targets = om_preds = om_targets = np.array([])
    ps_metrics = om_metrics = bootstrap_result = None

    if compare_baselines:
        if data_dict is None or test_years is None:
            raise ValueError(
                "compare_baselines=True requires data_dict and test_years."
            )
        ps_preds, ps_targets, om_preds, om_targets = _baseline_from_dict(
            data_dict, test_years
        )
        if len(ps_preds) > 0:
            ps_metrics = _compute_metrics(ps_preds, ps_targets)
        if len(om_preds) > 0:
            om_metrics = _compute_metrics(om_preds, om_targets)
            bootstrap_result = _bootstrap_mean_baseline(
                model_preds, model_targets, n_bootstrap=n_bootstrap
            )

    if printing:
        _print_metrics_block("MODEL: GNN", model_metrics, len(model_preds))
        if compare_baselines:
            if ps_metrics is not None:
                _print_metrics_block(
                    "BASELINE: Previous Year's Score (Persistence)",
                    ps_metrics,
                    len(ps_preds),
                )
            if om_metrics is not None:
                _print_metrics_block(
                    "BASELINE: Overall Mean (Global Historical Mean)",
                    om_metrics,
                    len(om_preds),
                )
            _print_comparison_table(model_metrics, ps_metrics, om_metrics)
            if bootstrap_result is not None:
                _print_bootstrap_result(bootstrap_result, "Overall-Mean Baseline")

    if export:
        resolved_dir = (
            export_dir
            if export_dir is not None
            else os.path.join(os.getcwd(), "eval_plots")
        )
        os.makedirs(resolved_dir, exist_ok=True)

        fig_scatter = _plot_scatter(
            model_targets,
            model_preds,
            model_metrics,
            title="GNN",
            font_scale=font_scale,
        )
        if printing:
            plt.show()
        _save_fig(fig_scatter, os.path.join(resolved_dir, "pred_scatter.png"))
        plt.close(fig_scatter)

        if compare_baselines and (ps_metrics is not None or om_metrics is not None):
            fig_bar = _plot_metrics_bar(
                model_metrics, ps_metrics, om_metrics, font_scale=font_scale
            )
            if printing:
                plt.show()
            _save_fig(fig_bar, os.path.join(resolved_dir, "error_metrics.png"))
            plt.close(fig_bar)

        if bootstrap_result is not None:
            fig_boot = _plot_bootstrap_standalone(
                bootstrap_result, font_scale=font_scale
            )
            if printing:
                plt.show()
            _save_fig(fig_boot, os.path.join(resolved_dir, "bootstrap.png"))
            plt.close(fig_boot)
    elif printing:
        if compare_baselines and (ps_metrics is not None or om_metrics is not None):
            _plot_comparison_grid(
                model_targets,
                model_preds,
                model_metrics,
                ps_targets,
                ps_preds,
                ps_metrics,
                om_targets,
                om_preds,
                om_metrics,
                bootstrap_result,
                font_scale=font_scale,
            )
        else:
            _plot_simple(
                model_targets,
                model_preds,
                model_metrics,
                title="GNN",
                font_scale=font_scale,
            )

    return {
        "model": {
            "predictions": model_preds,
            "targets": model_targets,
            "years": year_arr,
            "metrics": model_metrics,
        },
        "prev_score_baseline": {
            "predictions": ps_preds,
            "targets": ps_targets,
            "metrics": ps_metrics,
        }
        if ps_metrics is not None
        else None,
        "overall_mean_baseline": {
            "predictions": om_preds,
            "targets": om_targets,
            "metrics": om_metrics,
            "bootstrap": bootstrap_result,
        }
        if om_metrics is not None
        else None,
    }


def evaluate_mlp(
    model,
    test_loader,
    device,
    compare_baselines=False,
    train_mean=None,
    target_scaler=None,
    n_bootstrap=1000,
    export=False,
    export_dir=None,
    font_scale=1.0,
    printing=True,
):
    """Evaluate a standard MLP model on a test DataLoader.

    The DataLoader must yield ``(scalar, embedding, target, year)`` tuples.

    Parameters
    ----------
    model             : Trained MLP model.
    test_loader       : DataLoader yielding (scalar, embedding, target, year).
    device            : torch.device.
    compare_baselines : If True, compute prev-score and overall-mean baselines
                        and run a bootstrap significance test.
                        The prev-score baseline uses ``scalar[:, 3]``
                        (``score_d_1``).  Pass ``train_mean`` to avoid data
                        leakage in the overall-mean baseline.
    train_mean        : Pre-computed mean of *training* targets used for the
                        overall-mean baseline.  Should be in the same (original,
                        non-scaled) space as the targets.
    target_scaler     : Optional scaler with ``inverse_transform`` method.
                        Applied to both model outputs and targets before any
                        metric is computed.
    n_bootstrap       : Number of bootstrap iterations (default 1000).
    printing          : Whether to print metrics and show plots.

    Returns
    -------
    dict with keys:
        'model'                 : {predictions, targets, metrics}
        'prev_score_baseline'   : {predictions, targets, metrics} | None
        'overall_mean_baseline' : {predictions, targets, metrics, bootstrap} | None
    """
    model.eval()
    raw_preds, raw_targets, raw_scalars = [], [], []

    with torch.no_grad():
        for scalar, embedding, target, _year in test_loader:
            scalar = scalar.to(device)
            embedding = embedding.to(device)
            target = target.to(device)
            out = model(scalar, embedding)
            raw_preds.append(out.cpu())
            raw_targets.append(target.cpu())
            if compare_baselines:
                raw_scalars.append(scalar.cpu())

    model_preds = torch.cat(raw_preds).squeeze().numpy()
    model_targets = torch.cat(raw_targets).numpy()

    if target_scaler is not None:
        model_preds = target_scaler.inverse_transform(
            model_preds.reshape(-1, 1)
        ).flatten()
        model_targets = target_scaler.inverse_transform(
            model_targets.reshape(-1, 1)
        ).flatten()

    model_metrics = _compute_metrics(model_preds, model_targets)

    ps_preds = ps_targets = om_preds = om_targets = np.array([])
    ps_metrics = om_metrics = bootstrap_result = None

    if compare_baselines:
        all_scalars = torch.cat(raw_scalars).numpy()
        ps_preds, ps_targets, om_preds, om_targets = _baseline_from_loader(
            all_scalars, model_targets, train_mean=train_mean
        )
        if len(ps_preds) > 0:
            ps_metrics = _compute_metrics(ps_preds, ps_targets)
        if len(om_preds) > 0:
            om_metrics = _compute_metrics(om_preds, om_targets)
            bootstrap_result = _bootstrap_mean_baseline(
                model_preds, model_targets, n_bootstrap=n_bootstrap
            )

    if printing:
        _print_metrics_block("MODEL: MLP", model_metrics, len(model_preds))
        if compare_baselines:
            if ps_metrics is not None:
                _print_metrics_block(
                    "BASELINE: Previous Year's Score (Persistence)",
                    ps_metrics,
                    len(ps_preds),
                )
            if om_metrics is not None:
                _print_metrics_block(
                    "BASELINE: Overall Mean",
                    om_metrics,
                    len(om_preds),
                )
            _print_comparison_table(model_metrics, ps_metrics, om_metrics)
            if bootstrap_result is not None:
                _print_bootstrap_result(bootstrap_result, "Overall-Mean Baseline")

    if export:
        resolved_dir = (
            export_dir
            if export_dir is not None
            else os.path.join(os.getcwd(), "eval_plots")
        )
        os.makedirs(resolved_dir, exist_ok=True)

        fig_scatter = _plot_scatter(
            model_targets,
            model_preds,
            model_metrics,
            title="MLP",
            font_scale=font_scale,
        )
        if printing:
            plt.show()
        _save_fig(fig_scatter, os.path.join(resolved_dir, "pred_scatter.png"))
        plt.close(fig_scatter)

        if compare_baselines and (ps_metrics is not None or om_metrics is not None):
            fig_bar = _plot_metrics_bar(
                model_metrics, ps_metrics, om_metrics, font_scale=font_scale
            )
            if printing:
                plt.show()
            _save_fig(fig_bar, os.path.join(resolved_dir, "error_metrics.png"))
            plt.close(fig_bar)

        if bootstrap_result is not None:
            fig_boot = _plot_bootstrap_standalone(
                bootstrap_result, font_scale=font_scale
            )
            if printing:
                plt.show()
            _save_fig(fig_boot, os.path.join(resolved_dir, "bootstrap.png"))
            plt.close(fig_boot)
    elif printing:
        if compare_baselines and (ps_metrics is not None or om_metrics is not None):
            _plot_comparison_grid(
                model_targets,
                model_preds,
                model_metrics,
                ps_targets,
                ps_preds,
                ps_metrics,
                om_targets,
                om_preds,
                om_metrics,
                bootstrap_result,
                font_scale=font_scale,
            )
        else:
            _plot_simple(
                model_targets,
                model_preds,
                model_metrics,
                title="MLP",
                font_scale=font_scale,
            )

    return {
        "model": {
            "predictions": model_preds,
            "targets": model_targets,
            "metrics": model_metrics,
        },
        "prev_score_baseline": {
            "predictions": ps_preds,
            "targets": ps_targets,
            "metrics": ps_metrics,
        }
        if ps_metrics is not None
        else None,
        "overall_mean_baseline": {
            "predictions": om_preds,
            "targets": om_targets,
            "metrics": om_metrics,
            "bootstrap": bootstrap_result,
        }
        if om_metrics is not None
        else None,
    }


def evaluate_temporal_gnn(
    model,
    data_dict,
    train_years,
    val_years,
    test_years,
    device,
    compare_baselines=False,
    n_bootstrap=1000,
    export=False,
    export_dir=None,
    font_scale=1.0,
    printing=True,
):
    """Evaluate a TemporalGATGRU on the test set with proper GRU warm-up.

    The GRU hidden state is warmed up by running forward passes (without
    collecting predictions) through all train and validation years in
    chronological order.  Predictions are then collected for test years,
    restricted to active nodes via ``data.node_available``.

    Parameters
    ----------
    model             : Trained TemporalGATGRU instance.
    data_dict         : dict mapping year → PyG Data object (fixed-node graph).
    train_years       : Years used during training (for warm-up).
    val_years         : Years used during validation (for warm-up).
    test_years        : Held-out years to evaluate on.
    device            : torch.device.
    compare_baselines : If True, compute prev-score and overall-mean baselines
                        and run a bootstrap significance test vs the overall-mean
                        baseline.
    n_bootstrap       : Number of bootstrap iterations (default 1000).
    printing          : Whether to print metrics and show plots.

    Returns
    -------
    dict with keys:
        'model'                 : {predictions, targets, years, metrics}
        'prev_score_baseline'   : {predictions, targets, metrics} | None
        'overall_mean_baseline' : {predictions, targets, metrics, bootstrap} | None
    """
    model.eval()
    num_nodes = next(iter(data_dict.values())).x.shape[0]
    warmup_years = sorted(
        y for y in list(train_years) + list(val_years) if y in data_dict
    )
    sorted_test = sorted(y for y in test_years if y in data_dict)

    all_preds_list, all_targets_list, all_years_list = [], [], []

    with torch.no_grad():
        # Warm-up: propagate hidden state through train + val in order
        h = model.init_hidden(num_nodes, device)
        for year in warmup_years:
            _, h = model(data_dict[year].to(device), h)

        # Test inference: collect predictions for active nodes only
        for year in sorted_test:
            data = data_dict[year].to(device)
            mask = data.node_available
            out, h = model(data, h)
            all_preds_list.extend(out.view(-1)[mask].cpu().numpy())
            all_targets_list.extend(data.y[mask].cpu().numpy())
            all_years_list.extend([year] * mask.sum().item())

    model_preds = np.array(all_preds_list, dtype=float)
    model_targets = np.array(all_targets_list, dtype=float)
    year_arr = np.array(all_years_list)

    model_metrics = _compute_metrics(model_preds, model_targets)

    ps_preds = ps_targets = om_preds = om_targets = np.array([])
    ps_metrics = om_metrics = bootstrap_result = None

    if compare_baselines:
        # Temporal graphs contain zero-padded inactive nodes.
        # Keep baseline evaluation on the same active-node subset as the model.
        test_set = set(int(y) for y in sorted_test)

        train_arrays = []
        for year in data_dict:
            if int(year) in test_set:
                continue
            d_train = data_dict[year]
            train_mask = (
                d_train.node_available
                if hasattr(d_train, "node_available")
                else torch.ones(
                    d_train.y.shape[0], dtype=torch.bool, device=d_train.y.device
                )
            )
            train_arrays.append(d_train.y[train_mask].cpu().numpy())

        if not train_arrays:
            # Fallback: every year is a test year
            for year in sorted_test:
                if year not in data_dict:
                    continue
                d_train = data_dict[year]
                train_mask = (
                    d_train.node_available
                    if hasattr(d_train, "node_available")
                    else torch.ones(
                        d_train.y.shape[0], dtype=torch.bool, device=d_train.y.device
                    )
                )
                train_arrays.append(d_train.y[train_mask].cpu().numpy())

        overall_mean = float(np.mean(np.concatenate(train_arrays)))

        ps_preds_l, ps_targets_l = [], []
        om_preds_l, om_targets_l = [], []

        for year in sorted_test:
            if year not in data_dict:
                continue
            d_test = data_dict[year]
            test_mask = (
                d_test.node_available
                if hasattr(d_test, "node_available")
                else torch.ones(
                    d_test.y.shape[0], dtype=torch.bool, device=d_test.y.device
                )
            )

            active_targets = d_test.y[test_mask].cpu().numpy()
            om_targets_l.extend(active_targets.tolist())
            om_preds_l.extend([overall_mean] * len(active_targets))

            prev_scores = d_test.scalar[:, 3][test_mask].cpu().numpy()
            valid = prev_scores != 0.0
            ps_preds_l.extend(prev_scores[valid].tolist())
            ps_targets_l.extend(active_targets[valid].tolist())

        ps_preds = np.array(ps_preds_l, dtype=float)
        ps_targets = np.array(ps_targets_l, dtype=float)
        om_preds = np.array(om_preds_l, dtype=float)
        om_targets = np.array(om_targets_l, dtype=float)

        if len(ps_preds) > 0:
            ps_metrics = _compute_metrics(ps_preds, ps_targets)
        if len(om_preds) > 0:
            om_metrics = _compute_metrics(om_preds, om_targets)
            bootstrap_result = _bootstrap_mean_baseline(
                model_preds, model_targets, n_bootstrap=n_bootstrap
            )

    if printing:
        _print_metrics_block("MODEL: TemporalGATGRU", model_metrics, len(model_preds))
        if compare_baselines:
            if ps_metrics is not None:
                _print_metrics_block(
                    "BASELINE: Previous Year's Score (Persistence)",
                    ps_metrics,
                    len(ps_preds),
                )
            if om_metrics is not None:
                _print_metrics_block(
                    "BASELINE: Overall Mean (Global Historical Mean)",
                    om_metrics,
                    len(om_preds),
                )
            _print_comparison_table(model_metrics, ps_metrics, om_metrics)
            if bootstrap_result is not None:
                _print_bootstrap_result(bootstrap_result, "Overall-Mean Baseline")

    if export:
        resolved_dir = (
            export_dir
            if export_dir is not None
            else os.path.join(os.getcwd(), "eval_plots")
        )
        os.makedirs(resolved_dir, exist_ok=True)

        fig_scatter = _plot_scatter(
            model_targets,
            model_preds,
            model_metrics,
            title="TemporalGATGRU",
            font_scale=font_scale,
        )
        if printing:
            plt.show()
        _save_fig(fig_scatter, os.path.join(resolved_dir, "pred_scatter.png"))
        plt.close(fig_scatter)

        if compare_baselines and (ps_metrics is not None or om_metrics is not None):
            fig_bar = _plot_metrics_bar(
                model_metrics, ps_metrics, om_metrics, font_scale=font_scale
            )
            if printing:
                plt.show()
            _save_fig(fig_bar, os.path.join(resolved_dir, "error_metrics.png"))
            plt.close(fig_bar)

        if bootstrap_result is not None:
            fig_boot = _plot_bootstrap_standalone(
                bootstrap_result, font_scale=font_scale
            )
            if printing:
                plt.show()
            _save_fig(fig_boot, os.path.join(resolved_dir, "bootstrap.png"))
            plt.close(fig_boot)
    elif printing:
        if compare_baselines and (ps_metrics is not None or om_metrics is not None):
            _plot_comparison_grid(
                model_targets,
                model_preds,
                model_metrics,
                ps_targets,
                ps_preds,
                ps_metrics,
                om_targets,
                om_preds,
                om_metrics,
                bootstrap_result,
                font_scale=font_scale,
            )
        else:
            # Per-year scatter + MSE bar chart (original temporal-GNN view)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            lims = [
                min(model_targets.min(), model_preds.min()),
                max(model_targets.max(), model_preds.max()),
            ]
            axes[0].scatter(model_targets, model_preds, alpha=0.3, s=10)
            axes[0].plot(lims, lims, "r--", lw=1)
            axes[0].set_xlabel("True score", fontsize=_BASE_LABEL * font_scale)
            axes[0].set_ylabel("Predicted score", fontsize=_BASE_LABEL * font_scale)
            axes[0].set_title(
                f"TemporalGATGRU  R²={model_metrics['r2']:.3f}",
                fontsize=_BASE_TITLE * font_scale,
            )
            axes[0].tick_params(labelsize=_BASE_TICK * font_scale)

            year_mse = {}
            for year in sorted_test:
                idxs = np.where(year_arr == year)[0]
                if len(idxs) > 0:
                    year_mse[year] = float(
                        np.mean((model_preds[idxs] - model_targets[idxs]) ** 2)
                    )
            axes[1].bar(
                [str(y) for y in sorted_test],
                [year_mse.get(y, 0) for y in sorted_test],
            )
            axes[1].set_xlabel("Year", fontsize=_BASE_LABEL * font_scale)
            axes[1].set_ylabel("MSE", fontsize=_BASE_LABEL * font_scale)
            axes[1].set_title(
                "Per-year MSE on test set", fontsize=_BASE_TITLE * font_scale
            )
            axes[1].tick_params(labelsize=_BASE_TICK * font_scale)

            plt.tight_layout()
            plt.show()

    return {
        "model": {
            "predictions": model_preds,
            "targets": model_targets,
            "years": year_arr,
            "metrics": model_metrics,
        },
        "prev_score_baseline": {
            "predictions": ps_preds,
            "targets": ps_targets,
            "metrics": ps_metrics,
        }
        if ps_metrics is not None
        else None,
        "overall_mean_baseline": {
            "predictions": om_preds,
            "targets": om_targets,
            "metrics": om_metrics,
            "bootstrap": bootstrap_result,
        }
        if om_metrics is not None
        else None,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# evaluate_gnn with compare_baselines=False is identical to the old `evaluation`
evaluation = evaluate_gnn

# evaluation_mlp with compare_baselines=False is identical to the old name
evaluation_mlp = evaluate_mlp


def evaluate_with_baselines(
    model, test_loader, data_dict, test_years, device, printing=True
):
    """Backward-compatible wrapper — keeps the old positional-argument style.

    Equivalent to: evaluate_gnn(model, test_loader, device,
                                 compare_baselines=True,
                                 data_dict=data_dict, test_years=test_years,
                                 printing=printing)
    """
    return evaluate_gnn(
        model,
        test_loader,
        device,
        compare_baselines=True,
        data_dict=data_dict,
        test_years=test_years,
        printing=printing,
    )
