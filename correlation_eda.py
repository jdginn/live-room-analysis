"""
Correlation / EDA Module for Experiment Metrics Analysis.

This module provides functions to:
- Load CSV data from experiments
- Compute Pearson and Spearman correlations between input features and a selected figure of merit (FOM)
- Visualize correlations with bar plots and heatmaps

Usage as a script:
    python correlation_eda.py --csv experiment_metrics.csv --fom DrumDeadT30

Usage as a module:
    from correlation_eda import load_data, compute_correlations, plot_correlation_bars
    df = load_data('experiment_metrics.csv')
    correlations = compute_correlations(df, fom='DrumDeadT30')
    plot_correlation_bars(correlations, fom='DrumDeadT30')
"""

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Default input and derived feature columns
INPUT_FEATURES = [
    "l_num_reflectors",
    "l_reflector_angle",
    "l_reflector_depth",
    "l_start_offset",
    "l_finish_offset",
    "r_num_reflectors",
    "r_reflector_angle",
    "r_reflector_depth",
    "r_start_offset",
    "r_finish_offset",
]

DERIVED_FEATURES = [
    "avg_angle",
    "angle_diff",
    "total_depth",
    "avg_depth",
    "num_total",
    "num_diff",
    "spacing_asym",
]

# All features for correlation analysis
ALL_FEATURES = INPUT_FEATURES + DERIVED_FEATURES

# Output metrics (possible FOMs)
OUTPUT_METRICS = [
    "DrumDeadT30",
    "DrumLiveT30",
    "DrumDeltaT30",
    "DrumDiffusion",
    "VocalDeadT30",
    "VocalDoorToWindowT30",
    "VocalCenterToWindowT30",
    "VocalDiffT30",
    "VocalDiffusion",
]


def load_data(csv_path: str = "experiment_metrics.csv") -> pd.DataFrame:
    """
    Load experiment metrics from a CSV file.

    Args:
        csv_path: Path to the CSV file containing experiment data.

    Returns:
        DataFrame with experiment data.
    """
    df = pd.read_csv(csv_path)
    return df


def get_available_foms(df: pd.DataFrame) -> list:
    """
    Get list of available figure of merit (FOM) columns in the dataframe.

    Args:
        df: DataFrame with experiment data.

    Returns:
        List of column names that can be used as FOMs.
    """
    available = [col for col in OUTPUT_METRICS if col in df.columns]
    return available


def compute_correlations(
    df: pd.DataFrame,
    fom: str,
    features: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between features and a figure of merit.

    Args:
        df: DataFrame with experiment data.
        fom: Name of the figure of merit column to correlate against.
        features: List of feature column names. If None, uses ALL_FEATURES.

    Returns:
        DataFrame with columns ['feature', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p',
        'abs_pearson', 'abs_spearman'] sorted by absolute Pearson correlation (descending).
    """
    if features is None:
        features = [f for f in ALL_FEATURES if f in df.columns]

    if fom not in df.columns:
        raise ValueError(
            f"FOM '{fom}' not found in dataframe. Available columns: {list(df.columns)}"
        )

    results = []

    for feature in features:
        if feature not in df.columns:
            continue

        # Get valid pairs (non-NaN for both feature and FOM)
        valid_mask = df[feature].notna() & df[fom].notna()
        feat_values = df.loc[valid_mask, feature]
        fom_vals = df.loc[valid_mask, fom]

        if len(feat_values) < 3:
            # Not enough data points for correlation
            continue

        # Compute Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(feat_values, fom_vals)

        # Compute Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(feat_values, fom_vals)

        results.append(
            {
                "feature": feature,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "abs_pearson": abs(pearson_r),
                "abs_spearman": abs(spearman_r),
            }
        )

    results_df = pd.DataFrame(results)

    # Sort by absolute Pearson correlation (descending)
    if not results_df.empty:
        results_df = results_df.sort_values("abs_pearson", ascending=False).reset_index(
            drop=True
        )

    return results_df


def print_correlations(
    correlations: pd.DataFrame, fom: str, top_n: Optional[int] = None
) -> None:
    """
    Print correlation results in a formatted table.

    Args:
        correlations: DataFrame from compute_correlations.
        fom: Name of the FOM (for display purposes).
        top_n: If specified, only show top N features by absolute correlation.
    """
    print(f"\n{'='*70}")
    print(f"Correlation Analysis: Features vs {fom}")
    print(f"{'='*70}")

    if correlations.empty:
        print("No correlation data available.")
        return

    display_df = correlations.copy()
    if top_n is not None:
        display_df = display_df.head(top_n)

    print(
        f"\n{'Feature':<25} {'Pearson r':>12} {'p-value':>12} {'Spearman r':>12} {'p-value':>12}"
    )
    print("-" * 73)

    for _, row in display_df.iterrows():
        # Mark significance
        pearson_sig = "*" if row["pearson_p"] < 0.05 else " "
        spearman_sig = "*" if row["spearman_p"] < 0.05 else " "

        # Mark direction
        if row["pearson_r"] > 0.3:
            direction = " [+]"
        elif row["pearson_r"] < -0.3:
            direction = " [-]"
        else:
            direction = ""

        print(
            f"{row['feature']:<25} {row['pearson_r']:>11.4f}{pearson_sig} "
            f"{row['pearson_p']:>11.4f} {row['spearman_r']:>11.4f}{spearman_sig} "
            f"{row['spearman_p']:>11.4f}{direction}"
        )

    print("\n* indicates p < 0.05")
    print("[+] indicates strong positive correlation (r > 0.3)")
    print("[-] indicates strong negative correlation (r < -0.3)")


def plot_correlation_bars(
    correlations: pd.DataFrame,
    fom: str,
    top_n: int = 15,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a horizontal bar plot showing Pearson and Spearman correlations.

    Args:
        correlations: DataFrame from compute_correlations.
        fom: Name of the FOM (for title).
        top_n: Number of top features to display.
        figsize: Figure size tuple.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    if correlations.empty:
        print("No correlation data to plot.")
        return None

    # Take top N features
    plot_df = correlations.head(top_n).copy()

    # Reverse order for horizontal bar plot (so highest is at top)
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(plot_df))
    bar_height = 0.35

    # Create bars for Pearson and Spearman
    bars1 = ax.barh(
        y_pos - bar_height / 2,
        plot_df["pearson_r"],
        bar_height,
        label="Pearson",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax.barh(
        y_pos + bar_height / 2,
        plot_df["spearman_r"],
        bar_height,
        label="Spearman",
        color="darkorange",
        alpha=0.8,
    )

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Add reference lines at ±0.3
    ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Correlation Coefficient")
    ax.set_title(f"Feature Correlations with {fom}\n(sorted by |Pearson r|)")
    ax.legend(loc="lower right")
    ax.set_xlim(-1.1, 1.1)

    # Add significance markers
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        if row["pearson_p"] < 0.05:
            x_pos = row["pearson_r"]
            ax.annotate(
                "*",
                (x_pos, idx - bar_height / 2),
                fontsize=12,
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved bar plot to {save_path}")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    fom: str,
    features: Optional[list] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a heatmap showing correlations between all features and the FOM.

    Args:
        df: DataFrame with experiment data.
        fom: Name of the figure of merit column.
        features: List of feature columns. If None, uses ALL_FEATURES.
        figsize: Figure size tuple.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    if features is None:
        features = [f for f in ALL_FEATURES if f in df.columns]

    # Add FOM to features for correlation matrix
    cols_to_use = features + [fom]
    cols_to_use = [c for c in cols_to_use if c in df.columns]

    # Compute correlation matrix
    corr_matrix = df[cols_to_use].corr(method="pearson")

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
    )

    ax.set_title(f"Correlation Heatmap (including {fom})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")

    return fig


def plot_top_correlations_only(
    correlations: pd.DataFrame,
    fom: str,
    top_n: int = 10,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a simple bar plot showing only the top N correlated features.

    Bars are colored by direction (positive=green, negative=red).

    Args:
        correlations: DataFrame from compute_correlations.
        fom: Name of the FOM (for title).
        top_n: Number of top features to display.
        figsize: Figure size tuple.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    if correlations.empty:
        print("No correlation data to plot.")
        return None

    plot_df = correlations.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]  # Reverse for horizontal bars

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["forestgreen" if r > 0 else "firebrick" for r in plot_df["pearson_r"]]

    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["pearson_r"], color=colors, alpha=0.8, edgecolor="black")

    ax.axvline(x=0, color="black", linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_title(f"Top {top_n} Features Correlated with {fom}")
    ax.set_xlim(-1.1, 1.1)

    # Add value labels
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        r = row["pearson_r"]
        offset = 0.05 if r >= 0 else -0.05
        ha = "left" if r >= 0 else "right"
        ax.text(r + offset, idx, f"{r:.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved top correlations plot to {save_path}")

    return fig


def plot_scatter(
    df: pd.DataFrame,
    feature: str,
    fom: str,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Create a scatter plot of a feature vs the FOM with a trend line.

    Args:
        df: DataFrame with experiment data.
        feature: Name of the feature column (x-axis).
        fom: Name of the figure of merit column (y-axis).
        figsize: Figure size tuple.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object, or None if insufficient data.
    """
    # Get valid data points
    valid_mask = df[feature].notna() & df[fom].notna()
    x = df.loc[valid_mask, feature]
    y = df.loc[valid_mask, fom]

    if len(x) < 3:
        print(f"Not enough data points for scatter plot of {feature} vs {fom}")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, edgecolors="black", linewidths=0.5)

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Trend line")

    # Compute correlation for display
    pearson_r, pearson_p = stats.pearsonr(x, y)

    ax.set_xlabel(feature)
    ax.set_ylabel(fom)
    ax.set_title(
        f"{feature} vs {fom}\nPearson r = {pearson_r:.3f} (p = {pearson_p:.4f})"
    )
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved scatter plot to {save_path}")

    return fig


def plot_scatter_matrix(
    df: pd.DataFrame,
    fom: str,
    features: Optional[list] = None,
    top_n: int = 4,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Create a grid of scatter plots for top correlated features vs FOM.

    Args:
        df: DataFrame with experiment data.
        fom: Name of the figure of merit column.
        features: List of features to plot. If None, uses top_n from correlation analysis.
        top_n: Number of top features to plot (used if features is None).
        figsize: Figure size tuple.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object, or None if no features to plot.
    """
    if features is None:
        # Get top correlated features
        correlations = compute_correlations(df, fom)
        features = correlations.head(top_n)["feature"].tolist()

    n_features = len(features)
    if n_features == 0:
        print("No features to plot.")
        return None

    # Calculate grid dimensions
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Get valid data points
        valid_mask = df[feature].notna() & df[fom].notna()
        x = df.loc[valid_mask, feature]
        y = df.loc[valid_mask, fom]

        if len(x) < 3:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{feature} vs {fom}")
            continue

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, edgecolors="black", linewidths=0.5)

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)

        # Compute correlation for display
        pearson_r, _ = stats.pearsonr(x, y)

        ax.set_xlabel(feature)
        ax.set_ylabel(fom)
        ax.set_title(f"{feature} (r={pearson_r:.3f})")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Scatter Plots: Top Features vs {fom}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved scatter matrix to {save_path}")

    return fig


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by ensuring all required columns are present.

    Args:
        df: DataFrame with experiment data.

    Returns:
        Preprocessed DataFrame.
    """
    # Drum delta
    df["DrumDeltaT30"] = df["DrumLiveT30"] - df["DrumDeadT30"]

    # Vocal delta, using elementwise max
    if (
        "VocalDoorToWindowT30" in df
        and "VocalCenterToWindowT30" in df
        and "VocalDeadT30" in df
    ):
        df["VocalDeltaT30"] = (
            np.maximum(df["VocalDoorToWindowT30"], df["VocalCenterToWindowT30"])
            - df["VocalDeadT30"]
        )
    else:
        # Optionally, raise an error or warning
        print("Warning: Some vocal columns are missing in the DataFrame.")
        df["VocalDeltaT30"] = np.nan

    return df


def run_full_analysis(
    csv_path: str = "experiment_metrics.csv",
    fom: str = "DrumDeadT30",
    top_n: int = 15,
    show_plots: bool = True,
    save_plots: bool = False,
    scatter: bool = False,
    scatter_feature: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a complete correlation analysis and display results.

    This is the main entry point for typical usage.

    Args:
        csv_path: Path to experiment metrics CSV.
        fom: Figure of merit to analyze.
        top_n: Number of top features to highlight.
        show_plots: Whether to display matplotlib plots.
        save_plots: Whether to save plots to files.
        scatter: Whether to generate scatter plots for top features.
        scatter_feature: If provided, create a scatter plot for this specific feature.

    Returns:
        DataFrame with correlation results.
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} experiments with {len(df.columns)} columns")

    df = preprocess_dataframe(df)

    # Show available FOMs
    available_foms = get_available_foms(df)
    print(f"\nAvailable FOMs: {available_foms}")

    if fom not in available_foms:
        print(f"\nWarning: Selected FOM '{fom}' not in available FOMs.")
        print(f"Using first available FOM: {available_foms[0]}")
        fom = available_foms[0]

    print(f"\nAnalyzing correlations with: {fom}")

    # Compute correlations
    correlations = compute_correlations(df, fom)

    # Print results
    print_correlations(correlations, fom, top_n=top_n)

    # Generate plots
    if show_plots or save_plots:
        bar_save = f"correlation_bars_{fom}.png" if save_plots else None
        heatmap_save = f"correlation_heatmap_{fom}.png" if save_plots else None
        top_save = f"top_correlations_{fom}.png" if save_plots else None

        plot_correlation_bars(correlations, fom, top_n=top_n, save_path=bar_save)
        plot_correlation_heatmap(df, fom, save_path=heatmap_save)
        plot_top_correlations_only(
            correlations, fom, top_n=min(top_n, 10), save_path=top_save
        )

        # Generate scatter plots if requested
        if scatter:
            if scatter_feature:
                # Single feature scatter plot
                scatter_save = (
                    f"scatter_{scatter_feature}_vs_{fom}.png" if save_plots else None
                )
                plot_scatter(df, scatter_feature, fom, save_path=scatter_save)
            else:
                # Scatter matrix of top features
                scatter_save = f"scatter_matrix_{fom}.png" if save_plots else None
                plot_scatter_matrix(
                    df, fom, top_n=min(4, top_n), save_path=scatter_save
                )

        if show_plots:
            plt.show()

    return correlations


def main():
    """Command-line interface for correlation analysis."""
    parser = argparse.ArgumentParser(
        description="Compute correlations between experiment features and a figure of merit (FOM)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="experiment_metrics.csv",
        help="Path to experiment metrics CSV file",
    )
    parser.add_argument(
        "--fom",
        type=str,
        default="DrumDeadT30",
        help="Figure of merit column to analyze (e.g., DrumDeadT30, VocalT30)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top correlated features to display",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to PNG files",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for scripts)",
    )
    parser.add_argument(
        "--list-foms",
        action="store_true",
        help="List available FOMs and exit",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Generate scatter plots for top correlated features vs FOM",
    )
    parser.add_argument(
        "--scatter-feature",
        type=str,
        default=None,
        help="Generate a scatter plot for a specific feature vs FOM (e.g., avg_angle)",
    )

    args = parser.parse_args()

    if args.list_foms:
        df = load_data(args.csv)
        df = preprocess_dataframe(df)
        foms = get_available_foms(df)
        print("Available FOMs:")
        for fom in foms:
            print(f"  - {fom}")
        return

    run_full_analysis(
        csv_path=args.csv,
        fom=args.fom,
        top_n=args.top,
        show_plots=not args.no_show,
        save_plots=args.save_plots,
        scatter=args.scatter or args.scatter_feature is not None,
        scatter_feature=args.scatter_feature,
    )


if __name__ == "__main__":
    main()
