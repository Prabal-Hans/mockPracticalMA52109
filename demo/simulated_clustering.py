"""
This module analyses simulated data and identifies a plausible number of K-means clusters.

It compares candidate cluster counts using inertia, silhouette scores and
visualisations(elbow curve). Then it aves the results for inspection and reproducibility.

Run from the project root:
    python demo/simulated_clustering.py
"""

from itertools import combinations
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from cluster_maker import (
        calculate_descriptive_statistics,
        select_features,
        standardise_features,
        sklearn_kmeans,
        compute_inertia,
        silhouette_score_sklearn,
        plot_clusters_2d,
        plot_elbow,
        export_to_csv,
    )

except ImportError as e:
    raise SystemExit(
        f"Cannot import a required module: {e}. "
        "Check that cluster_maker and its dependencies are installed."
    ) from e


# Project paths and clustering configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "simulated_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "demo_output" / "simulated_clustering"

RANDOM_STATE = 42
MAX_K = 10


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save a Matplotlib figure to OUTPUT_DIR and close it afterwards."""

    try:
        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=150,
            bbox_inches="tight"
        )

    finally:
        plt.close(fig)


def plot_feature_pairs(
    X: np.ndarray,
    labels: np.ndarray,
    feature_cols: list[str],
    k: int
) -> None:
    """
    Save scatter plots for every pair of features in original units.

    X contains the original feature values, labels contains the predicted
    cluster indices, feature_cols contains the axis names, and k is the
    selected number of clusters.
    """

    # Generate every unique pair of feature indices.
    pairs = list(combinations(range(X.shape[1]), 2))

    # Determine the subplot layout dynamically.
    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    # Plot the same clustering in different two-dimensional projections.
    for ax, (i, j) in zip(axes.flat, pairs):

        for cluster_id in range(k):

            mask = labels == cluster_id

            ax.scatter(
                X[mask, i],
                X[mask, j],
                s=16,
                alpha=0.75,
                color=plt.get_cmap("tab10")(cluster_id),
                label=f"Cluster {cluster_id}",
            )

        ax.set(
            xlabel=feature_cols[i],
            ylabel=feature_cols[j]
        )

        ax.grid(alpha=0.2)

    # Hide unused subplots if the number of pairs is not a multiple of n_cols.
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)

    # Use one shared legend to avoid repetition across subplots.
    handles, names = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=min(k, 5)
    )

    fig.suptitle(
        f"Selected clustering across feature pairs (k={k})"
    )

    fig.tight_layout(rect=(0, 0.10, 1, 0.95))

    save_plot(fig, "selected_feature_pairs.png")


def main() -> None:
    """
    Load, preprocess, cluster and evaluate the simulated dataset.

    Compare K-means solutions, recommend the highest valid silhouette
    score, and save the plots, metrics and labelled data.

    Raise informative exceptions if the input data or analysis is unusable.
    """

    # ---------------------------------------------------------
    # 1. Load and inspect the dataset
    # ---------------------------------------------------------

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)

    # Exclude known label columns to avoid leaking cluster information.
    feature_cols = [
        col for col in data.select_dtypes(include="number").columns
        if col.lower() not in {"cluster", "true_cluster"}
    ]

    if len(feature_cols) < 2:
        raise ValueError(
            "At least two numerical features are required."
        )

    X = select_features(data, feature_cols).to_numpy(dtype=float)

    # Reject missing or infinite values before clustering.
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Selected features contain NaN or infinite values."
        )

    # Silhouette requires fewer clusters than observations.
    # Avoid requesting more clusters than distinct observations.
    max_k = min(
        MAX_K,
        len(X) - 1,
        len(np.unique(X, axis=0))
    )

    if max_k < 2:
        raise ValueError(
            "At least three observations and two distinct points are required."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Simulated data: clustering analysis ===")

    print(
        f"Observations: {len(X)} | "
        f"Features: {', '.join(feature_cols)}"
    )

    print("\nDescriptive statistics:")

    print(
        calculate_descriptive_statistics(
            data[feature_cols]
        ).round(3).to_string()
    )

    print(
        "\nClustering uses all selected features; "
        "individual plots show two."
    )

    # ---------------------------------------------------------
    # 2. Visualise the original dataset
    # ---------------------------------------------------------

    fig, ax = plt.subplots()

    ax.scatter(
        X[:, 0],
        X[:, 1],
        s=20,
        alpha=0.65
    )

    ax.set(
        xlabel=feature_cols[0],
        ylabel=feature_cols[1],
        title="Original simulated data"
    )

    fig.tight_layout()

    save_plot(fig, "original_data.png")

    # ---------------------------------------------------------
    # 3. Standardise the numerical features
    # ---------------------------------------------------------

    # Standardise once so every candidate is evaluated on identical data.
    X_scaled = standardise_features(X)

    results = []

    best_k = None
    best_score = -np.inf
    best_labels = None

    # ---------------------------------------------------------
    # 4. Perform clustering for different values of k
    # ---------------------------------------------------------

    for k in range(1, max_k + 1):

        try:
            # The package wrapper uses ten initialisations to reduce
            # sensitivity to poorly selected starting centroids.
            labels, centroids = sklearn_kmeans(
                X_scaled,
                k,
                random_state=RANDOM_STATE
            )

            inertia = compute_inertia(
                X_scaled,
                labels,
                centroids
            )

        except ValueError as exc:
            print(
                f"Skipping k={k}: {exc}",
                file=sys.stderr
            )
            continue

        # Count the number of clusters containing observations.
        occupied = len(np.unique(labels))

        score = np.nan

        # Silhouette is undefined for one cluster or one cluster per point.
        if 2 <= occupied < len(X):

            try:
                score = silhouette_score_sklearn(
                    X_scaled,
                    labels
                )

            except ValueError as exc:
                print(
                    f"Silhouette unavailable for k={k}: {exc}",
                    file=sys.stderr
                )

        # Store the evaluation metrics for later comparison.
        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": score,
            "occupied_clusters": occupied
        })

        print(
            f"k={k:2d} | "
            f"Inertia: {inertia:9.4f} | "
            f"Silhouette: {score:.4f}"
        )

        # Generate cluster visualisations using the package's plotting tools.
        if k >= 2:

            fig, ax = plot_clusters_2d(
                X_scaled,
                labels,
                centroids=centroids,
                title=f"K-means clustering (k={k})"
            )

            ax.set(
                xlabel=f"{feature_cols[0]} (standardised)",
                ylabel=f"{feature_cols[1]} (standardised)"
            )

            fig.tight_layout()

            save_plot(fig, f"clusters_k{k}.png")

        # Reject solutions where fewer clusters than requested are occupied.
        # Retain the configuration with the highest valid silhouette score.
        if (
            occupied == k
            and np.isfinite(score)
            and score > best_score
        ):

            best_k = k
            best_score = score
            best_labels = labels.copy()

    if not results:
        raise ValueError(
            "No clustering configuration could be evaluated."
        )

    # ---------------------------------------------------------
    # 5. Export metrics and generate the elbow curve
    # ---------------------------------------------------------

    summary = pd.DataFrame(results)

    export_to_csv(
        summary,
        str(OUTPUT_DIR / "clustering_metrics.csv")
    )

    # Use the package's plotting function to generate the elbow curve.
    fig, ax = plot_elbow(
        summary["k"].tolist(),
        summary["inertia"].tolist()
    )

    ax.set_xticks(summary["k"].tolist())

    save_plot(fig, "elbow_curve.png")

    if best_k is None or best_labels is None:
        raise ValueError(
            "No valid silhouette-based candidate. "
            "Inspect the saved metrics."
        )

    # ---------------------------------------------------------
    # 6. Compare silhouette scores
    # ---------------------------------------------------------

    valid = summary[
        (summary["occupied_clusters"] == summary["k"])
        & np.isfinite(summary["silhouette"])
    ]

    fig, ax = plt.subplots()

    # Highlight the selected clustering configuration.
    ax.bar(
        valid["k"],
        valid["silhouette"],
        color=[
            "tab:green" if k == best_k else "tab:blue"
            for k in valid["k"]
        ]
    )

    ax.set(
        xlabel="Number of clusters (k)",
        ylabel="Silhouette score",
        title="Silhouette comparison",
        ylim=(-1, 1)
    )

    ax.set_xticks(valid["k"].tolist())

    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    save_plot(fig, "silhouette_comparison.png")

    # ---------------------------------------------------------
    # 7. Examine the selected clustering across all feature pairs
    # ---------------------------------------------------------

    plot_feature_pairs(
        X,
        best_labels,
        feature_cols,
        best_k
    )

    # ---------------------------------------------------------
    # 8. Export and analyse the selected clustering
    # ---------------------------------------------------------

    # Retain the original feature values and append predicted cluster labels.
    labelled = data.copy()

    labelled["cluster"] = best_labels

    export_to_csv(
        labelled,
        str(OUTPUT_DIR / f"selected_k{best_k}.csv")
    )

    # Calculate cluster means in the original feature units.
    profile = labelled.groupby("cluster")[feature_cols].mean()

    # Include the number of observations assigned to each cluster.
    profile.insert(
        0,
        "count",
        labelled["cluster"].value_counts()
    )

    export_to_csv(
        profile.reset_index(),
        str(OUTPUT_DIR / "cluster_profiles.csv")
    )

    # ---------------------------------------------------------
    # 9. Present the final results
    # ---------------------------------------------------------

    print("\n=== Proposed clustering ===")

    print(
        f"Selected k: {best_k} | "
        f"Highest valid silhouette: {best_score:.4f}"
    )

    print("\nCluster profiles (original units):")

    print(profile.round(3).to_string())

    # Compare the inertia reductions immediately around the selected k.
    inertia_by_k = summary.set_index("k")["inertia"]

    for left in (best_k - 1, best_k):

        if left in inertia_by_k and left + 1 in inertia_by_k:

            reduction = (
                inertia_by_k[left] - inertia_by_k[left + 1]
            )

            print(
                f"Inertia reduction from k={left} "
                f"to k={left + 1}: {reduction:.4f}"
            )

    print(
        "\nUse the elbow curve and pairwise plots to assess whether the"
    )

    print(
        "silhouette-based candidate represents a plausible separation."
    )

    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":

    try:
        main()

    except (OSError, ValueError, TypeError, KeyError) as exc:

        print(
            f"Clustering analysis failed: {exc}",
            file=sys.stderr
        )

        sys.exit(1)