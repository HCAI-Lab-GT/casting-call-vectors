"""Visualization for persona vector analysis.

This module provides plotting functions for PCA projections, clustering,
and variance analysis. Supports both static (matplotlib) and interactive
(plotly) visualizations, with optional W&B integration.
"""

import logging
from pathlib import Path
from typing import Callable

from .geometry import ClusterResult, PCAResult, PersonaGeometry

logger = logging.getLogger(__name__)


class PersonaVisualizer:
    """Visualization for persona vector geometry.

    Creates 2D/3D PCA plots, variance explained charts, and cluster
    visualizations. Supports custom coloring schemes for any axis system.

    Example:
        >>> viz = PersonaVisualizer(geometry)
        >>> fig = viz.plot_pca_2d(pca_result, color_by=lambda pid, m: m.get("riasec_primary"))
        >>> fig.savefig("pca_2d.png")
    """

    def __init__(
        self,
        geometry: PersonaGeometry,
        default_colormap: str = "tab10",
    ):
        """Initialize visualizer.

        Args:
            geometry: PersonaGeometry instance with vectors and metadata
            default_colormap: Default colormap for categorical coloring
        """
        self.geometry = geometry
        self.default_colormap = default_colormap

    def _get_colors(
        self,
        color_by: str | Callable[[str, dict], str | None] | None,
        color_map: dict[str, str] | None = None,
    ) -> tuple[list[str], dict[str, str], list[str | None]]:
        """Get colors for each persona based on grouping function or key.

        Args:
            color_by: Metadata key string or function (persona_id, metadata) -> group
            color_map: Optional mapping of group -> color

        Returns:
            Tuple of (colors_list, legend_map, group_labels)
        """
        import matplotlib.pyplot as plt

        # Get group labels
        group_labels: list[str | None] = []
        for pid in self.geometry.persona_ids:
            meta = self.geometry.metadata.get(pid, {})
            if callable(color_by):
                label = color_by(pid, meta)
            elif isinstance(color_by, str):
                label = meta.get(color_by)
            else:
                label = None
            group_labels.append(label)

        # Get unique groups (excluding None)
        unique_groups = sorted({g for g in group_labels if g is not None})

        # Build color map if not provided
        if color_map is None:
            cmap = plt.get_cmap(self.default_colormap)
            color_map = {
                group: cmap(i / max(len(unique_groups) - 1, 1))
                for i, group in enumerate(unique_groups)
            }

        # Assign colors
        colors = []
        for label in group_labels:
            if label is not None and label in color_map:
                colors.append(color_map[label])
            else:
                colors.append("gray")

        return colors, color_map, group_labels

    def plot_pca_2d(
        self,
        pca_result: PCAResult,
        pc_x: int = 0,
        pc_y: int = 1,
        color_by: str | Callable[[str, dict], str | None] | None = None,
        color_map: dict[str, str] | None = None,
        figsize: tuple[float, float] = (10, 8),
        title: str | None = None,
        show_labels: bool = False,
        alpha: float = 0.7,
        s: float = 50,
    ):
        """Create 2D PCA projection plot.

        Args:
            pca_result: PCA result from geometry.compute_pca()
            pc_x: PC index for x-axis (0-indexed)
            pc_y: PC index for y-axis (0-indexed)
            color_by: Metadata key or function for coloring
            color_map: Optional group -> color mapping
            figsize: Figure size
            title: Plot title
            show_labels: Whether to show persona ID labels
            alpha: Point transparency
            s: Point size

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Get colors
        colors, legend_map, group_labels = self._get_colors(color_by, color_map)

        # Plot points
        x = pca_result.projections[:, pc_x]
        y = pca_result.projections[:, pc_y]

        ax.scatter(x, y, c=colors, alpha=alpha, s=s)

        # Add labels if requested
        if show_labels:
            for i, pid in enumerate(pca_result.persona_ids):
                ax.annotate(
                    pid[:15],
                    (x[i], y[i]),
                    fontsize=6,
                    alpha=0.7,
                )

        # Add legend
        if color_by and legend_map:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=g)
                for g, c in legend_map.items()
            ]
            ax.legend(handles=legend_elements, loc="best")

        # Labels and title
        var_x = pca_result.explained_variance_ratio[pc_x] * 100
        var_y = pca_result.explained_variance_ratio[pc_y] * 100
        ax.set_xlabel(f"PC{pc_x + 1} ({var_x:.1f}%)")
        ax.set_ylabel(f"PC{pc_y + 1} ({var_y:.1f}%)")
        ax.set_title(title or "Persona Vector PCA Projection")

        plt.tight_layout()
        return fig

    def plot_pca_3d(
        self,
        pca_result: PCAResult,
        color_by: str | Callable[[str, dict], str | None] | None = None,
        color_map: dict[str, str] | None = None,
        figsize: tuple[float, float] = (12, 10),
        title: str | None = None,
        alpha: float = 0.7,
        s: float = 50,
    ):
        """Create 3D PCA projection plot.

        Args:
            pca_result: PCA result from geometry.compute_pca()
            color_by: Metadata key or function for coloring
            color_map: Optional group -> color mapping
            figsize: Figure size
            title: Plot title
            alpha: Point transparency
            s: Point size

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Get colors
        colors, legend_map, _ = self._get_colors(color_by, color_map)

        # Plot points
        x = pca_result.projections[:, 0]
        y = pca_result.projections[:, 1]
        z = pca_result.projections[:, 2]

        ax.scatter(x, y, z, c=colors, alpha=alpha, s=s)

        # Labels and title
        var = pca_result.explained_variance_ratio
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
        ax.set_zlabel(f"PC3 ({var[2] * 100:.1f}%)")
        ax.set_title(title or "Persona Vector PCA Projection (3D)")

        # Add legend
        if color_by and legend_map:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=g)
                for g, c in legend_map.items()
            ]
            ax.legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        return fig

    def plot_variance_explained(
        self,
        pca_result: PCAResult,
        n_components: int | None = None,
        figsize: tuple[float, float] = (10, 5),
        title: str | None = None,
    ):
        """Plot variance explained by principal components.

        Args:
            pca_result: PCA result from geometry.compute_pca()
            n_components: Number of components to show (None for all)
            figsize: Figure size
            title: Plot title

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        n = n_components or len(pca_result.explained_variance_ratio)
        x = range(1, n + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Individual variance
        ax1.bar(x, pca_result.explained_variance_ratio[:n] * 100)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Variance Explained (%)")
        ax1.set_title("Individual Variance")
        ax1.set_xticks(x)

        # Cumulative variance
        ax2.plot(x, pca_result.cumulative_variance_ratio[:n] * 100, "b-o")
        ax2.axhline(y=90, color="r", linestyle="--", label="90% threshold")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance (%)")
        ax2.set_title("Cumulative Variance")
        ax2.set_xticks(x)
        ax2.legend()

        fig.suptitle(title or "PCA Variance Analysis")
        plt.tight_layout()
        return fig

    def plot_clusters(
        self,
        cluster_result: ClusterResult,
        pca_result: PCAResult,
        figsize: tuple[float, float] = (10, 8),
        title: str | None = None,
    ):
        """Plot cluster assignments on PCA projection.

        Args:
            cluster_result: Clustering result
            pca_result: PCA result for projection
            figsize: Figure size
            title: Plot title

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Color by cluster
        cmap = plt.get_cmap("tab10")
        colors = [cmap(label % 10) for label in cluster_result.labels]

        x = pca_result.projections[:, 0]
        y = pca_result.projections[:, 1]

        ax.scatter(x, y, c=colors, alpha=0.7, s=50)

        # Plot cluster centers if available
        if cluster_result.cluster_centers is not None:
            centers = cluster_result.cluster_centers
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="black",
                marker="x",
                s=200,
                linewidths=3,
                label="Centers",
            )

        ax.set_xlabel(f"PC1 ({pca_result.explained_variance_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_result.explained_variance_ratio[1] * 100:.1f}%)")
        ax.set_title(title or f"Cluster Assignments ({cluster_result.method})")

        plt.tight_layout()
        return fig

    def plot_distance_matrix(
        self,
        figsize: tuple[float, float] = (12, 10),
        title: str | None = None,
        sort_by: str | Callable[[str, dict], str | None] | None = None,
    ):
        """Plot pairwise distance matrix as heatmap.

        Args:
            figsize: Figure size
            title: Plot title
            sort_by: Optional grouping for sorting rows/columns

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        distances = self.geometry.compute_distances()

        # Optionally sort by group
        if sort_by:
            _, _, group_labels = self._get_colors(sort_by)
            # Sort indices by group
            sorted_indices = sorted(
                range(len(group_labels)),
                key=lambda i: (group_labels[i] or "", self.geometry.persona_ids[i]),
            )
            distances = distances[sorted_indices][:, sorted_indices]
            labels = [self.geometry.persona_ids[i] for i in sorted_indices]
        else:
            labels = self.geometry.persona_ids

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(distances, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="Cosine Distance")

        # Only show labels if not too many
        if len(labels) <= 50:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)

        ax.set_title(title or "Pairwise Cosine Distances")
        plt.tight_layout()
        return fig

    def plot_interactive_pca(
        self,
        pca_result: PCAResult,
        color_by: str | Callable[[str, dict], str | None] | None = None,
        hover_keys: list[str] | None = None,
        title: str | None = None,
    ):
        """Create interactive 3D PCA plot with Plotly.

        Args:
            pca_result: PCA result from geometry.compute_pca()
            color_by: Metadata key or function for coloring
            hover_keys: Metadata keys to show on hover
            title: Plot title

        Returns:
            plotly Figure
        """
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go  # noqa: F401
        except ImportError:
            logger.warning("plotly not installed, falling back to matplotlib")
            return self.plot_pca_3d(pca_result, color_by=color_by, title=title)

        # Build dataframe
        data = {
            "persona_id": pca_result.persona_ids,
            "PC1": pca_result.projections[:, 0],
            "PC2": pca_result.projections[:, 1],
            "PC3": pca_result.projections[:, 2] if pca_result.projections.shape[1] > 2 else 0,
        }

        # Add group labels
        if color_by:
            groups = []
            for pid in pca_result.persona_ids:
                meta = self.geometry.metadata.get(pid, {})
                if callable(color_by):
                    groups.append(color_by(pid, meta))
                else:
                    groups.append(meta.get(color_by))
            data["group"] = groups

        # Add hover data
        if hover_keys:
            for key in hover_keys:
                data[key] = [
                    self.geometry.metadata.get(pid, {}).get(key, "")
                    for pid in pca_result.persona_ids
                ]

        df = pd.DataFrame(data)

        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="group" if color_by else None,
            hover_name="persona_id",
            hover_data=hover_keys or [],
            title=title or "Interactive PCA Projection",
        )

        fig.update_traces(marker={"size": 5, "opacity": 0.8})
        fig.update_layout(
            scene={
                "xaxis_title": f"PC1 ({pca_result.explained_variance_ratio[0] * 100:.1f}%)",
                "yaxis_title": f"PC2 ({pca_result.explained_variance_ratio[1] * 100:.1f}%)",
                "zaxis_title": f"PC3 ({pca_result.explained_variance_ratio[2] * 100:.1f}%)"
                if pca_result.projections.shape[1] > 2
                else "PC3",
            }
        )

        return fig

    def log_to_wandb(
        self,
        run,
        pca_result: PCAResult,
        color_by: str | Callable[[str, dict], str | None] | None = None,
        prefix: str = "",
    ) -> None:
        """Log visualizations to W&B.

        Args:
            run: W&B run object
            pca_result: PCA result to visualize
            color_by: Metadata key or function for coloring
            prefix: Prefix for logged image names
        """
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
            return

        # Log 2D PCA
        fig_2d = self.plot_pca_2d(pca_result, color_by=color_by)
        run.log({f"{prefix}pca_2d": wandb.Image(fig_2d)})

        # Log variance explained
        fig_var = self.plot_variance_explained(pca_result)
        run.log({f"{prefix}variance_explained": wandb.Image(fig_var)})

        # Log 3D interactive if plotly available
        try:
            fig_3d = self.plot_interactive_pca(pca_result, color_by=color_by)
            run.log({f"{prefix}pca_3d_interactive": fig_3d})
        except Exception as e:
            logger.warning(f"Failed to log interactive plot: {e}")

        # Clean up figures
        import matplotlib.pyplot as plt

        plt.close("all")

    def save_all(
        self,
        output_dir: Path | str,
        pca_result: PCAResult,
        color_by: str | Callable[[str, dict], str | None] | None = None,
        cluster_result: ClusterResult | None = None,
    ) -> None:
        """Save all standard visualizations to disk.

        Args:
            output_dir: Directory for output files
            pca_result: PCA result to visualize
            color_by: Metadata key or function for coloring
            cluster_result: Optional clustering result
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # PCA 2D
        fig = self.plot_pca_2d(pca_result, color_by=color_by)
        fig.savefig(output_dir / "pca_2d.png", dpi=150)

        # PCA 3D
        fig = self.plot_pca_3d(pca_result, color_by=color_by)
        fig.savefig(output_dir / "pca_3d.png", dpi=150)

        # Variance
        fig = self.plot_variance_explained(pca_result)
        fig.savefig(output_dir / "variance_explained.png", dpi=150)

        # Distance matrix
        fig = self.plot_distance_matrix(sort_by=color_by)
        fig.savefig(output_dir / "distance_matrix.png", dpi=150)

        # Clusters if available
        if cluster_result:
            fig = self.plot_clusters(cluster_result, pca_result)
            fig.savefig(output_dir / "clusters.png", dpi=150)

        # Interactive HTML
        try:
            fig_interactive = self.plot_interactive_pca(pca_result, color_by=color_by)
            fig_interactive.write_html(output_dir / "pca_interactive.html")
        except Exception as e:
            logger.warning(f"Could not save interactive plot: {e}")

        # Clean up
        import matplotlib.pyplot as plt

        plt.close("all")

        logger.info(f"Saved visualizations to {output_dir}")
