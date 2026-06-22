# -*- coding: utf-8 -*-

"""Code to plots displayed in the manuscript."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from collections import defaultdict
from typing import Tuple, List, Optional
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib_venn import venn2, venn3
from upsetplot import from_contents, UpSet

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from benchmarking.ms2_quality import assess_ms2_quality
from benchmarking.constants import (
    TOOL_COLORS,
    TOOL_NAMES,
    DATASET_DESCRIPTORS,
    LIBRARY_ID_COLUMN,
    INCHIKEY_COLUMN,
    SCORE_COLUMN,
    LIBRARY_PRECURSOR_MZ_COLUMN,
    LIBRARY_MZS_COLUMN,
    LIBRARY_INTENSITIES_COLUMN,
    TRANSFORM_SETTINGS,
    ANNOTATION_COLORS,
    ANNOTATION_NAMES,
)
from benchmarking.metrics.base_metrics import compute_base_metrics, get_euler_data
from benchmarking.utils import apply_rclr_transform, apply_clr_transform

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("mode.chained_assignment", None)


def _plot_radar_single(
    metrics: dict,
    title,
    ax=None,
):
    """
    Plot MS1 Count, MS2 Count, and Annotated Count on a radar plot with individual scales for each metric.
    If ax is provided, plot on that axes; otherwise create a new figure.

    Parameters:
    -----------
    metrics : dict
        Dictionary with keys "MS1 Count", "MS2 Count", "Annotated Count" and tool names as subkeys (e.g., {"mzmine": {"MS1 Count": 1000, "MS2 Count": 500, "Annotated Count": 200}, ...})
    title : str
        Title for the plot
    ax : matplotlib axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    """
    # Select the three metrics
    metric_names = ["MS1 Count", "MS2 Count", "Annotated Count"]

    # Create DataFrame with original values
    df = pd.DataFrame(metrics).T

    # Normalize each metric independently (0-1 scale based on each column's min/max)
    df_normalized = df.copy()
    for col in df.columns:
        min_val = 0
        max_val = max(df[col].max(), 1)
        if max_val > min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0

    # Setup radar plot
    num_vars = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the circle

    # Plot each tool using normalized values
    for tool in df_normalized.index:
        values = df_normalized.loc[tool].tolist()
        values += values[:1]  # complete the circle

        color = TOOL_COLORS.get(tool, "#CCCCCC")
        label = TOOL_NAMES.get(tool, tool)

        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10)
    ax.tick_params(pad=20)

    # Set normalized y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add actual count labels near the data vertices for each metric
    tools = list(df.index)
    num_tools = len(tools)
    radial_offset = 0.06
    outside_push = 0.05
    angular_offset = np.deg2rad(3.0)

    for metric_idx, metric in enumerate(metric_names):
        angle = angles[metric_idx]
        for j, tool in enumerate(tools):
            color = TOOL_COLORS.get(tool, "#CCCCCC")
            actual_value = df.loc[tool, metric]
            base_r = df_normalized.loc[tool, metric]
            radial_jitter = (j - (num_tools - 1) / 2.0) * radial_offset
            angle_jitter = (j - (num_tools - 1) / 2.0) * angular_offset
            label_angle = angle + angle_jitter
            label_r = max(base_r + outside_push + radial_jitter, 0.05)
            ax.text(
                label_angle,
                label_r,
                f"{int(actual_value):,}",
                ha="center",
                va="center",
                size=8,
                color=color,
                fontweight="bold",
                clip_on=False,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=1.0,
                ),
            )

    ax.set_title(title, size=10, pad=15, fontweight="bold")


def plot_individual_radars(
    input_dataset_directories: list,
    title: str,
    grid_shape: tuple = None,
    similarity_threshold: float = 0.7,
    save_path: str = None,
    figsize_per_plot: tuple = (5, 5),
    annotation_subfolder="annotated_spectral_entropy",
    legend_position=(1.02, 0.5),
):
    """
    Plot a grid of radar plots for all datasets in input_dataset_directories.

    Parameters:
    -----------
    input_dataset_directories : list
        List of paths to dataset directories
    title : str
        Overall title for the figure
    grid_shape : tuple, optional
        (rows, cols) for the grid. If None, auto-calculated as square-ish grid.
    included_tools : list, optional
        List of tools to include. If None, includes all.
    similarity_threshold : float
        Score threshold for annotations
    save_path : str, optional
        Directory to save the figure
    figsize_per_plot : tuple
        Size of each individual radar plot (width, height)
    """
    n_datasets = len(input_dataset_directories)

    # Auto-calculate grid shape if not provided
    if grid_shape is None:
        n_cols = int(np.ceil(np.sqrt(n_datasets)))
        n_rows = int(np.ceil(n_datasets / n_cols))
    else:
        n_rows, n_cols = grid_shape

    # Calculate figure size
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    # Create figure with polar subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, subplot_kw=dict(projection="polar")
    )

    # Flatten axes for easy iteration (handle single row/col case)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    # Plot each dataset
    for i, dataset_path in enumerate(input_dataset_directories):
        ax = axes_flat[i]
        dataset_id = dataset_path.strip("/").split("/")[-1]
        dataset_label = DATASET_DESCRIPTORS.get(dataset_id, dataset_id)

        try:
            # Compute metrics for this dataset
            metrics = compute_base_metrics(
                dataset_path,
                similarity_threshold=similarity_threshold,
                annotation_subfolder=annotation_subfolder,
            )

            # Plot radar on this axes
            _plot_radar_single(
                metrics,
                title=dataset_label,
                ax=ax,
            )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error:\n{str(e)[:30]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title(dataset_label, size=10, pad=15)

    # Hide unused axes
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Create a shared legend
    # Get handles/labels from the first valid plot
    handles, labels = [], []
    for ax in axes_flat[:n_datasets]:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    # Remove individual legends and add shared legend
    for ax in axes_flat[:n_datasets]:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=legend_position,
            title="Workflow",
            fontsize=18,
            title_fontsize=20,
        )

    # Add overall title
    fig.suptitle(title, size=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_radar_combined(
    combined_metrics: dict,
    title: str = None,
    save_path: str = None,
    figsize: tuple = (10, 10),
    label_offsets=None,
    title_fontsize=30,
):
    """
    Plot a single radar plot with combined (summed) metrics across all datasets.

    Parameters:
    -----------
    combined_metrics : dict
        Dictionary with combined metrics for each tool (e.g., {"metaboscape": {"MS1 Count": 1000, "MS2 Count": 500, "Annotated Count": 200}, ...})
    title : str
        Title for the figure
    save_path : str, optional
        Directory to save the figure
    figsize : tuple
        Size of the figure (width, height), used only if ax is None
    label_offsets : dict, optional
        Dictionary with manual offsets for labels (e.g., {"mzmine": {"MS1 Count": (0.0, 0.0), ...}, ...})

    Returns:
    --------
    fig : matplotlib figure (or None if ax was provided)
    ax : matplotlib axes
    """
    # Create DataFrame with combined values
    df = pd.DataFrame(combined_metrics).T

    # Normalize each metric independently (0-1 scale)
    df_normalized = df.copy()
    for col in df.columns:
        min_val = 0
        max_val = max(df[col].max(), 1)
        if max_val > min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0

    # Setup radar plot
    if not label_offsets:
        label_offsets = {
            "mzmine": {
                "MS1 Count": (0, -1.0),
                "MS2 Count": (-0.2, 0),
                "Annotated Count": (0, 0.0),
            },
            "msdial": {
                "MS1 Count": (0, -20.0),
                "MS2 Count": (-0.05, -7.0),
                "Annotated Count": (-0.2, -10),
            },
            "metaboscape": {
                "MS1 Count": (0.1, 5.0),
                "MS2 Count": (-0.1, 4.0),
                "Annotated Count": (-0.05, 8.0),
            },
        }

    num_vars = 3  # MS1 Count, MS2 Count, Annotated Count
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the circle

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    # Plot each tool using normalized values
    for tool in df_normalized.index:
        values = df_normalized.loc[tool].tolist()
        values += values[:1]  # complete the circle

        color = TOOL_COLORS.get(tool, "#CCCCCC")
        label = TOOL_NAMES.get(tool, tool)

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill(angles, values, alpha=0.25, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["    MS1 Count", "MS2 Count", "Annotated Count"], size=15, fontweight="bold"
    )
    ax.tick_params(pad=30)

    # Set normalized y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add actual count labels near the data vertices for each metric
    tools = list(df.index)
    num_tools = len(tools)
    radial_offset = 0.06
    outside_push = 0.1
    angular_offset = np.deg2rad(4.0)

    for metric_idx, metric in enumerate(["MS1 Count", "MS2 Count", "Annotated Count"]):
        angle = angles[metric_idx]
        for j, tool in enumerate(tools):
            color = TOOL_COLORS.get(tool, "#CCCCCC")
            actual_value = df.loc[tool, metric]
            base_r = df_normalized.loc[tool, metric]
            radial_jitter = (j - (num_tools - 1) / 2.0) * radial_offset
            angle_jitter = (j - (num_tools - 1) / 2.0) * angular_offset
            # Apply manual offsets if provided
            manual_r_offset = 0.0
            manual_angle_offset = 0.0
            if (
                label_offsets
                and tool in label_offsets
                and metric in label_offsets[tool]
            ):
                manual_r_offset, manual_angle_offset = label_offsets[tool][metric]
                manual_angle_offset = np.deg2rad(manual_angle_offset)
            label_angle = angle + angle_jitter + manual_angle_offset
            label_r = max(base_r + outside_push + radial_jitter + manual_r_offset, 0.05)
            ax.text(
                label_angle,
                label_r,
                f"{int(actual_value):,}",
                ha="center",
                va="center",
                size=15,
                color=color,
                fontweight="bold",
                clip_on=False,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=1.0,
                ),
            )

    # Set title
    if title:
        ax.set_title(title, size=title_fontsize, pad=30, fontweight="bold")

    # Add legend only for standalone plots
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        # title="Workflow",
        fontsize=12,
        title_fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_individual_upset(
    input_dataset_directories: list,
    title: str,
    grid_shape: tuple = (3, 4),
    similarity_threshold: float = 0.7,
    save_path: str = None,
    figsize_per_plot: tuple = (7, 4),
):
    """
    Plot a grid of UpSet plots for all datasets.
    Since UpSet.plot() creates its own internal subfigures, we render each
    plot to a buffer image and then arrange them in a grid.
    """
    n_datasets = len(input_dataset_directories)

    # Auto-calculate grid shape if not provided
    if grid_shape is None:
        n_cols = int(np.ceil(np.sqrt(n_datasets)))
        n_rows = int(np.ceil(n_datasets / n_cols))
    else:
        n_rows, n_cols = grid_shape

    # Render each UpSet plot to an image buffer
    plot_images = []
    dataset_labels = []

    for i, dataset_path in enumerate(input_dataset_directories):
        dataset_id = dataset_path.strip("/").split("/")[-1]
        dataset_label = DATASET_DESCRIPTORS.get(dataset_id, dataset_id)
        dataset_labels.append(dataset_label)

        try:
            # Get euler data for this dataset
            labels, sets = get_euler_data(
                dataset_path,
                similarity_threshold=similarity_threshold,
                annotation_subfolder="annotated_spectral_entropy",
            )

            sets_dict = {
                TOOL_NAMES.get(label, label): s for label, s in zip(labels, sets)
            }

            upset_data = from_contents(sets_dict)

            # Create color mapping
            color_map = {
                TOOL_NAMES.get(label, label): TOOL_COLORS.get(label, "#CCCCCC")
                for label in labels
            }

            # Create individual figure for this upset plot
            temp_fig = plt.figure(figsize=figsize_per_plot)
            upset = UpSet(
                upset_data,
                subset_size="count",
                show_counts=True,
                element_size=25,
                intersection_plot_elements=4,
            )
            upset.plot(fig=temp_fig)

            # Apply coloring to the upset plot
            if len(temp_fig.axes) >= 3:
                totals_ax = temp_fig.axes[2]

                def _build_tick_lookup(ax):
                    ticks = ax.get_yticks()
                    labels_list = [tick.get_text() for tick in ax.get_yticklabels()]
                    return [
                        (tick, label)
                        for tick, label in zip(ticks, labels_list)
                        if label
                    ]

                def _label_for_position(lookup, y_value):
                    if not lookup:
                        return None
                    tick, label = min(lookup, key=lambda pair: abs(pair[0] - y_value))
                    return label

                totals_lookup = _build_tick_lookup(totals_ax)
                for patch in totals_ax.patches:
                    center_y = patch.get_y() + patch.get_height() / 2
                    label = _label_for_position(totals_lookup, center_y)
                    color = color_map.get(label)
                    if color:
                        patch.set_facecolor(color)
                        patch.set_edgecolor("black")
                        patch.set_linewidth(0.5)
            temp_fig.suptitle(dataset_label, fontsize=12, fontweight="bold", y=1.02)
            temp_fig.tight_layout()

            # Save to buffer
            buf = BytesIO()
            temp_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(temp_fig)
            buf.seek(0)
            plot_images.append(Image.open(buf))

        except Exception as e:
            print(f"Error processing {dataset_id}: {str(e)}")
            # Create error placeholder
            temp_fig, temp_ax = plt.subplots(figsize=figsize_per_plot)
            temp_ax.text(
                0.5, 0.5, f"Error: {str(e)[:30]}", ha="center", va="center", fontsize=8
            )
            temp_ax.set_xticks([])
            temp_ax.set_yticks([])
            temp_fig.suptitle(dataset_label, fontsize=10, fontweight="bold")

            buf = BytesIO()
            temp_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(temp_fig)
            buf.seek(0)
            plot_images.append(Image.open(buf))

    # Now create the grid figure by placing the images
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )

    # Handle single row/col case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    for i, img in enumerate(plot_images):
        ax = axes_flat[i]
        ax.imshow(np.array(img))
        ax.axis("off")

    # Hide unused axes
    for i in range(len(plot_images), len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Add overall title
    fig.suptitle(title, size=22, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_upset_combined(
    combined_sets: dict,
    title: str,
    save_path: str = None,
    figsize: tuple = (14, 8),
):
    """
    Plot a single UpSet plot with combined (unioned) InChIKeys across all datasets.

    Parameters:
    -----------
    combined_sets : dict
        Dictionary with tool names as keys and sets of InChIKeys as values
    title : str
        Title for the figure
    include_tools : list, optional
        List of tools to include. If None, includes all.
    save_path : str, optional
        Directory to save the figure
    figsize : tuple
        Size of the figure (width, height)

    Returns:
    --------
    fig : matplotlib figure
    """
    # Create sets dictionary with formatted tool names
    sets_dict = {
        TOOL_NAMES.get(label, label): combined_sets[label]
        for label in combined_sets.keys()
    }

    # Create color mapping
    color_map = {
        TOOL_NAMES.get(label, label): TOOL_COLORS.get(label, "#CCCCCC")
        for label in combined_sets.keys()
    }

    # Create UpSet data structure
    upset_data = from_contents(sets_dict)

    # Create UpSet plot
    fig = plt.figure(figsize=figsize)
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        element_size=40,
        intersection_plot_elements=6,
        sort_by="cardinality",
        sort_categories_by="cardinality",
    )

    upset.plot(fig=fig)
    intersections = upset.intersections
    inclusion_matrix = intersections.index.to_frame().astype(bool).values
    category_names = list(intersections.index.names)
    if any(name is None for name in category_names):
        category_names = [
            TOOL_NAMES.get(label, label) for label in combined_sets.keys()
        ]
    n_categories = len(category_names)
    membership_flags = (
        inclusion_matrix.flatten() if inclusion_matrix.size else np.array([])
    )
    subset_count = len(intersections)
    category_index_per_point = (
        np.tile(np.arange(n_categories), subset_count)
        if membership_flags.size
        else np.array([])
    )

    def _build_tick_lookup(ax):
        ticks = ax.get_yticks()
        labels_list = [tick.get_text() for tick in ax.get_yticklabels()]
        return [(tick, label) for tick, label in zip(ticks, labels_list) if label]

    def _label_for_position(lookup, y_value):
        if not lookup:
            return None
        tick, label = min(lookup, key=lambda pair: abs(pair[0] - y_value))
        return label

    if len(fig.axes) >= 3:
        totals_ax = fig.axes[2]
        totals_lookup = _build_tick_lookup(totals_ax)
        for patch in totals_ax.patches:
            center_y = patch.get_y() + patch.get_height() / 2
            label = _label_for_position(totals_lookup, center_y)
            color = color_map.get(label)
            if color:
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)

        matrix_ax = fig.axes[1]
        if membership_flags.size:
            formatted_tick_labels = [
                TOOL_NAMES.get(name, name) for name in category_names
            ]
            matrix_ax.set_yticklabels(formatted_tick_labels, fontsize=13)
            expected_points = membership_flags.size
            for collection in matrix_ax.collections:
                offsets = collection.get_offsets()
                original_facecolors = collection.get_facecolors()
                if (
                    offsets is None
                    or original_facecolors is None
                    or len(original_facecolors) != expected_points
                ):
                    continue
                facecolors = original_facecolors.copy()
                original_edgecolors = collection.get_edgecolors()
                if (
                    original_edgecolors is None
                    or len(original_edgecolors) != expected_points
                ):
                    edgecolors = np.tile(to_rgba("#4A4A4A"), (expected_points, 1))
                else:
                    edgecolors = original_edgecolors.copy()
                linewidths = collection.get_linewidths()
                if linewidths is None or len(linewidths) != expected_points:
                    linewidths = np.ones(expected_points)
                else:
                    linewidths = linewidths.copy()

                for idx in range(expected_points):
                    if not membership_flags[idx]:
                        facecolors[idx] = original_facecolors[idx]
                        edgecolors[idx] = to_rgba("#4A4A4A")
                        linewidths[idx] = max(linewidths[idx], 0.0)
                        continue
                    category_name = category_names[category_index_per_point[idx]]
                    rgba_color = to_rgba(color_map.get(category_name, "#CCCCCC"))
                    facecolors[idx] = rgba_color
                    edgecolors[idx] = rgba_color
                    linewidths[idx] = max(linewidths[idx], 1.5)

                collection.set_facecolors(facecolors)
                collection.set_edgecolors(edgecolors)
                collection.set_linewidths(linewidths)
                break

    plt.suptitle(title, y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_venn_combined(
    combined_sets: dict,
    title: str,
    save_path: str = None,
    figsize: tuple = (8, 8),
):
    """Plot combined annotation overlap as a Venn diagram.

    Uses the same ``combined_sets`` input shape as ``plot_upset_combined``.
    Supports exactly 2 or 3 tools.

    Parameters
    ----------
    combined_sets : dict
        Dictionary with tool keys and sets of InChIKeys as values.
    title : str
        Title for the figure.
    save_path : str, optional
        Path where the figure should be saved.
    figsize : tuple
        Figure size (width, height).
    """
    if not combined_sets:
        raise ValueError("combined_sets cannot be empty")

    n_tools = len(combined_sets)
    if n_tools not in (2, 3):
        raise ValueError(
            f"Venn diagram requires exactly 2 or 3 tools, got {n_tools}. "
            "Use plot_upset_combined for higher-order overlaps."
        )

    tool_keys = list(combined_sets.keys())
    tool_labels = [TOOL_NAMES.get(tool, tool) for tool in tool_keys]
    tool_colors = [TOOL_COLORS.get(tool, "#CCCCCC") for tool in tool_keys]
    tool_sets = [set(combined_sets[tool]) for tool in tool_keys]

    plt.figure(figsize=figsize)

    if n_tools == 2:
        venn = venn2(
            subsets=tool_sets,
            set_labels=tool_labels,
            subset_label_formatter=lambda x: f"{int(x):,}",
            set_colors=tool_colors,
            alpha=0.6,
        )
    else:
        venn = venn3(
            subsets=tool_sets,
            set_labels=tool_labels,
            subset_label_formatter=lambda x: f"{int(x):,}",
            set_colors=tool_colors,
            alpha=0.6,
        )

    for label in venn.set_labels:
        if label:
            label.set_fontsize(14)

    for label in venn.subset_labels:
        if label:
            label.set_fontsize(12)

    plt.title(title, fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_precursor_distributions(
    all_tables: dict,
    title: str = "Precursor M/Z Distribution",
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.25,
    fill: bool = True,
    save_path: str = None,
) -> None:
    """
    Plot combined Precursor M/Z distribution across all datasets.

    Parameters:
    -----------
    all_tables : dict
        Dictionary of all feature tables per tool (e.g., {"metaboscape": [df1, df2, ...], "mzmine": [df3, df4, ...], ...})
    title : str
        Title for the plot
    figsize : tuple
        Size of the figure
    alpha : float
        Transparency for density plots
    fill : bool
        Whether to fill under density curves
    save_path : str, optional
        Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for tool, combined_df in all_tables.items():
        all_precursors = combined_df["M/Z"].tolist()
        if not all_precursors:
            continue

        sns.kdeplot(
            all_precursors,
            ax=ax,
            label=f"{TOOL_NAMES.get(tool, tool)} ({len(all_precursors):,})",
            fill=fill,
            alpha=alpha,
            color=TOOL_COLORS.get(tool, "black"),
            cut=0,
        )

    ax.set_title(title, fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.set_xlabel("Precursor M/Z", fontsize=18, fontweight="bold")
    ax.set_ylabel("Density", fontsize=18, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)

    ax.set_yticks([])
    ax.legend(
        title="Workflow (# MS1 Features)",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_ms2_quality(
    all_tables: dict,
    title: str = "MS/MS Quality",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
) -> None:
    """
    Plot combined MS/MS quality fraction across all datasets.

    Parameters:
    -----------
    all_tables : dict
        Dictionary of all feature tables per tool (e.g., {"metaboscape": [df1, df2, ...], "mzmine": [df3, df4, ...], ...})
    title : str
        Title for the plot
    figsize : tuple
        Size of the figure
    save_path : str, optional
        Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for tool, combined_df in all_tables.items():
        # Combine all MS2 data and compute quality fraction
        total_good = 0
        total_assessable = 0

        df_ms2 = combined_df[combined_df["MS/MS_ASSIGNED"]]
        if len(df_ms2) == 0:
            continue

        ms2_quality = assess_ms2_quality(df_ms2)
        good_count = len(ms2_quality[ms2_quality["MS2_QC"] == "GOOD"])
        assessable_count = len(
            ms2_quality[
                (ms2_quality["MS2_QC"] == "GOOD") | (ms2_quality["MS2_QC"] == "BAD")
            ]
        )
        total_good += good_count
        total_assessable += assessable_count

        if total_assessable > 0:
            ms2_quality_fraction = total_good / total_assessable
        else:
            ms2_quality_fraction = 0.0

        ax.bar(
            x=TOOL_NAMES.get(tool, tool),
            height=ms2_quality_fraction,
            color=TOOL_COLORS.get(tool, "black"),
        )
        ax.bar_label(
            ax.containers[idx],
            fmt="%.2f",
            label_type="edge",
            padding=3,
            fontsize=18,
        )
        idx += 1

    ax.set_title(title, fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.set_ylabel(
        "Fraction of High-Quality\n MS2 Spectra", fontsize=15, fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_peak_count_distribution(
    all_tables: dict,
    title: str = "Peak Count Distribution",
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.25,
    fill: bool = True,
    save_path: str = None,
) -> None:
    """
    Plot combined Peak Count distribution across all datasets.

    Parameters:
    -----------
    all_tables : dict
        Dictionary of all feature tables per tool (e.g., {"metaboscape": [df1, df2, ...], "mzmine": [df3, df4, ...], ...})
    title : str
        Title for the plot
    figsize : tuple
        Size of the figure
    alpha : float
        Transparency for density plots
    fill : bool
        Whether to fill under density curves
    save_path : str, optional
        Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for tool, combined_df in all_tables.items():
        df_ms2 = combined_df[
            combined_df["MS/MS_ASSIGNED"]
        ]  # Only consider features with MS/MS spectra
        peak_counts = df_ms2["MS/MS_INTENSITIES"].apply(len).tolist()

        if not peak_counts:
            continue

        sns.kdeplot(
            peak_counts,
            ax=ax,
            label=f"{TOOL_NAMES.get(tool, tool)} ({len(peak_counts):,})",
            fill=fill,
            alpha=alpha,
            color=TOOL_COLORS.get(tool, "black"),
            cut=0,
            clip=(0, 250),
        )

    ax.set_title(
        title,
        fontdict={"fontsize": 20, "fontweight": "bold"},
    )
    ax.set_xlabel("MS2 Fragment Counts", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 250)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_ylabel("Density", fontsize=18, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)

    ax.set_yticks([])
    ax.legend(
        title="Workflow (# MS2 Features)",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_sample_missingness(
    all_tables: dict,
    title: str = "Sample Missingness",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
) -> None:
    """
    Plot combined Sample Missingness across all datasets.
    For each tool, computes the average missingness curve across all datasets.

    Parameters:
    -----------
    all_tables : dict
        Dictionary of dataframes for each tool
    title : str
        Title for the plot
    figsize : tuple
        Size of the figure
    title : str
        Title for the plot
    figsize : tuple
        Size of the figure
    include_tools : list, optional
        List of tools to include. If None, includes all.
    save_path : str, optional
        Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    tools = list(all_tables.keys())
    tick_labels = []

    # remove feature columns
    cols_to_drop_base = [
        "FEATURE_ID",
        "M/Z",
        "RT",
        "ION_MOBILITY",
        "CCS",
        "ADDUCT",
        "MS/MS_ASSIGNED",
        "MS/MS_MZS",
        "MS/MS_INTENSITIES",
        LIBRARY_ID_COLUMN,
        INCHIKEY_COLUMN,
        SCORE_COLUMN,
        LIBRARY_PRECURSOR_MZ_COLUMN,
        LIBRARY_MZS_COLUMN,
        LIBRARY_INTENSITIES_COLUMN,
        "CLIQUE_ID",
    ]

    for pos, tool in enumerate(tools, start=1):
        combined_df = all_tables[tool]
        combined_df.drop(columns=cols_to_drop_base, errors="ignore", inplace=True)

        total_counts = combined_df.shape[1]
        missing_counts = combined_df.isnull().sum(axis=1)
        zero_counts = (combined_df == 0).sum(axis=1)
        total_missingness_fractions = (
            total_counts - (missing_counts + zero_counts)
        ) / total_counts

        # Convert to percentage for better readability
        total_missingness_fractions *= 100

        ax.boxplot(
            total_missingness_fractions,
            positions=[pos],
            patch_artist=True,
            boxprops=dict(facecolor=TOOL_COLORS.get(tool, "#CCCCCC")),
            medianprops=dict(color="black", linewidth=1.5),
            showfliers=False,
            widths=0.5,
        )
        tick_labels.append(TOOL_NAMES.get(tool, tool.capitalize()))

    ax.set_xlim(0.5, len(tools) + 0.5)
    ax.set_xticks(range(1, len(tools) + 1))
    ax.set_xticklabels(tick_labels, fontsize=16)

    ax.set_title(title, fontdict={"fontsize": 20, "fontweight": "bold"})
    # ax.set_xlabel("Tool", fontsize=15, fontweight="bold")
    ax.set_ylabel("Missingness fraction (%)", fontsize=15, fontweight="bold")

    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_clique_metrics(
    combined_metrics: dict,
    title: str = "Clique Metrics",
    figsize: tuple = (14, 8),
    save_path: str = None,
):
    """
    Plot combined clique metrics across all datasets.
    Sums self_match_count, group_clique_count, and total_clique_count across datasets,
    then computes group_clique_fraction = sum(group_clique_count) / sum(total_clique_count).

    Parameters:
    -----------
    combined_metrics : dict
        Dictionary of combined metrics across all datasets
    title : str
        Title for the plot
    figsize : tuple
        Size of the figure
    save_path : str, optional
        Directory to save the figure
    """
    # Compute group_clique_fraction and prepare metrics for plotting
    plot_metrics = {}
    for tool, metrics in combined_metrics.items():
        total_cliques = metrics["total_clique_count"]
        group_cliques = metrics["group_clique_count"]
        group_clique_fraction = (
            group_cliques / total_cliques if total_cliques > 0 else 0.0
        )

        plot_metrics[tool] = {
            "Self Match Count": metrics["self_match_count"],
            "Group Clique Fraction": group_clique_fraction,
            "Total Clique Count": metrics["total_clique_count"],
        }

    # Convert to DataFrame for plotting
    df = pd.DataFrame(plot_metrics).T
    df = df.reset_index().rename(columns={"index": "Tool"})
    df = df.melt(id_vars=["Tool"], var_name="Metric", value_name="Value")

    # Create single figure with all metrics on one axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique metrics and tools
    metrics = df["Metric"].unique()
    tools = sorted(df["Tool"].unique())

    # Bar height and positions
    bar_height = 0.15
    y_positions = np.arange(len(metrics))

    # Plot bars for each tool
    for idx, tool in enumerate(tools):
        tool_data = df[df["Tool"] == tool]
        # Sort by metric to align with y_positions
        tool_data = tool_data.set_index("Metric").reindex(metrics)
        offset = (idx - len(tools) / 2 + 0.5) * bar_height

        bars = ax.barh(
            y_positions + offset,
            tool_data["Value"].values,
            bar_height,
            label=TOOL_NAMES.get(tool, tool),
            color=TOOL_COLORS.get(tool, "#CCCCCC"),
        )

        # Add value labels
        for i, (value, bar) in enumerate(zip(tool_data["Value"].values, bars)):
            label_text = f"{value:,.0f}" if value >= 1 else f"{value:.3f}"
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"  {label_text}",
                va="center",
                ha="left",
                fontsize=20,
            )

    # Set y-axis ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        ["Self Match\n Count", "Group\n Clique Fraction", "Total\n Clique Count"],
        fontsize=25,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xscale("log")

    ax.tick_params(axis="x", labelsize=25)

    ax.set_title(title, fontdict={"fontsize": 30, "fontweight": "bold"}, y=1.05)

    # Add legend with formatted tool names
    ax.legend(
        # title="Workflow",
        # title_fontsize=20,
        fontsize=18,
        frameon=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_pr_curves(
    score_range: np.ndarray,
    precision_recall_curves: dict,
    confusion_matrix_data: dict,
    x_var: str,
    y_var: str,
    confusion_matrix_threshold: float,
    title: str = "Precision-Recall Curves",
    fig_size=(10, 10),
    save_path=None,
    ax=None,
):
    """
    Plots precision-recall curves for multiple tools.


    Parameters:
    -----------
    score_range (np.ndarray):
        Range of score thresholds used for the curves
    precision_recall_curves (dict):
        Dictionary containing precision, recall, f1, and score data for each tool
    confusion_matrix_data (dict):
        Dictionary containing confusion matrix metrics (TP, FP, FN, F1) for each tool
    x_var (str):
        Variable to plot on the x-axis ('precision', 'recall', 'f1', or 'score')
    y_var (str):
        Variable to plot on the y-axis ('precision', 'recall', 'f1', or 'score')
    confusion_matrix_threshold (float):
        Threshold for the confusion matrix
    title (str, optional):
        Title of the plot. Defaults to "Precision-Recall Curves".
    fig_size (tuple, optional):
        Size of the figure. Defaults to (10, 10).
    save_path (str, optional):
        Path to save the figure. Defaults to None.
    ax (matplotlib.axes.Axes, optional):
        Axes to draw on. If None, a new figure is created. Defaults to None.

    """
    _fig_created = ax is None
    if _fig_created:
        _, ax = plt.subplots(figsize=fig_size)
    lines = {}

    if x_var not in ["precision", "recall", "f1", "score"]:
        raise ValueError("x_var must be 'precision', 'recall', 'f1', or 'score'")

    if y_var not in ["precision", "recall", "f1", "score"]:
        raise ValueError("y_var must be 'precision', 'recall', 'f1', or 'score'")

    axes_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "score": "MS2 similarity threshold",
    }

    for item in precision_recall_curves.items():
        if x_var != "score":
            x_data = item[1][x_var]
        else:
            x_data = score_range

        if y_var != "score":
            y_data = item[1][y_var]
        else:
            y_data = score_range

        (line,) = ax.plot(
            x_data,
            y_data,
            label=TOOL_NAMES.get(item[0], item[0]),
            color=TOOL_COLORS.get(item[0], None),
            linewidth=3,
        )

        ax.set_xlim(0.00, 1.01)
        ax.set_xlabel(axes_labels[x_var], fontsize=18, fontweight="bold")
        ax.set_ylabel(axes_labels[y_var], fontsize=18, fontweight="bold")

        threshold_recall = confusion_matrix_data[item[0]]["recall"]
        threshold_precision = confusion_matrix_data[item[0]]["precision"]
        threshold_score = confusion_matrix_threshold

        threshold_types = {
            "precision": threshold_precision,
            "recall": threshold_recall,
            "score": threshold_score,
        }

        if x_var != "score":
            ax.scatter(
                threshold_types[x_var],
                threshold_types[y_var],
                marker="x",
                color="black",
                s=100,
            )

        # store line handles to access colors later
        lines[item[0]] = line

    ax.legend(
        loc="best",
        fontsize=16,
        title_fontsize=18,
    )

    ax.set_xticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_xticklabels([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], fontsize=16)
    ax.set_yticklabels([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], fontsize=16)

    if x_var == "score":
        ax.axvline(x=confusion_matrix_threshold, color="black", linestyle="--")

    # Now add annotation boxes below the plot
    ymin, ymax = ax.get_ylim()
    ypos = ymin - 0.1 * (ymax - ymin)  # starting below the axis
    xpos_step = 1.0 / len(confusion_matrix_data)  # equally spaced across x-axis
    for i, (tool, metrics) in enumerate(confusion_matrix_data.items()):
        textstr = (
            f"TP={metrics['TP']:,}\n"
            f"FP={metrics['FP']:,}\n"
            f"FN={metrics['FN']:,}\n"
            f"F1 score={metrics['f1']:.3f}"
        )
        ax.text(
            i * xpos_step + xpos_step / 2,
            ypos,
            textstr,
            ha="center",
            va="top",
            color=lines[tool].get_color(),
            bbox=dict(
                facecolor="white",
                edgecolor=lines[tool].get_color(),
                boxstyle="round,pad=0.5",
            ),
            fontdict={"fontsize": 12, "fontweight": "bold"},
        )

    ax.set_title(title, size=20, pad=15, fontweight="bold")
    # Expand plot so text fits
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.01, 1.00)

    if _fig_created and save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_specific_pr_curves(
    score_range: np.ndarray,
    precision_recall_curves: dict,
    confusion_matrix_data: dict,
    x_var: str,
    y_var: str,
    confusion_matrix_threshold: float,
    title: str = "Precision-Recall Curves",
    fig_size=(10, 10),
    save_path=None,
    ax=None,
):
    """
    Plots precision-recall curves for multiple tools.


    Parameters:
    -----------
    score_range (np.ndarray):
        Range of score thresholds used for the curves
    precision_recall_curves (dict):
        Dictionary containing precision, recall, f1, and score data for each tool
    confusion_matrix_data (dict):
        Dictionary containing confusion matrix metrics (TP, FP, FN, F1) for each tool
    x_var (str):
        Variable to plot on the x-axis ('precision', 'recall', 'f1', or 'score')
    y_var (str):
        Variable to plot on the y-axis ('precision', 'recall', 'f1', or 'score')
    confusion_matrix_threshold (float):
        Threshold for the confusion matrix
    title (str, optional):
        Title of the plot. Defaults to "Precision-Recall Curves".
    fig_size (tuple, optional):
        Size of the figure. Defaults to (10, 10).
    save_path (str, optional):
        Path to save the figure. Defaults to None.
    ax (matplotlib.axes.Axes, optional):
        Axes to draw on. If None, a new figure is created. Defaults to None.

    """
    _fig_created = ax is None
    if _fig_created:
        _, ax = plt.subplots(figsize=fig_size)
    lines = {}

    if x_var not in ["precision", "recall", "f1", "score"]:
        raise ValueError("x_var must be 'precision', 'recall', 'f1', or 'score'")

    if y_var not in ["precision", "recall", "f1", "score"]:
        raise ValueError("y_var must be 'precision', 'recall', 'f1', or 'score'")

    axes_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "score": "MS2 similarity threshold",
    }

    for item in precision_recall_curves.items():
        if x_var != "score":
            x_data = item[1][x_var]
        else:
            x_data = score_range

        if y_var != "score":
            y_data = item[1][y_var]
        else:
            y_data = score_range

        (line,) = ax.plot(
            x_data,
            y_data,
            label=TOOL_NAMES.get(item[0], item[0]),
            color=TOOL_COLORS.get(item[0], None),
            linewidth=3,
        )

        ax.set_xlim(0.00, 1.01)
        ax.set_xlabel(axes_labels[x_var], fontsize=18, fontweight="bold")
        ax.set_ylabel(axes_labels[y_var], fontsize=18, fontweight="bold")

        threshold_recall = confusion_matrix_data[item[0]]["recall"]
        threshold_precision = confusion_matrix_data[item[0]]["precision"]
        threshold_score = confusion_matrix_threshold

        threshold_types = {
            "precision": threshold_precision,
            "recall": threshold_recall,
            "score": threshold_score,
        }

        if x_var != "score":
            ax.scatter(
                threshold_types[x_var],
                threshold_types[y_var],
                marker="x",
                color="black",
                s=100,
            )

        # store line handles to access colors later
        lines[item[0]] = line

    ax.legend(
        loc="best",
        fontsize=16,
        title_fontsize=18,
    )

    ax.set_xticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_xticklabels([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], fontsize=16)
    ax.set_yticklabels([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], fontsize=16)

    if x_var == "score":
        ax.axvline(x=confusion_matrix_threshold, color="black", linestyle="--")

    # Now add annotation boxes below the plot
    ymin, ymax = ax.get_ylim()
    ypos = ymin - 0.1 * (ymax - ymin)  # starting below the axis
    xpos_step = 1.0 / len(confusion_matrix_data)  # equally spaced across x-axis
    for i, (tool, metrics) in enumerate(confusion_matrix_data.items()):
        tp = metrics["TP"]
        fn = metrics["FN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        textstr = f"Recall = {tp}/{tp+fn}\n" f"  = {recall:.3f}\n"
        ax.text(
            i * xpos_step + xpos_step / 2,
            ypos,
            textstr,
            ha="center",
            va="top",
            color=lines[tool].get_color(),
            bbox=dict(
                facecolor="white",
                edgecolor=lines[tool].get_color(),
                boxstyle="round,pad=0.5",
            ),
            fontdict={"fontsize": 12, "fontweight": "bold"},
        )

    ax.set_title(title, size=20, pad=15, fontweight="bold")
    # Expand plot so text fits
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.01, 1.00)

    if _fig_created and save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_pr_curves_multi(
    score_range: dict,
    all_precision_recall_curves: dict,
    all_confusion_matrix_data: dict,
    x_var: str,
    y_var: str,
    confusion_matrix_threshold: float,
    title: str = "Precision-Recall Curves",
    fig_size=(10, 10),
    save_path: str = None,
    show_tool_names: bool = False,
):
    """
    Plot precision/recall curves for multiple (tool, annotation_type) pairs.

    Parameters:
    -----------
    score_range (dict):
        Dictionary mapping annotation types to score threshold ranges
    all_precision_recall_curves (dict):
        Nested dictionary containing precision, recall, f1, and score data for each (tool, annotation_type) pair
    all_confusion_matrix_data (dict):
        Nested dictionary containing confusion matrix metrics (TP, FP, FN, F1) for each (tool, annotation_type) pair
    x_var (str):
        Variable to plot on the x-axis ('precision', 'recall', 'f1', or 'score')
    y_var (str):
        Variable to plot on the y-axis ('precision', 'recall', 'f1', or 'score')
    confusion_matrix_threshold (float):
        Threshold for the confusion matrix
    title (str, optional):
        Title of the plot. Defaults to "Precision-Recall Curves".
    fig_size (tuple, optional):
        Size of the figure. Defaults to (10, 10).
    save_path (str, optional):
        Path to save the figure. Defaults to None.
    show_tool_names (bool, optional):
        Whether to include tool names in the legend labels. Defaults to False.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    lines = {}

    if x_var not in ["precision", "recall", "f1", "score"]:
        raise ValueError("x_var must be 'precision', 'recall', 'f1', or 'score'")
    if y_var not in ["precision", "recall", "f1", "score"]:
        raise ValueError("y_var must be 'precision', 'recall', 'f1', or 'score'")

    axes_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "score": "MS2 similarity threshold",
    }

    plotted_pairs = []

    for annotation_type, precision_recall_curves in all_precision_recall_curves.items():
        confusion_matrix_data = all_confusion_matrix_data.get(annotation_type, {})

        for tool, curves in precision_recall_curves.items():
            if x_var != "score":
                x_data = curves[x_var]
            else:
                x_data = score_range[annotation_type]

            if y_var != "score":
                y_data = curves[y_var]
            else:
                y_data = score_range[annotation_type]

            # Create label
            annot_label = ANNOTATION_NAMES.get(annotation_type, annotation_type)

            if show_tool_names:
                tool_label = TOOL_NAMES.get(tool, tool)
                label = f"{tool_label} ({annot_label})"
            else:
                label = annot_label

            (line,) = ax.plot(
                x_data,
                y_data,
                label=label,
                color=ANNOTATION_COLORS.get(annot_label, None),
                linewidth=3,
            )

            ax.set_xlim(0.00, 1.01)
            ax.set_xlabel(axes_labels[x_var], fontsize=18, fontweight="bold")
            ax.set_ylabel(axes_labels[y_var], fontsize=18, fontweight="bold")

            # Store line for confusion matrix boxes
            pair_key = (tool, annotation_type)
            lines[pair_key] = line
            plotted_pairs.append(pair_key)

            # Add threshold marker
            if x_var != "score" and tool in confusion_matrix_data:
                threshold_recall = confusion_matrix_data[tool]["recall"]
                threshold_precision = confusion_matrix_data[tool]["precision"]
                threshold_types = {
                    "precision": threshold_precision,
                    "recall": threshold_recall,
                    "score": confusion_matrix_threshold,
                }
                ax.scatter(
                    threshold_types[x_var],
                    threshold_types[y_var],
                    marker="x",
                    color=TOOL_COLORS.get(tool, "black"),
                    s=100,
                )

    ax.set_xticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.set_xticklabels([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], fontsize=16)
    ax.set_yticklabels([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], fontsize=16)

    ax.legend(
        loc="upper left",
        fontsize=16,
        title_fontsize=18,
    )

    if x_var == "score":
        ax.axvline(
            x=confusion_matrix_threshold, color="black", linestyle="--", alpha=0.5
        )

    # Add confusion matrix boxes if requested
    if plotted_pairs:
        ymin, ymax = ax.get_ylim()
        ypos = ymin - 0.1 * (ymax - ymin)
        xpos_step = 1.0 / len(plotted_pairs)

        for i, (tool, annotation_type) in enumerate(plotted_pairs):
            confusion_matrix_data = all_confusion_matrix_data.get(annotation_type, {})
            if tool not in confusion_matrix_data:
                continue
            metrics = confusion_matrix_data[tool]
            textstr = (
                f"TP={metrics['TP']:,}\n"
                f"FP={metrics['FP']:,}\n"
                f"FN={metrics['FN']:,}\n"
                f"F1 score={metrics['f1']:.3f}"
            )
            ax.text(
                i * xpos_step + xpos_step / 2,
                ypos,
                textstr,
                ha="center",
                va="top",
                color=lines[(tool, annotation_type)].get_color(),
                bbox=dict(
                    facecolor="white",
                    edgecolor=lines[(tool, annotation_type)].get_color(),
                    boxstyle="round,pad=0.3",
                ),
                fontdict={"fontsize": 12, "fontweight": "bold"},
            )

    ax.set_title(title, size=20, pad=15, fontweight="bold")

    # Expand plot so text fits
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.01, 1.00)

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_intensity_distributions(
    plot_data: list,
    save_path=None,
    figsize=(20, 5),
    suptitle_text="Feature Intensity Distributions Across Workflows",
    suptitle_fontsize=50,
    suptitle_fontweight="bold",
):
    """Plots intensity distributions for detected features and true positives across multiple tools.

    Parameters:
    -----------
    plot_data : list
        List of tuples containing (tool_name, all_intensities, true_positive_intensities)
    save_path : str, optional
        Path to save the figure. Defaults to None.
    figsize : tuple, optional
        Size of the figure. Defaults to (20, 5).
    suptitle_text : str, optional
        Title for the entire figure. Defaults to "Feature Intensity Distributions Across Workflows".
    suptitle_fontsize : int, optional
        Font size for the suptitle. Defaults to 50.
    """
    # Creating one combined plot with all tools' intensity distributions
    fig, ax = plt.subplots(figsize=figsize, sharey=True, nrows=1, ncols=3)

    idx = 0
    for tool, all_intensities, true_positive_intensities in plot_data:
        sns.kdeplot(
            all_intensities,
            fill=True,
            alpha=0.3,
            log_scale=True,
            color="gray",
            ax=ax[idx],
        )
        sns.kdeplot(
            true_positive_intensities,
            fill=True,
            alpha=0.3,
            log_scale=True,
            color="green",
            ax=ax[idx],
        )
        ax[idx].set_title(
            f"{TOOL_NAMES.get(tool, tool)}",
            fontdict={"fontsize": 18},
        )

        ax[idx].set_ylabel("Density", fontdict={"fontsize": 18})
        ax[idx].set_xlim(1e2, 1e8)
        ax[idx].set_yticks([])
        # ax[idx].set_xlabel("")
        ax[idx].tick_params(axis="x", labelsize=15)
        idx += 1

    legend_labels = ["Detected Features", "True Positives"]

    legend_colors = ["gray", "green"]
    handles = [
        plt.Line2D([0], [0], color=color, lw=10, alpha=0.3) for color in legend_colors
    ]

    fig.suptitle(
        suptitle_text,
        fontsize=suptitle_fontsize,
        fontweight=suptitle_fontweight,
        y=0.98,
    )

    # display legend below all subplots
    plt.subplots_adjust(top=0.75, bottom=0.2)

    ax[1].legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        title_fontsize=20,
        fontsize=18,
    )

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_true_positive_overlap(
    tool_inchikeys: dict,
    title: str = None,
    save_path: str = None,
    figsize: tuple = (8, 8),
):
    plt.figure(figsize=figsize)

    v = venn3(
        subsets=tool_inchikeys.values(),
        set_labels=tool_inchikeys.keys(),
        subset_label_formatter=lambda x: f"{int(x):,}",
        set_colors=[TOOL_COLORS.get(tool, "#CCCCCC") for tool in tool_inchikeys.keys()],
        alpha=0.6,
    )

    # Increase font size for set labels (circle labels)
    for label in v.set_labels:
        if label:
            label.set_fontsize(18)

    # Increase font size for subset count labels (region counts)
    for label in v.subset_labels:
        if label:
            label.set_fontsize(16)

    if title:
        plt.title(title, fontsize=20, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=400)


def plot_ccs_distribution(
    ccs_error_distributions,
    save_path=None,
    output_type="relative",
    show_points=False,
    point_jitter=0.25,
    point_size=2.5,
    point_alpha=0.45,
    figsize=(15, 8),
):
    """
    Plots the distribution of CCS errors for each tool using violin plots.

    Parameters:
    -----------
    ccs_error_distributions : dict
        Dictionary mapping tool names to lists of CCS errors (either relative or absolute)
    save_path : str, optional
        Path to save the figure. Defaults to None.
    output_type : str, optional
        Type of CCS error ('relative' for percentage error, 'absolute' for Å² error). Defaults to 'relative'.
    show_points : bool, optional
        Whether to overlay individual data points on the violin plot. Defaults to False.
    point_jitter : float, optional
        Amount of jitter to apply to individual points if show_points is True. Defaults to 0.25.
    point_size : float, optional
        Size of the individual points if show_points is True. Defaults to 2.5.
    point_alpha : float, optional
        Transparency of the individual points if show_points is True. Defaults to 0.45.
    figsize : tuple, optional
        Size of the figure. Defaults to (15, 8).
    """
    plt.figure(figsize=figsize)
    data_to_plot = []
    labels = []

    ccs_error_distributions = dict(sorted(ccs_error_distributions.items()))
    for tool, errors in ccs_error_distributions.items():
        data_to_plot.append(errors)
        labels.append(tool)

    plot_df = pd.DataFrame(
        {
            "Software": np.repeat(labels, [len(errors) for errors in data_to_plot]),
            "CCS Error (%)": np.concatenate(data_to_plot),
        }
    )

    sns.violinplot(
        x="Software",
        y="CCS Error (%)",
        data=plot_df,
        palette=TOOL_COLORS,
        cut=0,
    )

    if show_points:
        sns.stripplot(
            x="Software",
            y="CCS Error (%)",
            data=plot_df,
            color="black",
            jitter=point_jitter,
            size=point_size,
            alpha=point_alpha,
        )

    ax = plt.gca()
    ax.set_xticklabels(
        [TOOL_NAMES.get(label, label) for label in labels],
        fontdict={"fontsize": 18},
    )

    counts = plot_df.groupby("Software")["CCS Error (%)"].size().reindex(labels)

    ymin, ymax = ax.get_ylim()
    new_ymax = ymax + 20  # add some extra space for the count annotations
    ax.set_ylim(ymin, new_ymax)  # add some extra space for the count annotations
    y_text = new_ymax - 0.03 * (new_ymax - ymin)

    for i, (tool, n) in enumerate(counts.items()):
        ax.text(i, y_text, f"n={n:,}", ha="center", va="top", fontsize=18)

    plt.xlabel("")
    plt.ylabel(
        "CCS Error (%)" if output_type == "relative" else "CCS Error (Å²)",
        fontsize=18,
        fontweight="bold",
    )
    plt.yticks(fontsize=18)
    plt.title("CCS Error Distributions", fontsize=20, fontweight="bold", pad=15)

    if save_path:
        plt.savefig(save_path, dpi=400)


def plot_zoomed_ccs_distribution(
    ccs_error_distributions,
    save_path=None,
    output_type="relative",
    show_points=False,
    point_jitter=0.25,
    point_size=2.5,
    point_alpha=0.45,
    figsize=(15, 8),
    ymax=None,
):
    """
    Plots the distribution of CCS errors for each tool using violin plots alongside their 1% and 5% cutoff values.

    Parameters:
    -----------
    ccs_error_distributions : dict
        Dictionary mapping tool names to lists of CCS errors (either relative or absolute)
    save_path : str, optional
        Path to save the figure. Defaults to None.
    output_type : str, optional
        Type of CCS error ('relative' for percentage error, 'absolute' for Å² error). Defaults to 'relative'.
    show_points : bool, optional
        Whether to overlay individual data points on the violin plot. Defaults to False.
    point_jitter : float, optional
        Amount of jitter to apply to individual points if show_points is True. Defaults to 0.25.
    point_size : float, optional
        Size of the individual points if show_points is True. Defaults to 2.5.
    point_alpha : float, optional
        Transparency of the individual points if show_points is True. Defaults to 0.45.
    figsize : tuple, optional
        Size of the figure. Defaults to (15, 8).
    ymax : float, optional
        Maximum value for the y-axis. If None, it will be set automatically based on the data. Defaults to None.
    """
    plt.figure(figsize=figsize)
    data_to_plot = []
    labels = []

    ccs_error_distributions = dict(sorted(ccs_error_distributions.items()))
    for tool, errors in ccs_error_distributions.items():
        data_to_plot.append(errors)
        labels.append(tool)

    plot_df = pd.DataFrame(
        {
            "Software": np.repeat(labels, [len(errors) for errors in data_to_plot]),
            "CCS Error (%)": np.concatenate(data_to_plot),
        }
    )

    sns.violinplot(
        x="Software",
        y="CCS Error (%)",
        data=plot_df,
        palette=TOOL_COLORS,
        # inner="box",
        cut=0,
    )

    if show_points:
        sns.stripplot(
            x="Software",
            y="CCS Error (%)",
            data=plot_df,
            color="black",
            jitter=point_jitter,
            size=point_size,
            alpha=point_alpha,
        )

    ax = plt.gca()
    ax.set_xticklabels(
        [TOOL_NAMES.get(label, label) for label in labels],
        fontdict={"fontsize": 18},
    )

    # Add red horizontal lines at ±1% and ±5%
    cutoffs = [1, 5]
    linestyles = ["--", "-"]
    for cutoff, ls in zip(cutoffs, linestyles):
        ax.axhline(y=cutoff, color="black", linestyle=ls, linewidth=1.2, alpha=0.8)
        ax.axhline(y=-cutoff, color="black", linestyle=ls, linewidth=1.2, alpha=0.8)

    # Annotate each violin with the % within 1% and 5% cutoffs
    for i, (tool, errors) in enumerate(zip(labels, data_to_plot)):
        errors_arr = np.array(errors)
        pct_1 = np.mean(np.abs(errors_arr) <= 1) * 100
        pct_5 = np.mean(np.abs(errors_arr) <= 5) * 100

        # Position text just above each cutoff line for this violin
        x_pos = i
        ax.text(
            x_pos + 0.1,
            1.15,
            f"{pct_1:.0f}%",
            ha="left",
            va="bottom",
            fontsize=10,
            color="black",
            fontweight="bold",
        )
        ax.text(
            x_pos + 0.1,
            5.15,
            f"{pct_5:.0f}%",
            ha="left",
            va="bottom",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    if ymax is not None:
        ax.set_ylim(-ymax, ymax)

    plt.xlabel("")
    plt.ylabel(
        "CCS Error (%)" if output_type == "relative" else "CCS Error (Å²)",
        fontsize=18,
        fontweight="bold",
    )
    plt.yticks(fontsize=18)
    plt.title("CCS Error Distributions", fontsize=20, fontweight="bold", pad=15)

    if save_path:
        plt.savefig(save_path, dpi=400)


def plot_varied_tolerance_metrics(
    varied_cosine_output,
    ms_tol_type,
    metric,
    title,
    save_path=None,
    display_argmax=False,
    xaxis_label=None,
    yaxis_label=None,
    figsize=(10, 6),
    y_max=None,
):
    """
    Plot a metric vs MS2 tolerance for each tool.

    Parameters:
    -----------
    varied_cosine_output : dict
        Nested dictionary containing metric values for each tool and MS tolerance type
    ms_tol_type : str
        Type of MS tolerance ('ms1' or 'ms2')
    metric : str
        Metric to plot (e.g., 'precision', 'recall', 'f1', 'cosine_similarity', etc.)
    title : str
        Title for the plot
    save_path : str, optional
        Directory to save the figure
    display_argmax : bool, optional
        Whether to annotate the point with the maximum metric value for each tool
    xaxis_label : str, optional
        Custom label for the x-axis. If None, a default label based on ms_tol_type will be used.
    yaxis_label : str, optional
        Custom label for the y-axis. If None, a default label based on the metric name will be used.
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to (10, 6).
    y_max : float, optional
        Maximum value for the y-axis. If None, it will be set automatically based on the metric type
    """
    fig, ax = plt.subplots(figsize=figsize)

    # create a df with tool, tolerance, and metric value for easier plotting with seaborn
    data = []

    for tool, ms_tol_dict in varied_cosine_output.items():
        for tol, metrics in ms_tol_dict[ms_tol_type].items():
            data.append(
                {"tool": tool, "tolerance": float(tol), "metric_value": metrics[metric]}
            )

    df = pd.DataFrame(data)

    # Sort the data by tolerance for each tool
    sorted_df = df.sort_values(by=["tolerance", "tool"])

    for tool in sorted_df["tool"].unique():
        tool_df = sorted_df[sorted_df["tool"] == tool]
        ax.plot(
            tool_df["tolerance"],
            tool_df["metric_value"],
            label=TOOL_NAMES.get(tool, tool),
            color=TOOL_COLORS.get(tool, None),
            marker="o",
            markersize=4,
        )

        # Label the argmax point with the tolerance value
        if display_argmax:
            max_idx = np.argmax(tool_df["metric_value"])
            max_tolerance = tool_df["tolerance"].iloc[max_idx]
            max_value = tool_df["metric_value"].iloc[max_idx]
            ax.annotate(
                f"{max_tolerance:.2g}",
                xy=(max_tolerance, max_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=TOOL_COLORS.get(tool, "black"),
                fontweight="bold",
            )

    if xaxis_label is None:
        xaxis_label = (
            f"{ms_tol_type.upper()} Tolerance (ppm)"
            if ms_tol_type == "ms1"
            else f"{ms_tol_type.upper()} Tolerance (Da)"
        )

    if yaxis_label is None:
        yaxis_label = metric.replace("_", " ").title()

    ax.set_xlabel(xaxis_label, fontsize=18, fontweight="bold")
    ax.set_ylabel(yaxis_label, fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Set y-axis limits based on metric type
    if "threshold" not in metric:
        ax.set_ylim(0, 1.05)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")


def plot_r2_distribution(
    r2_dict,
    significant_only=False,
    significance_threshold=0.05,
    jitter=0.35,
    save_path=None,
    title=None,
    figsize=(10, 6),
    ax=None,
):
    """
    Plot r² distributions as violin plots for all tools.

    Parameters:
    -----------
    r2_dict : dict
        Dictionary containing r² values and p-values for each tool (e.g., {"metaboscape": [(r2_value1, p_value1), ...], "mzmine": [...], ...})
    significant_only : bool, optional
        Whether to include only significant r² values based on the p-value threshold. Defaults to False (i.e., include all values).
    significance_threshold : float, optional
        P-value threshold for determining significance when significant_only is True. Defaults to 0.05.
    jitter : float, optional
        Amount of jitter to apply to the points in the strip plot. Defaults to 0.35.
    save_path : str, optional
        Directory to save the figure. Defaults to None.
    title : str, optional
        Title for the plot. If None, a default title will be generated based on whether significant_only is True or False.
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to (10, 6).
    """
    # Collect all r² values into a DataFrame for violin plot
    all_data = []

    for tool_name, values in r2_dict.items():
        for r2_value, p_value in values:
            # Skip non-significant values if requested
            if significant_only and p_value >= significance_threshold:
                continue

            all_data.append(
                {
                    "Tool": TOOL_NAMES.get(tool_name, tool_name),
                    "tool_key": tool_name,
                    "r²": r2_value,
                    "p_value": p_value,
                }
            )

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data to plot.")
        return

    # Count number of points per tool
    tool_counts = df.groupby("Tool").size()

    _fig_created = ax is None
    if _fig_created:
        plt.figure(figsize=figsize)

    # Create violin plot
    ax = sns.violinplot(
        data=df,
        x="Tool",
        y="r²",
        palette={TOOL_NAMES.get(k, k): v for k, v in TOOL_COLORS.items()},
        inner=None,  # No inner box, we'll add stripplot instead
        cut=0,
        alpha=0.3,
        ax=ax,
    )

    # Add jittered points on top
    sns.stripplot(
        data=df,
        x="Tool",
        y="r²",
        palette={TOOL_NAMES.get(k, k): v for k, v in TOOL_COLORS.items()},
        jitter=jitter,
        size=6,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # Update x-axis labels to include counts
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    new_labels = [f"{label}\n(n={tool_counts[label]})" for label in current_labels]
    ax.set_xticklabels(new_labels, fontdict={"fontsize": 16})

    if title is None:
        if significant_only:
            title = f"Coefficient of Determination Distributions Across Tools\n(Significant only, p < {significance_threshold})"
        else:
            title = "Coefficient of Determination Distributions Across Tools"

    ax.set_xlabel("")
    ax.set_title(title, size=20, pad=15, fontweight="bold")
    ax.set_ylabel("Coefficient of Determination (r²)", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, alpha=0.3, axis="y")

    if _fig_created:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()


def plot_transform_combinations_lineplot(
    merged_df,
    ground_truth_data_combined,
    combinations,
    concentrations: list = [50, 100, 200],
    score_threshold=0.7,
    hist_width=1.6,
    figsize=(12, 6),
    jitter=0.1,
    alpha=0.55,
    log_base_x=None,
    log_base_y=None,
    save_path=None,
):
    """
    Plot side-by-side boxplots (one per tool) at each concentration.

    Parameters:
    -----------
    merged_df : dict
        Dictionary containing merged DataFrames for each tool
    ground_truth_data_combined : pd.DataFrame
        Combined ground truth DataFrame with metabolite names and InChIKeys
    combinations : list of tuples
        List of (tool_name, transform_type) pairs to plot. transform_type can be 'rclr', 'clr', or 'raw'.
    concentrations : list, optional
        List of concentrations to include in the plot. Defaults to [50, 100, 200].
    score_threshold : float, optional
        Minimum score threshold for including a feature in the plot. Defaults to 0.7.
    hist_width : float, optional
        Width of the boxplots. Defaults to 1.6.
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to (12, 6).
    jitter : float, optional
        Amount of jitter to apply to individual points in the boxplot. Defaults to 0.1.
    alpha : float, optional
        Transparency level for the boxplots. Defaults to 0.55.
    log_base_x : int or None, optional
        If specified, apply log transformation to x-axis with the given base (e.g., 10). Defaults to None (no log transform).
    log_base_y : int or None, optional
        If specified, apply log transformation to y-axis with the given base (e.g., 10). Defaults to None (no log transform).
    save_path : str or None, optional
        If specified, save the plot to the given file path. Defaults to None (do not save).

    """
    # Formatted transform names for legends
    formatted_transform_names = {
        "rclr": "RCLR",
        "clr": "CLR",
        "raw": "Raw",
    }

    tools_in_order = [tool for tool, _ in combinations]
    unique_tools = list(
        dict.fromkeys(tools_in_order)
    )  # Preserve order while getting unique tools

    tool_offsets = {
        tool: (i - (len(unique_tools) - 1) / 2) * (hist_width * 2.6)
        for i, tool in enumerate(unique_tools)
    }

    fig, ax = plt.subplots(figsize=figsize)

    for tool_name, transform_type in combinations:
        if transform_type not in TRANSFORM_SETTINGS:
            print(
                f"Warning: Transform '{transform_type}' not recognized. Use 'rclr', 'clr', or 'raw'. Skipping."
            )
            continue

        tic_norm, transform_arg = TRANSFORM_SETTINGS[transform_type]

        tool_df = merged_df[tool_name]

        column_names = [col for col in tool_df.columns if "SAMPLE" in col]

        intensities_df = tool_df[column_names]
        if tic_norm:
            intensities_df = intensities_df.div(intensities_df.sum(axis=0), axis=1)
        if transform_arg == "rclr":
            intensities_df = intensities_df.apply(apply_rclr_transform, axis=0)
        elif transform_arg == "clr":
            intensities_df = intensities_df.apply(apply_clr_transform, axis=0)

        formatted_samples = defaultdict(dict)

        for _, column in enumerate(column_names):
            compound_name = column.split("_")[1]
            conc = int(column.split("_")[2].replace("ng", "").strip())
            if conc not in concentrations:
                continue

            formatted_samples[compound_name][conc] = column

        conc_to_values = {c: [] for c in concentrations}

        for _, mapping in formatted_samples.items():
            for conc, column in mapping.items():
                compound_name = column.split("_")[1]

                # Identify the non-isomeric InChIKey for the compound from the ground truth data
                inchikey_nonisomeric = list(
                    set(
                        ground_truth_data_combined[
                            ground_truth_data_combined["metabolite_name"]
                            == compound_name
                        ]["inchikey_2d"].values
                    )
                )

                matching_rows = tool_df[
                    (tool_df[INCHIKEY_COLUMN].isin(inchikey_nonisomeric))
                    & (tool_df[SCORE_COLUMN] >= score_threshold)
                ]

                if matching_rows.empty:
                    continue

                best_match = matching_rows.loc[matching_rows[SCORE_COLUMN].idxmax()]
                intensity_val = intensities_df.loc[best_match.name, column]
                conc_to_values[conc].append(intensity_val)

        # Optional log transform for y-axis (intensity) plotting
        if log_base_y is not None:
            log_func_y = np.log10 if log_base_y == 10 else np.log
            conc_to_values = {
                c: [log_func_y(v) for v in values if v > 0]
                for c, values in conc_to_values.items()
            }

        color = TOOL_COLORS.get(tool_name, "#333333")
        label = TOOL_NAMES.get(tool_name, tool_name)
        offset = tool_offsets.get(tool_name, 0)

        positions = [c + offset for c in concentrations]
        data = [conc_to_values[c] for c in concentrations]

        # Custom boxplot with mean ± std dev instead of median/IQR
        for pos, values in zip(positions, data):
            if len(values) == 0:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)

            # Box: mean ± 1 std dev
            box_lower = mean_val - std_val
            box_upper = mean_val + std_val

            # Whiskers: mean ± 2 std dev
            whisker_lower = mean_val - 2 * std_val
            whisker_upper = mean_val + 2 * std_val

            # Draw box (mean ± 1 std)
            box_width = hist_width
            box = plt.Rectangle(
                (pos - box_width / 2, box_lower),
                box_width,
                box_upper - box_lower,
                facecolor="white",
                edgecolor=color,
                linewidth=2.6,
                alpha=alpha,
            )
            ax.add_patch(box)

            # Draw mean line (horizontal)
            ax.plot(
                [pos - box_width / 2, pos + box_width / 2],
                [mean_val, mean_val],
                color=color,
                linewidth=2.6,
            )

            # Draw whiskers
            ax.plot([pos, pos], [box_upper, whisker_upper], color=color, linewidth=2.0)
            ax.plot([pos, pos], [box_lower, whisker_lower], color=color, linewidth=2.0)

            # Draw caps
            cap_width = box_width * 0.5
            ax.plot(
                [pos - cap_width / 2, pos + cap_width / 2],
                [whisker_upper, whisker_upper],
                color=color,
                linewidth=2.0,
            )
            ax.plot(
                [pos - cap_width / 2, pos + cap_width / 2],
                [whisker_lower, whisker_lower],
                color=color,
                linewidth=2.0,
            )

            # Outliers: points beyond ± 2 std
            outliers = [v for v in values if v < whisker_lower or v > whisker_upper]
            if outliers:
                ax.scatter(
                    [pos] * len(outliers),
                    outliers,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    alpha=0.7,
                )

        # Jittered points
        for x_center, values in zip(positions, data):
            if len(values) == 0:
                continue
            x_jitter = np.random.normal(loc=x_center, scale=jitter, size=len(values))
            ax.scatter(
                x_jitter,
                values,
                s=24,
                color=color,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.4,
            )

        # Trend line using linear fit over means
        means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in data]
        x_fit = np.array(concentrations, dtype=float)
        y_fit = np.array(means, dtype=float)
        mask = np.isfinite(y_fit)
        if mask.sum() >= 2:
            # Fit in log space if log_base_x is specified
            if log_base_x is not None:
                log_func_x = np.log10 if log_base_x == 10 else np.log
                x_fit_log = log_func_x(x_fit[mask])
                coeffs = np.polyfit(x_fit_log, y_fit[mask], 1)
                y_line = np.polyval(coeffs, log_func_x(x_fit))
            else:
                coeffs = np.polyfit(x_fit[mask], y_fit[mask], 1)
                y_line = np.polyval(coeffs, x_fit)

            ax.plot(
                [c + offset for c in concentrations],
                y_line,
                color="#7f7f7f",
                linewidth=2.2,
                linestyle="--",
            )

        # Legend handle per tool
        ax.plot([], [], color=color, linewidth=7, label=label)

    # Set x-axis scale and labels
    if log_base_x is not None:
        if log_base_x == 10:
            ax.set_xscale("log")
        else:
            ax.set_xscale("log", base=np.e)
        xlabel = f"Concentration (log scale)"
    else:
        xlabel = "Concentration"

    ax.set_xticks(concentrations)
    ax.set_xticklabels([f"{c} ng" for c in concentrations], fontsize=16)
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)

    transform_set = set([t for _, t in combinations])
    if len(transform_set) == 1:
        t = next(iter(transform_set))
        formatted_t = formatted_transform_names.get(t, t)
        ylabel = f"Intensity ({formatted_t})"
    else:
        ylabel = "Intensity (mixed transforms)"

    if log_base_y is not None:
        log_label = f"log{log_base_y}" if log_base_y == 10 else "log"
        ylabel = f"{log_label} {ylabel}"

    ax.set_ylabel(ylabel, fontsize=16)

    ax.set_title(
        "Intensity Distributions Across Concentrations", fontsize=18, fontweight="bold"
    )
    ax.grid(True, alpha=0.4, axis="y", color="white")
    # ax.legend(loc="best", fontsize=14, title_fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")

    plt.show()


def make_gapped_array(
    arr: List, intensities: bool = False, negative: bool = False
) -> np.array:
    """
    Create a gapped array for plotting.

    Args:
        arr (List): The input array.
        intensities (bool): Whether the array represents intensities.
        negative (bool): Whether to invert the array values.

    Returns:
        np.array: The gapped array.
    """
    arr = np.array(arr)

    if negative:  # needed for the inverted plot
        arr *= -1

    # [1,2] -> [1,1,1,2,2,2]
    arr = np.repeat(arr, 3)

    if intensities:
        # [2,2,2,3,3,3] -> [0,2,2,0,3,3]
        arr = [np.nan if i % 3 == 0 else x for i, x in enumerate(arr)]
        # [2,2,2,3,3,3] -> [2,0,2,3,0,3]
        arr = [0 if i % 3 == 1 else x for i, x in enumerate(arr)]
        return arr

    arr = [0 if i % 3 == 0 else x for i, x in enumerate(arr)]
    return arr


def normalize_intensities(intensities: np.array, upper_limit=1) -> np.array:
    """
    Normalize intensities to a given upper limit.

    Args:
        intensities (np.array): The input intensities.
        upper_limit (float): The upper limit for normalization.

    Returns:
        np.array: The normalized intensities.
    """
    return np.array(intensities) * upper_limit / max(intensities)


def calculate_axis_range_with_padding(
    arr1: np.array, arr2: Optional[np.array] = None, padding: float = 10
) -> Tuple[int, int]:
    """
    Calculate the range for a mirror plot.

    Args:
        arr1 (np.array): The first array.
        arr2 (np.array): The second array.
        padding (float): The padding to add to the range.

    Returns:
        Tuple[int, int]: The range for the x-axis.
    """
    # calculate the range of the x axis [leftmost point - 10, rightmost point + 10]
    if arr2 is None:
        return (
            int(min(arr1) - padding),
            int(max(arr1) + padding),
        )

    return (
        int(min([min(arr1), min(arr2)]) - padding),
        int(max([max(arr1), max(arr2)]) + padding),
    )


def make_spectrum_plot(
    spectra: List[dict],
    title: str = "Spectra",
    n_cols: int = 2,
    subplot_height: int = 300,
    x_max: float = 500,
) -> go.Figure:
    """
    Plot one or more spectra as a grid of subplots.

    When a spectrum dict contains ``ref_mzs`` and ``ref_intensities`` keys, the subplot
    becomes a mirror plot: the query spectrum is drawn on top (positive y) and the
    reference spectrum is drawn on the bottom (negative/inverted y), separated by a
    zero line.

    Args:
        spectra: List of dicts, each with keys:
            - ``mzs``             (np.ndarray): query fragment m/z values.
            - ``intensities``     (np.ndarray): query fragment intensities.
            - ``ref_mzs``         (np.ndarray, optional): reference fragment m/z values.
              If provided together with ``ref_intensities``, a mirror plot is rendered.
            - ``ref_intensities`` (np.ndarray, optional): reference fragment intensities.
            - ``title``           (str, optional): subplot title. Defaults to "Spectrum <i>".
            - ``tool``            (str, optional): tool name for consistent color lookup
              via ``TOOL_COLORS``. Falls back to ``"#CCCCCC"`` if absent or unknown.
        title: Overall figure title.
        n_cols: Number of columns in the subplot grid.
        subplot_height: Height in pixels for each row of subplots.
        x_max: Upper bound for the shared x-axis (m/z). Defaults to 500.

    Returns:
        go.Figure with one subplot per spectrum.
    """
    n = len(spectra)
    if n == 0:
        return go.Figure()

    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    subplot_titles = [
        s.get("title", f"Spectrum {i + 1}") for i, s in enumerate(spectra)
    ]

    # Compute a shared x-axis range across all spectra, capped at x_max
    all_mzs = np.concatenate(
        [np.asarray(s["mzs"]) for s in spectra if len(s.get("mzs", [])) > 0]
    )
    raw_range = calculate_axis_range_with_padding(all_mzs, padding=10)
    shared_xaxis_range = [raw_range[0], min(raw_range[1], int(x_max))]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Collect unique (tool, color) pairs in order of first appearance for the legend
    seen_tools: dict = {}
    for s in spectra:
        tool = s.get("tool")
        if tool and tool not in seen_tools:
            seen_tools[tool] = TOOL_COLORS.get(tool, "#CCCCCC")

    has_any_reference = False

    for i, spectrum in enumerate(spectra):
        row = i // n_cols + 1
        col = i % n_cols + 1

        mzs = np.asarray(spectrum["mzs"])
        intensities = np.asarray(spectrum["intensities"])

        if len(mzs) == 0 or len(intensities) == 0:
            continue

        ref_mzs_raw = spectrum.get("ref_mzs")
        ref_ints_raw = spectrum.get("ref_intensities")
        has_ref = (
            ref_mzs_raw is not None
            and ref_ints_raw is not None
            and len(ref_mzs_raw) > 0
            and len(ref_ints_raw) > 0
        )
        if has_ref:
            has_any_reference = True

        # Query spectrum (top / positive)
        norm_intensities = normalize_intensities(intensities)
        mzs_gapped = make_gapped_array(mzs)
        intensities_gapped = make_gapped_array(norm_intensities, True, False)

        tool = spectrum.get("tool")
        color = TOOL_COLORS.get(tool, "#CCCCCC") if tool else "#CCCCCC"

        fig.add_trace(
            go.Scatter(
                x=mzs_gapped,
                y=intensities_gapped,
                mode="lines",
                line=dict(color=color),
                hovertemplate="m/z: %{x:.4f}<br>Intensity: %{y:.3f}<extra>Query</extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Reference spectrum (bottom / negative mirror)
        if has_ref:
            ref_mzs_arr = np.asarray(ref_mzs_raw)
            ref_ints_arr = np.asarray(ref_ints_raw)
            norm_ref = normalize_intensities(ref_ints_arr)
            ref_mzs_gapped = make_gapped_array(ref_mzs_arr)
            ref_ints_gapped = make_gapped_array(
                norm_ref, intensities=True, negative=True
            )
            fig.add_trace(
                go.Scatter(
                    x=ref_mzs_gapped,
                    y=ref_ints_gapped,
                    mode="lines",
                    line=dict(color="#888888"),
                    hovertemplate="m/z: %{x:.4f}<br>Intensity: %{y:.3f}<extra>Reference</extra>",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        y_range = [-1.1, 1.1] if has_ref else [0, 1.1]
        axis_idx = "" if i == 0 else str(i + 1)
        fig.update_layout(
            {
                f"xaxis{axis_idx}": dict(title="m/z", range=shared_xaxis_range),
                f"yaxis{axis_idx}": dict(
                    title="Intensity",
                    range=y_range,
                    zeroline=has_ref,
                    zerolinewidth=1,
                    zerolinecolor="lightgrey",
                ),
            }
        )

    # Legend: one dummy trace per tool + one for Reference (if any mirror plots present)
    for tool, color in seen_tools.items():
        label = TOOL_NAMES.get(tool, tool)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=3),
                name=label,
                showlegend=True,
            )
        )

    if has_any_reference:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="#888888", width=3),
                name="Reference",
                showlegend=True,
            )
        )

    has_legend = bool(seen_tools) or has_any_reference
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=subplot_height * n_rows,
        template="simple_white",
        showlegend=has_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )
    return fig
