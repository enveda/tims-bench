annotation_method = "annotated_entropy_NEW"

quantification_df_paths = {
    "metaboscape": "../data/spike_in_datasets/spike_in_individual/harmonized/spike_in_individual_metaboscape_harmonized.parquet",
    "msdial": "../data/spike_in_datasets/spike_in_individual/harmonized/spike_in_individual_msdial_harmonized.parquet",
    "mzmine": "../data/spike_in_datasets/spike_in_individual/harmonized/spike_in_individual_mzmine_harmonized.parquet",
}

quantification_df_annotated_paths = {
    "metaboscape": f"../data/spike_in_datasets/spike_in_individual/{annotation_method}/spike_in_individual_metaboscape_{annotation_method}.parquet",
    "msdial": f"../data/spike_in_datasets/spike_in_individual/{annotation_method}/spike_in_individual_msdial_{annotation_method}.parquet",
    "mzmine": f"../data/spike_in_datasets/spike_in_individual/{annotation_method}/spike_in_individual_mzmine_{annotation_method}.parquet",
}


tool_colors = {
    "metaboscape": "#D46428",
    "metaboscape-dorresteinlab": "#FF0000",
    "mzmine": "#0074B3",
    "msdial": "#009F75",
}

formatted_tool_names = {
    "metaboscape": "MetaboScape 2025 14.0.3",
    "mzmine": "MZmine 4.9",
    "msdial": "MS-DIAL 5.5",
}

# Pattern definitions for each transform type
transform_histogram_patterns = {
    "rclr": {"fill": True, "linestyle": "-", "alpha": 0.5},
    "clr": {"fill": False, "linestyle": "--", "alpha": 0.8},
    "raw": {"fill": False, "linestyle": ":", "alpha": 0.8},
}

# Marker definitions for each transform type (for pointwise plots)
transform_marker_styles = {
    "rclr": {"marker": "o", "markersize": 8},
    "clr": {"marker": "x", "markersize": 8},
    "raw": {"marker": "s", "markersize": 6},
}

# Formatted transform names for legends
formatted_transform_names = {
    "rclr": "RCLR",
    "clr": "CLR",
    "raw": "Raw",
}

# Transform settings: (tic_norm, transform_type)
transform_settings = {
    "rclr": (True, "rclr"),
    "clr": (True, "clr"),
    "raw": (False, None),
}

import numpy as np
import pandas as pd

ground_truth_directories = [
    "/home/prajit.rajkumar/benchmarking-untargeted-metabolomics-software/data/spike_in_metadata/spike_in_individual_positive_50ng/cleaned_metadata.csv",
    "/home/prajit.rajkumar/benchmarking-untargeted-metabolomics-software/data/spike_in_metadata/spike_in_individual_positive_100ng/cleaned_metadata.csv",
]

ground_truth_data_combined = pd.concat(
    [pd.read_csv(path) for path in ground_truth_directories], ignore_index=True
)
ground_truth_data_combined["INCHIKEY_NONISOMERIC"] = (
    ground_truth_data_combined["INCHIKEY"].dropna().apply(lambda x: x.split("-")[0])
)
ground_truth_inchikeys = (
    ground_truth_data_combined["INCHIKEY_NONISOMERIC"].dropna().unique().tolist()
)
ground_truth_data_combined.index = ground_truth_data_combined[
    "Sample Description"
].str.strip()


def apply_rclr_transform(sample_column):
    # Mask nonzero values
    mask = sample_column > 0
    filtered_values = sample_column[mask]

    # compute log(GM) of non-zero values
    log_gm = np.mean(np.log(filtered_values))

    # Apply rclr transformation to non-zero values
    rclr_values = np.where(mask, np.log(sample_column) - log_gm, 0)
    return rclr_values


def apply_clr_transform(sample_column, pseudocount="min"):
    # Add pseudocount to avoid log(0)
    if pseudocount == "min":
        min_nonzero = sample_column[sample_column > 0].min()
        pseudocount = min_nonzero

    adjusted_values = sample_column + pseudocount

    # Compute log(GM) of all values
    log_gm = np.mean(np.log(adjusted_values))

    # Apply clr transformation
    clr_values = np.log(adjusted_values) - log_gm
    return clr_values


def assess_quantification(
    input_df,
    ground_truth_df,
    tool_type,
    tic_norm=True,
    transform_type=None,
    return_pvalues=False,
    inchikey_column="INCHIKEY_SPECTRAL_ENTROPY",
    score_column="SCORE_SPECTRAL_ENTROPY",
    score_threshold=0.7,
):
    from scipy.stats import linregress

    column_names = [col for col in input_df.columns if "SAMPLE" in col]
    concentrations = [
        int(col.split("_")[3].replace("ng", "").strip()) for col in column_names
    ]
    compound_names = [col.split("_")[2] for col in column_names]
    print(compound_names)
    print(len(compound_names))
    print(len(set(compound_names)))
    intensities_df = input_df[column_names]
    # Normalize columns by sum
    if tic_norm:
        intensities_df = intensities_df.div(intensities_df.sum(axis=0), axis=1)
    if transform_type == "rclr":
        intensities_df = intensities_df.apply(apply_rclr_transform, axis=0)
    elif transform_type == "clr":
        intensities_df = intensities_df.apply(apply_clr_transform, axis=0)

    formatted_samples = {}
    for i, column in enumerate(column_names):
        curr_compound_name = compound_names[i]
        if curr_compound_name == "methyl gallate":
            curr_compound_name = "methylgallate"
        elif curr_compound_name == "hydroxytyrosol acetate":
            curr_compound_name = "hydroxytyrosolacetate"
        if curr_compound_name not in formatted_samples:
            formatted_samples[curr_compound_name] = {}
        formatted_samples[curr_compound_name][concentrations[i]] = column

    print(len(formatted_samples))
    formatted_samples_filtered = {}
    for sample in formatted_samples:
        if len(formatted_samples[sample]) > 1:
            formatted_samples_filtered[sample] = formatted_samples[sample]

    print(len(formatted_samples_filtered))
    output = {}
    pvalues = {}
    for sample in formatted_samples_filtered:
        concs = []
        intensities = []
        prev_adduct = None
        for conc in sorted(formatted_samples_filtered[sample].keys()):

            if "mzmine" in tool_type.lower():
                formatted_column_name = (
                    formatted_samples_filtered[sample][conc].split(" Peak")[0].strip()
                    + ".d"
                )
            else:
                formatted_column_name = (
                    formatted_samples_filtered[sample][conc].strip() + ".d"
                )

            # Get the corresponding inchikey
            inchikey_nonisomeric = ground_truth_df.loc[
                formatted_column_name, "INCHIKEY_NONISOMERIC"
            ]

            # Get rows in input_df matching the inchikey and above score threshold
            matching_rows = input_df[
                (input_df[inchikey_column].str.contains(inchikey_nonisomeric, na=False))
                & (input_df[score_column] >= score_threshold)
            ]

            if not matching_rows.empty:
                # If multiple matches, take the one with highest score
                best_match = matching_rows.loc[matching_rows[score_column].idxmax()]
                intensities.append(
                    intensities_df.loc[
                        best_match.name, formatted_samples_filtered[sample][conc]
                    ]
                )
                concs.append(conc)
            else:
                # If no match found, use minimum value in that column
                print(
                    f"Warning: No matching feature found for sample {sample} at concentration {conc}ng with INCHIKEY {inchikey_nonisomeric}\
                      and tool {tool_type}. Using minimum intensity value for that concentration."
                )
                continue
                intensities.append(
                    intensities_df[formatted_samples_filtered[sample][conc]].min()
                )

            curr_adduct = (
                best_match.get("ADDUCT", None) if not matching_rows.empty else None
            )
            if (
                curr_adduct is not None
                and prev_adduct is not None
                and curr_adduct != prev_adduct
                and curr_adduct is not np.nan
                and pd.notna(curr_adduct)
            ):
                print(
                    f"Warning: Adduct changed for sample {sample} at concentration {conc}ng: {prev_adduct} -> {curr_adduct}"
                )
            if (
                curr_adduct is not None
                and curr_adduct is not np.nan
                and pd.notna(curr_adduct)
            ):
                prev_adduct = curr_adduct

        if len(concs) >= 3:
            result = linregress(concs, intensities)
            r_squared = result.rvalue**2
            pvalue = result.pvalue
        else:
            r_squared, pvalue = np.nan, np.nan
        output[sample] = r_squared
        pvalues[sample] = pvalue

    if return_pvalues:
        return output, pvalues
    return output


def plot_correlation_histogram(correlation_dict, tool_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    correlations = list(correlation_dict.values())
    plt.figure(figsize=(8, 6))
    sns.histplot(
        correlations, bins=10, kde=True, color=tool_colors.get(tool_name, "#333333")
    )
    plt.title(f"R² Histogram for {formatted_tool_names.get(tool_name, tool_name)}")
    plt.xlabel("Coefficient of Determination (r²)")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.show()


def plot_all_tool_correlations_histogram(
    quantification_df_paths, tic_norm=True, transform_type="rclr"
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    plt.figure(figsize=(10, 6))

    for tool_name, file_path in quantification_df_paths.items():
        harmonized_df = pd.read_parquet(file_path)
        annotated_df_path = quantification_df_annotated_paths.get(tool_name, None)
        if not annotated_df_path:
            raise ValueError(
                f"Annotated dataframe path not found for tool: {tool_name}"
            )
        annotated_df = pd.read_parquet(annotated_df_path)
        annotated_df["FEATURE_ID"] = annotated_df["FEATURE_ID"].astype(
            dtype=harmonized_df["FEATURE_ID"].dtype
        )
        input_df = pd.merge(harmonized_df, annotated_df, on="FEATURE_ID", how="left")
        correlation_dict = assess_quantification(
            input_df,
            ground_truth_data_combined,
            tool_name,
            tic_norm=tic_norm,
            transform_type=transform_type,
        )
        correlations = list(correlation_dict.values())

        sns.kdeplot(
            correlations,
            label=formatted_tool_names.get(tool_name, tool_name),
            color=tool_colors.get(tool_name, "#333333"),
            fill=True,
            alpha=0.5,
        )

    plt.title("Coefficient of Determination Distributions Across Tools")
    plt.xlabel("Coefficient of Determination (r²)")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_tool_correlations_violin(
    quantification_df_paths, tic_norm=True, transform_type="rclr", jitter=0.35
):
    """
    Plot r² distributions as violin plots for all tools.

    Args:
        quantification_df_paths: dict mapping tool names to parquet file paths
        tic_norm: bool, whether to apply TIC normalization
        transform_type: "rclr", "clr", or None for raw
        jitter: float, amount of horizontal jitter for the points (default 0.35)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Collect all r² values into a DataFrame for violin plot
    all_data = []

    for tool_name, file_path in quantification_df_paths.items():
        harmonized_df = pd.read_parquet(file_path)
        annotated_df_path = quantification_df_annotated_paths.get(tool_name, None)
        if not annotated_df_path:
            raise ValueError(
                f"Annotated dataframe path not found for tool: {tool_name}"
            )
        annotated_df = pd.read_parquet(annotated_df_path)
        annotated_df["FEATURE_ID"] = annotated_df["FEATURE_ID"].astype(
            dtype=harmonized_df["FEATURE_ID"].dtype
        )
        input_df = pd.merge(harmonized_df, annotated_df, on="FEATURE_ID", how="left")
        correlation_dict = assess_quantification(
            input_df,
            ground_truth_data_combined,
            tool_name,
            tic_norm=tic_norm,
            transform_type=transform_type,
        )

        for compound, r2 in correlation_dict.items():
            all_data.append(
                {
                    "Tool": formatted_tool_names.get(tool_name, tool_name),
                    "tool_key": tool_name,
                    "r²": r2,
                    "Compound": compound,
                }
            )

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Create violin plot
    ax = sns.violinplot(
        data=df,
        x="Tool",
        y="r²",
        palette={formatted_tool_names.get(k, k): v for k, v in tool_colors.items()},
        inner=None,  # No inner box, we'll add stripplot instead
        cut=0,
        alpha=0.3,
    )

    # Add jittered points on top
    sns.stripplot(
        data=df,
        x="Tool",
        y="r²",
        palette={formatted_tool_names.get(k, k): v for k, v in tool_colors.items()},
        jitter=jitter,
        size=6,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    plt.title("Coefficient of Determination Distributions Across Tools")
    plt.xlabel("Software")
    plt.ylabel("Coefficient of Determination (r²)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_tool_transform_combinations_histogram(
    quantification_df_paths,
    combinations: list[tuple[str, str]],
    smooth: bool = True,
):
    """
    Plot r² distributions for specified (tool, transform) combinations.

    Args:
        quantification_df_paths: dict mapping tool names to parquet file paths
        combinations: list of (tool_name, transform_type) tuples
            - tool_name: key in quantification_df_paths (e.g., "mzmine", "msdial", "metaboscape")
            - transform_type: "rclr", "clr", or "raw"
                - "rclr": tic_norm=True, transform="rclr"
                - "clr": tic_norm=True, transform="clr"
                - "raw": tic_norm=False, transform=None
        smooth: bool, default True
            - True: use KDE plot (smooth density estimate)
            - False: use standard binned histogram

    Example:
        combinations = [
            ("mzmine", "raw"),
            ("mzmine", "rclr"),
            ("msdial", "rclr"),
            ("metaboscape", "clr"),
        ]
        plot_tool_transform_combinations_histogram(quantification_df_paths, combinations, smooth=True)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if not combinations:
        print("No combinations specified.")
        return

    plt.figure(figsize=(12, 7))

    # Cache loaded dataframes to avoid re-reading
    df_cache = {}

    for tool_name, transform_type in combinations:
        if tool_name not in quantification_df_paths:
            print(
                f"Warning: Tool '{tool_name}' not found in quantification_df_paths. Skipping."
            )
            continue

        if transform_type not in transform_settings:
            print(
                f"Warning: Transform '{transform_type}' not recognized. Use 'rclr', 'clr', or 'raw'. Skipping."
            )
            continue

        # Load dataframe (use cache if already loaded)
        if tool_name not in df_cache:
            harmonized_file_path = quantification_df_paths[tool_name]
            harmonized_df = pd.read_parquet(harmonized_file_path)
            annotated_df_path = quantification_df_annotated_paths.get(tool_name, None)
            if not annotated_df_path:
                raise ValueError(
                    f"Annotated dataframe path not found for tool: {tool_name}"
                )
            annotated_df = pd.read_parquet(annotated_df_path)
            annotated_df["FEATURE_ID"] = annotated_df["FEATURE_ID"].astype(
                dtype=harmonized_df["FEATURE_ID"].dtype
            )
            input_df = pd.merge(
                harmonized_df, annotated_df, on="FEATURE_ID", how="left"
            )
            df_cache[tool_name] = input_df
        input_df = df_cache[tool_name]

        # Get transform settings
        tic_norm, transform_arg = transform_settings[transform_type]

        # Compute r² values
        r2_dict = assess_quantification(
            input_df,
            ground_truth_data_combined,
            tool_name,
            tic_norm=tic_norm,
            transform_type=transform_arg,
        )
        r2_values = list(r2_dict.values())

        # Get visual styling
        pattern = transform_histogram_patterns[transform_type]
        color = tool_colors.get(tool_name, "#333333")
        formatted_name = formatted_tool_names.get(tool_name, tool_name)
        formatted_transform = formatted_transform_names.get(
            transform_type, transform_type
        )
        label = f"{formatted_name} ({formatted_transform})"

        # Plot KDE or histogram based on smooth parameter
        if smooth:
            sns.kdeplot(
                r2_values,
                label=label,
                color=color,
                fill=pattern["fill"],
                linestyle=pattern["linestyle"],
                alpha=pattern["alpha"],
                linewidth=2,
            )
        else:
            plt.hist(
                r2_values,
                bins=20,
                label=label,
                color=color,
                alpha=pattern["alpha"],
                edgecolor="black",
                linewidth=0.5,
                histtype="stepfilled" if pattern["fill"] else "step",
            )

    plt.title("Coefficient of Determination Distributions")
    plt.xlabel("Coefficient of Determination (r²)")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tool_transform_combinations_pointwise(
    quantification_df_paths,
    combinations: list[tuple[str, str]],
    line: bool = False,
    discard_ns: bool = False,
    jitter: float = 0.15,
):
    """
    Plot r² for each compound, with compound names on x-axis and r² on y-axis.

    Args:
        quantification_df_paths: dict mapping tool names to parquet file paths
        combinations: list of (tool_name, transform_type) tuples
            - tool_name: key in quantification_df_paths (e.g., "mzmine", "msdial", "metaboscape")
            - transform_type: "rclr", "clr", or "raw"
                - "rclr": tic_norm=True, transform="rclr"
                - "clr": tic_norm=True, transform="clr"
                - "raw": tic_norm=False, transform=None
        line: bool, default False
            - True: connect points with a line
            - False: scatter plot only
        discard_ns: bool, default False
            - True: do not plot non-significant values (p >= 0.05)
            - False: plot all values
        jitter: float, default 0.15
            - Amount of horizontal jitter to add to separate overlapping points
            - Set to 0 for no jitter

    Returns:
        dict: {(tool_name, transform_type): {compound: {"r2": r_squared, "p": pvalue}, ...}, ...}

    Example:
        combinations = [
            ("mzmine", "raw"),
            ("mzmine", "rclr"),
            ("msdial", "rclr"),
            ("metaboscape", "clr"),
        ]
        results = plot_tool_transform_combinations_pointwise(quantification_df_paths, combinations, line=True, discard_ns=True)
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if not combinations:
        print("No combinations specified.")
        return {}

    # Cache loaded dataframes to avoid re-reading
    df_cache = {}

    # Collect all compound names across all combinations
    all_compounds = set()
    results = []

    # Output dictionary to return
    output_dict = {}

    for tool_name, transform_type in combinations:
        if tool_name not in quantification_df_paths:
            print(
                f"Warning: Tool '{tool_name}' not found in quantification_df_paths. Skipping."
            )
            continue

        if transform_type not in transform_settings:
            print(
                f"Warning: Transform '{transform_type}' not recognized. Use 'rclr', 'clr', or 'raw'. Skipping."
            )
            continue

        # Load dataframe (use cache if already loaded)
        if tool_name not in df_cache:
            harmonized_file_path = quantification_df_paths[tool_name]
            harmonized_df = pd.read_parquet(harmonized_file_path)
            annotated_df_path = quantification_df_annotated_paths.get(tool_name, None)
            if not annotated_df_path:
                raise ValueError(
                    f"Annotated dataframe path not found for tool: {tool_name}"
                )
            annotated_df = pd.read_parquet(annotated_df_path)
            annotated_df["FEATURE_ID"] = annotated_df["FEATURE_ID"].astype(
                dtype=harmonized_df["FEATURE_ID"].dtype
            )
            input_df = pd.merge(
                harmonized_df, annotated_df, on="FEATURE_ID", how="left"
            )
            df_cache[tool_name] = input_df
        input_df = df_cache[tool_name]

        # Get transform settings
        tic_norm, transform_arg = transform_settings[transform_type]

        # Compute r² values (returns dict: compound_name -> r²)
        r2_dict, pvalue_dict = assess_quantification(
            input_df,
            ground_truth_data_combined,
            tool_name,
            tic_norm=tic_norm,
            transform_type=transform_arg,
            return_pvalues=True,
        )

        # Store in output dictionary
        output_dict[(tool_name, transform_type)] = {
            compound: {"r2": r2_dict[compound], "p": pvalue_dict[compound]}
            for compound in r2_dict
        }

        all_compounds.update(r2_dict.keys())
        results.append((tool_name, transform_type, r2_dict, pvalue_dict))

    if not results:
        print("No valid combinations to plot.")
        return output_dict

    # Sort compounds alphabetically
    sorted_compounds = sorted(all_compounds)
    compound_indices = {compound: i for i, compound in enumerate(sorted_compounds)}

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))

    # Calculate jitter offsets for each combination
    n_combinations = len(results)
    jitter_offsets = (
        np.linspace(
            -jitter * (n_combinations - 1) / 2,
            jitter * (n_combinations - 1) / 2,
            n_combinations,
        )
        if n_combinations > 1
        else [0]
    )

    # Plot each combination
    for idx, (tool_name, transform_type, r2_dict, pvalue_dict) in enumerate(results):
        # Get visual styling
        marker_style = transform_marker_styles[transform_type]
        color = tool_colors.get(tool_name, "#333333")
        formatted_name = formatted_tool_names.get(tool_name, tool_name)
        formatted_transform = formatted_transform_names.get(
            transform_type, transform_type
        )
        label = f"{formatted_name} ({formatted_transform})"

        # Get x positions and y values
        x_positions = []
        y_values = []
        sig_x_positions = []  # For significant points overlay
        sig_y_values = []
        for compound, r2 in r2_dict.items():
            # Skip non-significant values if discard_ns is True
            if discard_ns and pvalue_dict.get(compound, 1.0) >= 0.05:
                continue
            x_pos = compound_indices[compound] + jitter_offsets[idx]
            x_positions.append(x_pos)
            y_values.append(r2)
            # Track significant points for star overlay
            if not discard_ns and pvalue_dict.get(compound, 1.0) < 0.05:
                sig_x_positions.append(x_pos)
                sig_y_values.append(r2)

        # Sort by x position for proper line connection
        if line:
            sorted_pairs = sorted(zip(x_positions, y_values), key=lambda p: p[0])
            x_positions = [p[0] for p in sorted_pairs]
            y_values = [p[1] for p in sorted_pairs]
            ax.plot(
                x_positions,
                y_values,
                color=color,
                linestyle="-",
                linewidth=1,
                alpha=0.5,
            )

        ax.scatter(
            x_positions,
            y_values,
            label=label,
            color=color,
            marker=marker_style["marker"],
            s=marker_style["markersize"] ** 2,
            alpha=0.7,
        )

        # Add black asterisk overlay for significant points when discard_ns=False
        if not discard_ns and sig_x_positions:
            ax.scatter(
                sig_x_positions,
                sig_y_values,
                color="black",
                marker="$*$",
                s=50,
                alpha=0.9,
                zorder=10,
            )

    # Configure axes
    ax.set_xticks(range(len(sorted_compounds)))
    ax.set_xticklabels(sorted_compounds, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Compound Name")
    ax.set_ylabel("Coefficient of Determination (r²)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Coefficient of Determination by Compound")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    return output_dict
