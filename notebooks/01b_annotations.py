# -*- coding: utf-8 -*-

"""Running the DreaMS matching."""

import os
import faiss
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

import benchmarking.constants as cons

import warnings
from numba.core.errors import NumbaTypeSafetyWarning

# Filter out the specific numba type safety warning
warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

tqdm.pandas()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

faiss.omp_set_num_threads(os.cpu_count())
print(f"Using {faiss.omp_get_max_threads()} threads")


def main():
    # Read in and prepare the spectral library for matching
    library_data_filtered = pd.read_parquet(
        "../data/library_spectra/all_sorted_library_spectra.parquet"
    )

    logger.info(f"Loaded library data with {len(library_data_filtered)} spectra")

    # Prepare library for matching
    reference_inchis = library_data_filtered["inchikey_2d"].to_numpy(dtype=object)
    reference_ids = library_data_filtered.index.to_numpy(dtype=object)
    reference_mzs = library_data_filtered["normalized_mzs"].to_numpy(dtype=object)
    reference_intensities = library_data_filtered["normalized_intensities"].to_numpy(
        dtype=object
    )
    reference_precursors = library_data_filtered["precursor_mz"].to_numpy(dtype=object)

    logger.info("Prepared library spectra and metadata for matching")

    ###### Specify input feature tables
    # base_dir_public = "../data/public_dataset"
    # public_datasets = [
    #     "MSV000090327",
    #     "MSV000091642",
    #     "MSV000095813",
    #     "MSV000097967",
    #     "MSV000097015",
    #     "MSV000096291",
    #     "MSV000096189",
    #     "ST002402",
    #     "MSV000084402",
    #     "MTBLS12332",
    # ]

    # base_dir_internal = "../data/groundtruth_dataset"
    # internal_datasets = [
    #     "MSV000098263",
    #     "plant_spikein",
    #     "NIST_SRM",
    # ]

    # final_dir_list = [
    #     f"{base_dir_public}/{dataset}/harmonized" for dataset in public_datasets
    # ] + [f"{base_dir_internal}/{dataset}/harmonized" for dataset in internal_datasets
    # ]

    final_dir_list = ["../data/groundtruth_dataset/MSV000098263"]

    input_feature_tables = []
    input_embeddings = []

    for dataset_dir in final_dir_list:
        input_dir = f"{dataset_dir}/harmonized"
        if not os.path.exists(input_dir):
            print(
                f"Directory {input_dir} does not exist. Skipping dataset {dataset_dir}."
            )
            continue

        for file in os.listdir(input_dir):
            input_feature_tables.append(f"{input_dir}/{file}")
            input_embeddings.append(
                f"{input_dir}/{file}".replace(
                    "_harmonized.parquet", "_embeddings.npz"
                ).replace("harmonized/", "embeddings/")
            )

    output_feature_tables_dreams = []

    for file_path in input_feature_tables:
        output_feature_tables_dreams.append(
            file_path.replace("harmonized", "annotated_dreams_similarity")
        )
        os.makedirs(os.path.dirname(output_feature_tables_dreams[-1]), exist_ok=True)

    ### Dreams similarity
    library_embeddings = np.load(
        "../data/library_spectra/all_sorted_library_spectra_dreams_embeddings.npz",
    )["dreams_embeddings"]
    library_embeddings = np.ascontiguousarray(library_embeddings, dtype=np.float32)
    faiss.normalize_L2(library_embeddings)
    library_index = faiss.IndexFlatIP(1024)
    library_index.add(library_embeddings)

    logger.info(f"Loaded library embeddings with shape {library_embeddings.shape}")

    counter = 0

    for input_feature_table, output_feature_table_dreams, input_embedding in tqdm(
        zip(input_feature_tables, output_feature_tables_dreams, input_embeddings),
        total=len(input_feature_tables),
    ):
        if os.path.exists(output_feature_table_dreams):
            counter += 1
            continue

        logger.info(f"Processing {input_feature_table}")
        harmonized_feature_table = pd.read_parquet(input_feature_table)
        harmonized_feature_table_spectra_available = harmonized_feature_table[
            harmonized_feature_table["MS/MS_ASSIGNED"] == True
        ]

        feature_ids = harmonized_feature_table_spectra_available["FEATURE_ID"].to_list()

        query_dreams_embeddings = np.load(input_embedding)["dreams_embeddings"]
        faiss.normalize_L2(query_dreams_embeddings)
        similarities, indices = library_index.search(query_dreams_embeddings, 1)

        invalid_mask = (indices.flatten() < 0) | (
            indices.flatten() >= len(reference_ids)
        )
        if invalid_mask.any():
            logger.warning(
                f"Warning: {invalid_mask.sum()} queries had no valid match in {input_feature_table}"
            )

        idx_updated = np.where(invalid_mask, 0, indices.flatten())

        output = pd.DataFrame(
            {
                cons.FEATURE_ID_COLUMN: feature_ids,
                cons.LIBRARY_ID_COLUMN: reference_ids[idx_updated],
                cons.METHOD_COLUMN: ["dreams_similarity"] * len(feature_ids),
                cons.INCHIKEY_COLUMN: reference_inchis[idx_updated],
                cons.SCORE_COLUMN: similarities.flatten().tolist(),
                cons.LIBRARY_PRECURSOR_MZ_COLUMN: reference_precursors[idx_updated],
                cons.LIBRARY_MZS_COLUMN: reference_mzs[idx_updated],
                cons.LIBRARY_INTENSITIES_COLUMN: reference_intensities[idx_updated],
            }
        )

        os.makedirs(os.path.dirname(output_feature_table_dreams), exist_ok=True)
        output.to_parquet(output_feature_table_dreams, index=False, engine="pyarrow")

    print(
        f"Dreams similarity annotation skipped for {counter}/{len(input_feature_tables)} files as output already exists."
    )


if __name__ == "__main__":
    main()
