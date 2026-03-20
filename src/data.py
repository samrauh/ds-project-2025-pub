import os
import sqlite3
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from .models import (
    GraphEdgeFeatures,
    GraphNodeFeatures,
    IPCPopularity,
    YearlyGraphData,
    YearlyTensorData,
    YearlyTensorDataset,
)


def get_data_loaders(
    data_dict: Dict[int, YearlyTensorDataset],
    train_years: range,
    val_years: range,
    test_years: range,
    batch_size: int = 32,
):
    """
    Creates DataLoaders for train, validation, and test sets based on year ranges.
    """
    train_dataset = ConcatDataset(
        [data_dict[year] for year in train_years if year in data_dict]
    )
    val_dataset = ConcatDataset(
        [data_dict[year] for year in val_years if year in data_dict]
    )
    test_dataset = ConcatDataset(
        [data_dict[year] for year in test_years if year in data_dict]
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def calc_popularity_scores(
    db_path: str, score_type="rel"
) -> Dict[int, List[IPCPopularity]]:
    """
    Calculates popularity scores from the database and returns a dictionary mapping years to lists of IPCPopularity objects.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    # for each ipc code count occurences per year
    query = """
    SELECT ipc_code, YEAR AS pub_year, COUNT(*) AS count
    FROM patent_ipc pi
    GROUP BY ipc_code, YEAR
    """
    df_ipc_counts = pd.read_sql_query(query, conn)
    conn.close()

    threshold_max = 20
    threshold_min = 20
    lag = 1

    # calculate the log-difference of count per year
    df_lagged = df_ipc_counts.copy()
    df_ipc_counts["target_year"] = df_ipc_counts["pub_year"] + lag

    df_merged = pd.merge(
        df_ipc_counts,
        df_lagged[["ipc_code", "pub_year", "count"]],
        left_on=["ipc_code", "target_year"],
        right_on=["ipc_code", "pub_year"],
        suffixes=("", "_next"),
    )

    # calculate relative and log differences
    df_merged["rel_count_diff"] = (
        df_merged["count_next"] - df_merged["count"]
    ) / df_merged["count"]
    k = 1  # smoothing constant
    df_merged["log_count_diff"] = np.log(df_merged["count_next"] + k) - np.log(
        df_merged["count"] + k
    )

    # Keep the linear max_count to maintain high emphasis on large values
    df_merged["max_count"] = df_merged[["count", "count_next"]].max(axis=1)

    # Calculate raw score
    power = 0.5
    df_merged["weighted_score_raw"] = df_merged["rel_count_diff"] * (
        df_merged["max_count"] ** power
    )

    # Final Target Variable
    if score_type == "rel":
        df_merged["score"] = df_merged["rel_count_diff"]
    elif score_type == "comple":
        theta = 10.0
        df_merged["score"] = np.arcsinh(
            df_merged["weighted_score_raw"] / theta
        )  # Renamed to score
    elif score_type == "log":
        df_merged["score"] = df_merged["log_count_diff"]
    else:
        df_merged["score"] = df_merged["rel_count_diff"]

    # get score from previous year as feature (score_d_1)
    df_prev_year = df_merged[["ipc_code", "target_year", "score", "count"]].copy()

    df_merged = pd.merge(
        df_merged,
        df_prev_year,
        how="left",
        left_on=["ipc_code", "pub_year"],
        right_on=["ipc_code", "target_year"],
        suffixes=("", "_prev"),
    )
    df_merged.rename(columns={"score_prev": "score_d_1"}, inplace=True)

    # get score from two years ago (score_d_2)
    df_merged = pd.merge(
        df_merged,
        df_prev_year,
        how="left",
        left_on=["ipc_code", "pub_year"],
        right_on=["ipc_code", pd.Series([y - 1 for y in df_merged["pub_year"]])],
        suffixes=("", "_d2"),
    )

    df_merged.rename(columns={"score_d2": "score_d_2"}, inplace=True)

    # calculate top quartile for each year
    def top_quartile(x):
        return x.quantile(0.75)

    yearly_quartiles = (
        df_merged.groupby("pub_year")["score"].apply(top_quartile).reset_index()
    )
    yearly_quartiles = yearly_quartiles.rename(columns={"score": "yearly_quartile"})

    df_merged = pd.merge(df_merged, yearly_quartiles, on="pub_year", how="left")
    df_merged["is_top_quartile"] = df_merged["score"] >= df_merged["yearly_quartile"]

    # Filter
    df_merged = df_merged[df_merged["pub_year"] >= 2006]
    df_merged = df_merged[
        (df_merged["max_count"] > threshold_max) & (df_merged["count"] > threshold_min)
    ]

    # Convert to dictionary of IPCPopularity objects
    popularity_by_year = {}
    for year, group in df_merged.groupby("pub_year"):
        popularity_by_year[year] = [
            IPCPopularity(
                ipc_code=row.ipc_code,
                pub_year=row.pub_year,
                count=row.count,
                count_prev=int(row.count_prev) if not pd.isna(row.count_prev) else None,
                score=row.score,
                score_d_1=row.score_d_1 if not pd.isna(row.score_d_1) else None,
                score_d_2=row.score_d_2 if not pd.isna(row.score_d_2) else None,
                is_top_quartile=row.is_top_quartile,
            )
            for row in group.itertuples()
        ]

    return popularity_by_year


def create_data_dict(
    data_folder: str = "../data/ipc_mean_year_abstract",
    db_path: str = "../data/patent.db",
    score_type="rel",
) -> Dict[int, YearlyTensorDataset]:
    """
    Creates a data dictionary mapping years to dataset objects.
    """

    print("Calculating popularity scores...")
    # 1. Get Popularity Data
    popularity_dict = calc_popularity_scores(db_path=db_path, score_type=score_type)

    # 2. Create Global IPC Map using the static method from IPCPopularity
    print("Creating global IPC mapping...")
    ipc_code_to_int = IPCPopularity.create_mapping_from_sequences(popularity_dict)
    print(f"Total unique IPC codes: {len(ipc_code_to_int)}")

    data_year_datasets = {}

    year_cutoff = 2006
    last_year = 2022

    for year in tqdm(range(year_cutoff, last_year + 1), desc="Processing years"):
        if year not in popularity_dict:
            continue

        file_path = os.path.join(data_folder, f"ipc_mean_{year}.pkl")

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping year {year}.")
            continue

        # Load embeddings: Dict[ipc_code, embedding]
        embeddings_dict = joblib.load(file_path)

        # Get popularity objects for this year
        year_popularity = popularity_dict[year]

        # Prepare list of IPCSamples
        samples = []

        for pop_item in year_popularity:
            code = pop_item.ipc_code

            # Check if we have an embedding for this code
            if code not in embeddings_dict:
                continue

            if code not in ipc_code_to_int:
                # Should not happen if map is created from popularity_dict,
                # but purely defensive
                continue

            # Assign integer mapping to popularity object
            pop_item.ipc_int = ipc_code_to_int[code]

            embedding = embeddings_dict[code]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Use the method on IPCPopularity to create the sample
            sample = pop_item.to_sample(embedding=embedding)
            samples.append(sample)

        if not samples:
            continue

        # Create YearlyTensorData
        year_data = YearlyTensorData(year=year, items=samples)

        # Convert to Dataset
        data_year_datasets[year] = year_data.to_tensor_dataset()

    return data_year_datasets


def normalize_data_dict(
    data_dict: Dict[int, YearlyTensorDataset],
    train_years,
) -> Tuple[Dict[int, YearlyTensorDataset], Dict[str, float]]:
    """
    Fits a z-score normalizer on training years only and applies it to all years
    for the MLP data dict produced by create_data_dict().

    Normalizes all scalar features in-place:
      - scalar[:, 0]  -> count
      - scalar[:, 1]  -> count_prev
      - scalar[:, 2]  -> score_d_1
      - scalar[:, 3]  -> score_d_2
      - scalar[:, 4:12] -> one-hot IPC category (a-h)

    Embeddings and targets are left untouched.

    Args:
        data_dict:    Dict mapping year -> YearlyTensorDataset (from create_data_dict).
        train_years:  Iterable of years used for training (used to fit the scaler).

    Returns:
        (data_dict, norm_stats) where norm_stats contains the mean/std values used,
        keyed as 'scalar_mean', 'scalar_std'.
    """
    available_train_years = [y for y in train_years if y in data_dict]
    if not available_train_years:
        raise ValueError("None of the supplied train_years are present in data_dict.")

    # Fit: collect training-year scalar tensors
    scalar_train = torch.cat(
        [data_dict[y].scalar for y in available_train_years], dim=0
    )  # shape [N_train, num_scalar_features]

    scalar_mean = scalar_train.mean(dim=0)
    scalar_std = scalar_train.std(dim=0, unbiased=False).clamp(min=1e-6)

    # Transform: apply to every year
    for dataset in data_dict.values():
        dataset.scalar = (dataset.scalar - scalar_mean) / scalar_std

    norm_stats = {
        "scalar_mean": scalar_mean.tolist(),
        "scalar_std": scalar_std.tolist(),
    }

    print(
        f"Normalization fitted on {len(available_train_years)} training year(s).\n"
        f"  scalar mean={norm_stats['scalar_mean']}, std={norm_stats['scalar_std']}"
    )

    return data_dict, norm_stats


def create_edgelist(db_path: str = "../data/patent.db") -> pd.DataFrame:
    """
    Creates an edgelist of IPC code co-occurrences with Salton similarity.

    Based on the extract_network.ipynb notebook, this function:
    1. Finds all co-occurrences of IPC codes in patents (per year)
    2. Calculates individual IPC code counts per year
    3. Computes Salton's cosine similarity: weight / sqrt(count1 * count2)

    Returns:
        DataFrame with columns: ipc1, ipc2, pub_year, weight, count1, count2, salton_similarity
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    # Create edge list for IPC codes by counting co-occurrences for each year
    query = """
    SELECT 
        ipc1.ipc_code AS ipc1, 
        ipc2.ipc_code AS ipc2, 
        ipc1.YEAR AS pub_year,
        COUNT(*) AS weight
    FROM patent_ipc AS ipc1
    JOIN patent_ipc AS ipc2 ON ipc1.patent_id = ipc2.patent_id
    WHERE ipc1.ipc_code < ipc2.ipc_code
    GROUP BY ipc1.ipc_code, ipc2.ipc_code, ipc1.YEAR
    """
    df_ipc_cooccurrence = pd.read_sql_query(query, conn)
    df_ipc_cooccurrence["pub_year"] = df_ipc_cooccurrence["pub_year"].astype(int)

    # Get total occurrence (degree) of each IPC code per year
    query_counts = """
    SELECT 
        ipc_code, 
        YEAR AS pub_year, 
        COUNT(*) AS count
    FROM patent_ipc
    GROUP BY ipc_code, YEAR
    """
    df_ipc_counts = pd.read_sql_query(query_counts, conn)
    df_ipc_counts["pub_year"] = df_ipc_counts["pub_year"].astype(int)
    conn.close()

    # Merge individual counts back to the edge list
    df_ipc_cooccurrence = (
        df_ipc_cooccurrence.merge(
            df_ipc_counts,
            left_on=["ipc1", "pub_year"],
            right_on=["ipc_code", "pub_year"],
        )
        .rename(columns={"count": "count1"})
        .drop(columns="ipc_code")
    )

    df_ipc_cooccurrence = (
        df_ipc_cooccurrence.merge(
            df_ipc_counts,
            left_on=["ipc2", "pub_year"],
            right_on=["ipc_code", "pub_year"],
        )
        .rename(columns={"count": "count2"})
        .drop(columns="ipc_code")
    )

    # Calculate Salton's Cosine Similarity: weight / sqrt(count1 * count2)
    df_ipc_cooccurrence["salton_similarity"] = df_ipc_cooccurrence["weight"] / np.sqrt(
        df_ipc_cooccurrence["count1"] * df_ipc_cooccurrence["count2"]
    )

    return df_ipc_cooccurrence


def create_graph_data_dict(
    data_folder: str = "../data/ipc_mean_year_abstract",
    db_path: str = "../data/patent.db",
    score_type="rel",
) -> Dict[int, Data]:
    """
    Creates a data dictionary mapping years to PyTorch Geometric Data objects for GNN training.

    This function:
    1. Loads popularity scores (targets) from the database
    2. Creates edgelist of IPC co-occurrences with Salton similarity
    3. For each year, combines node features (embeddings + scalar features) with edge features
    4. Converts to PyG Data objects with proper index mapping

    Returns:
        Dictionary mapping year -> PyTorch Geometric Data object
    """
    print("Calculating popularity scores...")
    # 1. Get Popularity Data
    popularity_dict = calc_popularity_scores(db_path=db_path, score_type=score_type)

    # 2. Create Global IPC Map
    print("Creating global IPC mapping...")
    ipc_code_to_int = IPCPopularity.create_mapping_from_sequences(popularity_dict)
    print(f"Total unique IPC codes: {len(ipc_code_to_int)}")

    # 3. Create Edgelist
    print("Creating edgelist with Salton similarity...")
    df_edgelist = create_edgelist(db_path=db_path)

    # Map IPC codes to integers in the edgelist
    df_edgelist["ipc1_int"] = df_edgelist["ipc1"].map(ipc_code_to_int)
    df_edgelist["ipc2_int"] = df_edgelist["ipc2"].map(ipc_code_to_int)

    # Drop rows where mapping failed
    df_edgelist = df_edgelist.dropna(subset=["ipc1_int", "ipc2_int"])
    df_edgelist["ipc1_int"] = df_edgelist["ipc1_int"].astype(int)
    df_edgelist["ipc2_int"] = df_edgelist["ipc2_int"].astype(int)

    data_graph_dict = {}

    year_cutoff = 2006
    last_year = 2022

    for year in tqdm(
        range(year_cutoff, last_year + 1), desc="Processing years for graph"
    ):
        if year not in popularity_dict:
            continue

        file_path = os.path.join(data_folder, f"ipc_mean_{year}.pkl")

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping year {year}.")
            continue

        # Load embeddings: Dict[ipc_code, embedding]
        embeddings_dict = joblib.load(file_path)

        # Get popularity objects for this year
        year_popularity = popularity_dict[year]

        # Create node features
        node_features = []

        for pop_item in year_popularity:
            code = pop_item.ipc_code

            # Check if we have an embedding for this code
            if code not in embeddings_dict:
                continue

            if code not in ipc_code_to_int:
                continue

            # Assign integer mapping
            pop_item.ipc_int = ipc_code_to_int[code]

            embedding = embeddings_dict[code]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Scalar features: [count, count_prev, score_d_1, score_d_2]
            # Use 0.0 for missing values
            scalar_features = [
                float(pop_item.count),
                float(pop_item.count_prev) if pop_item.count_prev is not None else 0.0,
                float(pop_item.score_d_1) if pop_item.score_d_1 is not None else 0.0,
                float(pop_item.score_d_2) if pop_item.score_d_2 is not None else 0.0,
            ]

            node_features.append(
                GraphNodeFeatures(
                    ipc_code=code,
                    ipc_int=pop_item.ipc_int,
                    embedding=embedding,
                    scalar_features=scalar_features,
                    target=pop_item.score,  # Target is NEXT year's score
                )
            )

        if not node_features:
            continue

        # Get valid IPC codes for this year
        valid_ipc_ints = {node.ipc_int for node in node_features}

        # Filter edgelist for this year and valid nodes
        df_edgelist_year = df_edgelist[
            (df_edgelist["pub_year"] == year)
            & df_edgelist["ipc1_int"].isin(valid_ipc_ints)
            & df_edgelist["ipc2_int"].isin(valid_ipc_ints)
        ].copy()

        # Get previous year edgelist for temporal features
        df_edgelist_prev = df_edgelist[
            (df_edgelist["pub_year"] == year - 1)
            & df_edgelist["ipc1_int"].isin(valid_ipc_ints)
            & df_edgelist["ipc2_int"].isin(valid_ipc_ints)
        ].copy()

        # Merge current and previous year edge data
        df_edgelist_merged = df_edgelist_year.merge(
            df_edgelist_prev[["ipc1", "ipc2", "salton_similarity", "weight"]],
            on=["ipc1", "ipc2"],
            how="left",
            suffixes=("", "_prev"),
        )

        # Create edge features with temporal information
        edge_features = []
        for _, row in df_edgelist_merged.iterrows():
            # Get previous year values (use 0.0 if missing)
            salton_prev = (
                float(row["salton_similarity_prev"])
                if pd.notna(row.get("salton_similarity_prev"))
                else 0.0
            )
            weight_prev = (
                float(row["weight_prev"]) if pd.notna(row.get("weight_prev")) else 0.0
            )

            # Current year values
            salton_current = float(row["salton_similarity"])
            weight_current = float(row["weight"])

            # Calculate changes
            similarity_change = salton_current - salton_prev if salton_prev > 0 else 0.0
            weight_change = weight_current - weight_prev if weight_prev > 0 else 0.0

            edge_features.append(
                GraphEdgeFeatures(
                    ipc1=row["ipc1"],
                    ipc2=row["ipc2"],
                    ipc1_int=int(row["ipc1_int"]),
                    ipc2_int=int(row["ipc2_int"]),
                    pub_year=int(row["pub_year"]),
                    salton_similarity=salton_current,
                    weight=weight_current,
                    salton_similarity_prev=salton_prev if salton_prev > 0 else None,
                    weight_prev=weight_prev if weight_prev > 0 else None,
                    similarity_change=similarity_change if salton_prev > 0 else None,
                    weight_change=weight_change if weight_prev > 0 else None,
                )
            )

        # Create YearlyGraphData and convert to PyG Data
        graph_data = YearlyGraphData(
            year=year, node_features=node_features, edge_features=edge_features
        )

        data_graph_dict[year] = graph_data.to_pyg_data()

    return data_graph_dict


def normalize_graph_data(
    data_dict: Dict[int, Data],
    train_years,
) -> Tuple[Dict[int, Data], Dict[str, float]]:
    """
    Fits a z-score normalizer on training years only and applies it to all years.

    Normalizes the unstandardized raw-count features in-place:
      - Node scalar[:, 0]  -> count
      - Node scalar[:, 1]  -> count_prev
      - Edge  edge_attr[:, 1] -> weight
      - Edge  edge_attr[:, 3] -> weight_prev
      - Edge  edge_attr[:, 5] -> weight_change

    All other features (arcsinh-compressed scores, Salton similarity, one-hot
    IPC category) are left untouched.

    Args:
        data_dict:    Dict mapping year -> PyG Data object (from create_graph_data_dict).
        train_years:  Iterable of years used for training (used to fit the scaler).

    Returns:
        (data_dict, norm_stats) where norm_stats contains the mean/std values used,
        keyed as 'count_mean', 'count_std', 'weight_mean', 'weight_std'.
    """
    available_train_years = [y for y in train_years if y in data_dict]
    if not available_train_years:
        raise ValueError("None of the supplied train_years are present in data_dict.")

    # --- Fit: collect training-year tensors ---
    count_train = torch.cat(
        [data_dict[y].scalar[:, 0:2] for y in available_train_years], dim=0
    )  # shape [N_train_nodes, 2]

    edge_weight_train = torch.cat(
        [data_dict[y].edge_attr[:, [1, 3, 5]] for y in available_train_years], dim=0
    )  # shape [N_train_edges, 3]

    count_mean = count_train.mean(dim=0)  # [2]
    count_std = count_train.std(dim=0, unbiased=False).clamp(min=1e-6)  # [2]

    weight_mean = edge_weight_train.mean(dim=0)  # [3]
    weight_std = edge_weight_train.std(dim=0, unbiased=False).clamp(min=1e-6)  # [3]

    # --- Transform: apply to every year ---
    for data in data_dict.values():
        data.scalar[:, 0:2] = (data.scalar[:, 0:2] - count_mean) / count_std
        data.edge_attr[:, [1, 3, 5]] = (
            data.edge_attr[:, [1, 3, 5]] - weight_mean
        ) / weight_std

    norm_stats = {
        "count_mean": count_mean.tolist(),
        "count_std": count_std.tolist(),
        "weight_mean": weight_mean.tolist(),
        "weight_std": weight_std.tolist(),
    }

    print(
        f"Normalization fitted on {len(available_train_years)} training year(s).\n"
        f"  count   mean={norm_stats['count_mean']}, std={norm_stats['count_std']}\n"
        f"  weight  mean={norm_stats['weight_mean']}, std={norm_stats['weight_std']}"
    )

    return data_dict, norm_stats


def get_graph_data_loaders(
    data_dict: Dict[int, Data],
    train_years: range,
    val_years: range,
    test_years: range,
    batch_size: int = 1,
):
    """
    Creates PyTorch Geometric DataLoaders for train, validation, and test sets based on year ranges.

    Args:
        data_dict: Dictionary mapping year to PyG Data objects
        train_years: Range of years for training
        val_years: Range of years for validation
        test_years: Range of years for previous
        batch_size: Batch size for dataloaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_data = [data_dict[year] for year in train_years if year in data_dict]
    val_data = [data_dict[year] for year in val_years if year in data_dict]
    test_data = [data_dict[year] for year in test_years if year in data_dict]

    print(f"Train graphs: {len(train_data)}")
    print(f"Validation graphs: {len(val_data)}")
    print(f"Test graphs: {len(test_data)}")

    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_temp_graph_data_dict(
    data_folder: str = "../data/ipc_mean_year_abstract",
    db_path: str = "../data/patent.db",
    score_type="rel",
) -> Dict[int, Data]:
    """
    Creates a temporal graph data dictionary where every year's graph shares the same
    node structure (all IPC codes across all years).

    Unlike create_graph_data_dict, every Data object has:
    - Exactly num_nodes = len(global IPC mapping) nodes
    - Zero-padded features (embedding, scalar, target) for IPC codes not active in a year
    - A boolean ``node_available`` mask of shape [num_nodes] indicating which nodes are
      actually present in that year

    Edge structure mirrors create_graph_data_dict (only edges between active nodes for
    each year). Node indices equal the global ipc_code_to_int integers directly, so
    the same IPC code always occupies the same row across all year graphs.

    Returns:
        Dictionary mapping year -> PyTorch Geometric Data object with attributes:
          - x             : node embeddings,    shape [num_nodes, embedding_dim]
          - edge_index    : COO edge indices,   shape [2, num_edges]
          - edge_attr     : edge features,      shape [num_edges, 6]
          - y             : node targets (next year score), shape [num_nodes]
          - scalar        : scalar + one-hot features,      shape [num_nodes, 13]
          - node_available: bool mask,          shape [num_nodes]
          - year          : year value per node, shape [num_nodes]
    """
    print("Calculating popularity scores...")
    popularity_dict = calc_popularity_scores(db_path=db_path, score_type=score_type)

    print("Creating global IPC mapping...")
    ipc_code_to_int = IPCPopularity.create_mapping_from_sequences(popularity_dict)
    num_nodes = len(ipc_code_to_int)
    int_to_ipc_code = {v: k for k, v in ipc_code_to_int.items()}
    print(f"Total unique IPC codes: {num_nodes}")

    print("Creating edgelist with Salton similarity...")
    df_edgelist = create_edgelist(db_path=db_path)

    df_edgelist["ipc1_int"] = df_edgelist["ipc1"].map(ipc_code_to_int)
    df_edgelist["ipc2_int"] = df_edgelist["ipc2"].map(ipc_code_to_int)
    df_edgelist = df_edgelist.dropna(subset=["ipc1_int", "ipc2_int"])
    df_edgelist["ipc1_int"] = df_edgelist["ipc1_int"].astype(int)
    df_edgelist["ipc2_int"] = df_edgelist["ipc2_int"].astype(int)

    # Determine embedding dimension from the first available file
    embedding_dim = None
    year_cutoff = 2006
    last_year = 2022

    for year in range(year_cutoff, last_year + 1):
        file_path = os.path.join(data_folder, f"ipc_mean_{year}.pkl")
        if os.path.exists(file_path):
            sample_embeddings = joblib.load(file_path)
            first_embedding = next(iter(sample_embeddings.values()))
            embedding_dim = len(
                first_embedding
                if not isinstance(first_embedding, np.ndarray)
                else first_embedding.tolist()
            )
            break

    if embedding_dim is None:
        raise ValueError("No embedding files found in the data folder.")

    # Pre-build one-hot IPC category matrix (same for every year)
    ipc_categories = ["a", "b", "c", "d", "e", "f", "g", "h"]
    category_to_index = {cat: i for i, cat in enumerate(ipc_categories)}

    one_hot_categories = torch.zeros(
        (num_nodes, len(ipc_categories)), dtype=torch.float32
    )
    for ipc_int, ipc_code in int_to_ipc_code.items():
        cat = ipc_code[0].lower()
        if cat in category_to_index:
            one_hot_categories[ipc_int, category_to_index[cat]] = 1.0

    data_graph_dict = {}

    for year in tqdm(
        range(year_cutoff, last_year + 1), desc="Processing years for temporal graph"
    ):
        if year not in popularity_dict:
            continue

        file_path = os.path.join(data_folder, f"ipc_mean_{year}.pkl")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping year {year}.")
            continue

        embeddings_dict = joblib.load(file_path)
        year_popularity = popularity_dict[year]

        # Collect active nodes for this year
        active_node_data: Dict[
            int, tuple
        ] = {}  # ipc_int -> (embedding, scalar_5, target)

        for pop_item in year_popularity:
            code = pop_item.ipc_code
            if code not in embeddings_dict:
                continue
            if code not in ipc_code_to_int:
                continue

            ipc_int = ipc_code_to_int[code]
            embedding = embeddings_dict[code]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            scalar = [
                float(pop_item.count),
                float(pop_item.count_prev) if pop_item.count_prev is not None else 0.0,
                float(pop_item.score_d_1) if pop_item.score_d_1 is not None else 0.0,
                float(pop_item.score_d_2) if pop_item.score_d_2 is not None else 0.0,
            ]

            active_node_data[ipc_int] = (embedding, scalar, pop_item.score)

        # Build full-size node tensors; zero-pad inactive nodes
        node_embeddings = torch.zeros((num_nodes, embedding_dim), dtype=torch.float32)
        node_scalar_raw = torch.zeros((num_nodes, 4), dtype=torch.float32)
        node_targets = torch.zeros(num_nodes, dtype=torch.float32)
        node_available = torch.zeros(num_nodes, dtype=torch.bool)

        for ipc_int, (embedding, scalar, target) in active_node_data.items():
            node_embeddings[ipc_int] = torch.tensor(embedding, dtype=torch.float32)
            node_scalar_raw[ipc_int] = torch.tensor(scalar, dtype=torch.float32)
            node_targets[ipc_int] = target
            node_available[ipc_int] = True

        # Concatenate scalar features with one-hot IPC category -> [num_nodes, 13]
        node_scalar = torch.cat([node_scalar_raw, one_hot_categories], dim=1)

        # Build edges (only between active nodes for this year)
        active_ipc_ints = set(active_node_data.keys())

        df_edgelist_year = df_edgelist[
            (df_edgelist["pub_year"] == year)
            & df_edgelist["ipc1_int"].isin(active_ipc_ints)
            & df_edgelist["ipc2_int"].isin(active_ipc_ints)
        ].copy()

        df_edgelist_prev = df_edgelist[
            (df_edgelist["pub_year"] == year - 1)
            & df_edgelist["ipc1_int"].isin(active_ipc_ints)
            & df_edgelist["ipc2_int"].isin(active_ipc_ints)
        ].copy()

        df_edgelist_merged = df_edgelist_year.merge(
            df_edgelist_prev[["ipc1", "ipc2", "salton_similarity", "weight"]],
            on=["ipc1", "ipc2"],
            how="left",
            suffixes=("", "_prev"),
        )

        edge_list = []
        edge_attrs = []

        for _, row in df_edgelist_merged.iterrows():
            salton_prev = (
                float(row["salton_similarity_prev"])
                if pd.notna(row.get("salton_similarity_prev"))
                else 0.0
            )
            weight_prev = (
                float(row["weight_prev"]) if pd.notna(row.get("weight_prev")) else 0.0
            )
            salton_current = float(row["salton_similarity"])
            weight_current = float(row["weight"])

            similarity_change = salton_current - salton_prev if salton_prev > 0 else 0.0
            weight_change = weight_current - weight_prev if weight_prev > 0 else 0.0

            # ipc1_int / ipc2_int are already global indices — no remapping needed
            edge_list.append([int(row["ipc1_int"]), int(row["ipc2_int"])])
            edge_attrs.append(
                [
                    salton_current,
                    weight_current,
                    salton_prev,
                    weight_prev,
                    similarity_change,
                    weight_change,
                ]
            )

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float32)

        year_tensor = torch.full((num_nodes,), year, dtype=torch.long)

        data_graph_dict[year] = Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_targets,
            scalar=node_scalar,
            node_available=node_available,
            year=year_tensor,
        )

    return data_graph_dict


def normalize_temp_graph_data(
    data_dict: Dict[int, Data],
    train_years,
) -> Tuple[Dict[int, Data], Dict[str, float]]:
    """
    Fits a z-score normalizer on training years only and applies it to all years
    for the temporal graph data dict produced by create_temp_graph_data_dict().

    Normalizes the raw-count features in-place:
      - Node scalar[:, 0]  -> count
      - Node scalar[:, 1]  -> count_prev
      - Edge edge_attr[:, 1] -> weight
      - Edge edge_attr[:, 3] -> weight_prev
      - Edge edge_attr[:, 5] -> weight_change

    Uses the node_available mask when fitting so that zero-padded inactive nodes
    do not bias the statistics. All other features (scores, Salton similarity,
    one-hot IPC category) are left untouched.

    Args:
        data_dict:    Dict mapping year -> PyG Data object (from create_temp_graph_data_dict).
        train_years:  Iterable of years used for training (used to fit the scaler).

    Returns:
        (data_dict, norm_stats) where norm_stats contains the mean/std values used,
        keyed as 'count_mean', 'count_std', 'weight_mean', 'weight_std'.
    """
    available_train_years = [y for y in train_years if y in data_dict]
    if not available_train_years:
        raise ValueError("None of the supplied train_years are present in data_dict.")

    # Fit: gather only active-node statistics to avoid bias from zero-padded nodes
    count_train = torch.cat(
        [
            data_dict[y].scalar[data_dict[y].node_available, 0:2]
            for y in available_train_years
        ],
        dim=0,
    )  # shape [N_active_train_nodes, 2]

    edge_weight_train = torch.cat(
        [data_dict[y].edge_attr[:, [1, 3, 5]] for y in available_train_years], dim=0
    )  # shape [N_train_edges, 3]

    count_mean = count_train.mean(dim=0)  # [2]
    count_std = count_train.std(dim=0, unbiased=False).clamp(min=1e-6)  # [2]

    weight_mean = edge_weight_train.mean(dim=0)  # [3]
    weight_std = edge_weight_train.std(dim=0, unbiased=False).clamp(min=1e-6)  # [3]

    # Transform: apply to every year (inactive nodes normalise to near-zero, still masked out)
    for data in data_dict.values():
        data.scalar[:, 0:2] = (data.scalar[:, 0:2] - count_mean) / count_std
        if data.edge_attr.shape[0] > 0:
            data.edge_attr[:, [1, 3, 5]] = (
                data.edge_attr[:, [1, 3, 5]] - weight_mean
            ) / weight_std

    norm_stats = {
        "count_mean": count_mean.tolist(),
        "count_std": count_std.tolist(),
        "weight_mean": weight_mean.tolist(),
        "weight_std": weight_std.tolist(),
    }

    print(
        f"Normalization fitted on {len(available_train_years)} training year(s).\n"
        f"  count   mean={norm_stats['count_mean']}, std={norm_stats['count_std']}\n"
        f"  weight  mean={norm_stats['weight_mean']}, std={norm_stats['weight_std']}"
    )

    return data_dict, norm_stats
