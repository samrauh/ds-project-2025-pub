from typing import List, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import TensorDataset
from torch_geometric.data import Data


class YearlyTensorDataset(TensorDataset):
    """
    Dataset class that holds scalar features, embeddings, targets, and year info.
    """

    def __init__(
        self, year: int, scalar_tensor, embedding_tensor, target_tensor, ipc_code
    ):
        self.year: int = year
        self.scalar = scalar_tensor
        self.embeddings = embedding_tensor
        self.targets = target_tensor
        self.ipc_code = ipc_code

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            self.scalar[index],
            self.embeddings[index],
            self.targets[index],
            self.year,
        )


class IPCPopularity(BaseModel):
    """Schema for IPC code popularity data."""

    ipc_code: str
    pub_year: int
    count: int
    count_prev: Optional[int] = None
    score: float
    score_d_1: Optional[float] = None
    score_d_2: Optional[float] = None
    is_top_quartile: bool
    ipc_int: Optional[int] = Field(
        default=None, description="Integer mapping of the IPC code"
    )

    @staticmethod
    def create_mapping_from_sequences(
        sequences: dict[int, list["IPCPopularity"]],
    ) -> dict[str, int]:
        """Creates a mapping from IPC codes to integers from a dictionary of year -> list of IPCPopularity."""
        codes = set()
        for items in sequences.values():
            for item in items:
                codes.add(item.ipc_code)
        return {code: i for i, code in enumerate(sorted(list(codes)))}

    def to_sample(self, embedding: List[float]) -> "IPCSample":
        """Converts IPCPopularity data to IPCSample format for model input."""
        if self.ipc_int is None:
            raise ValueError(
                f"IPC code {self.ipc_code} has not been mapped to an integer yet. Please assign 'ipc_int'."
            )

        ipc_categories = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ipc_category = self.ipc_code[0].lower()
        one_hot = [1.0 if cat == ipc_category else 0.0 for cat in ipc_categories]

        scalar_features = [
            float(self.count),
            float(self.count_prev) if self.count_prev is not None else 0.0,
            float(self.score_d_1) if self.score_d_1 is not None else 0.0,
            float(self.score_d_2) if self.score_d_2 is not None else 0.0,
        ] + one_hot

        return IPCSample(
            ipc_code_int=self.ipc_int,
            year=self.pub_year,
            embedding=embedding,
            scalar_features=scalar_features,
            target=self.score,
        )


class IPCSample(BaseModel):
    """
    Schema for a single IPC code's features in a specific year.

    Features include:
    - ID
    - Year of the data
    - Embedding vector representing the IPC code
    - Scalar features associated with the IPC code
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ipc_code_int: int
    year: int
    embedding: List[float] = Field(..., min_length=1)
    scalar_features: List[float] = Field(..., min_length=1)
    target: float


class YearlyTensorData(BaseModel):
    """
    Schema for yearly tensor data including scalar features, embeddings, targets, and IPC codes.
    """

    year: int
    items: List[IPCSample]

    def to_tensor_dataset(self) -> YearlyTensorDataset:
        scalar_tensor = torch.tensor(
            [item.scalar_features for item in self.items], dtype=torch.float
        )
        embedding_tensor = torch.tensor(
            [item.embedding for item in self.items], dtype=torch.float
        )
        target_tensor = torch.tensor(
            [item.target for item in self.items], dtype=torch.float
        )
        ipc_codes = torch.tensor(
            [item.ipc_code_int for item in self.items], dtype=torch.long
        )

        return YearlyTensorDataset(
            year=self.year,
            scalar_tensor=scalar_tensor,
            embedding_tensor=embedding_tensor,
            target_tensor=target_tensor,
            ipc_code=ipc_codes,
        )


class GraphNodeFeatures(BaseModel):
    """
    Schema for node features in the graph.

    scalar_features contains (in order):
    - [0] count: Current year count
    - [1] count_prev: Previous year count (0 if not available)
    - [2] score_d_1: this years score
    - [3] score_d_2: previous years ago score (0 if not available)
    """

    ipc_code: str
    ipc_int: int
    embedding: List[float]
    scalar_features: List[float]  # 5 features as documented above
    target: float


class GraphEdgeFeatures(BaseModel):
    """
    Schema for edge features in the graph with temporal information.

    Contains both current year and previous year features to capture temporal dynamics.
    """

    ipc1: str
    ipc2: str
    ipc1_int: int
    ipc2_int: int
    pub_year: int

    # Current year features
    salton_similarity: float
    weight: float  # Co-occurrence count

    # Previous year features (Optional - may not exist for first year or new edges)
    salton_similarity_prev: Optional[float] = None
    weight_prev: Optional[float] = None

    # Derived temporal change features
    similarity_change: Optional[float] = (
        None  # salton_similarity - salton_similarity_prev
    )
    weight_change: Optional[float] = None  # weight - weight_prev


class YearlyGraphData(BaseModel):
    """
    Schema for yearly graph data containing nodes, edges, and metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    node_features: List[GraphNodeFeatures]
    edge_features: List[GraphEdgeFeatures]

    def to_pyg_data(self) -> Data:
        """
        Converts the validated graph data to a PyTorch Geometric Data object.
        Returns a Data object with:
        - x: node features (embeddings) - shape [num_nodes, embedding_dim]
        - edge_index: connectivity in COO format - shape [2, num_edges]
        - edge_attr: edge features - shape [num_edges, 6] containing:
            [0] salton_similarity (current)
            [1] weight (current co-occurrence count)
            [2] salton_similarity_prev (previous year, 0 if N/A)
            [3] weight_prev (previous year, 0 if N/A)
            [4] similarity_change (current - previous, 0 if N/A)
            [5] weight_change (current - previous, 0 if N/A)
        - y: node targets (next year's score) - shape [num_nodes]
        - scalar: additional node scalar features - shape [num_nodes, 5] containing:
            [0] count (current)
            [1] count_prev (previous year, 0 if N/A)
            [2] score_d_1 (score current year, 0 if N/A)
            [3] score_d_2 (score previous years, 0 if N/A)
            [4] ipc_category (one-hot encoded category of the IPC code)
        - year: year tensor for each node - shape [num_nodes]
        """
        if not self.node_features:
            raise ValueError(f"No node features found for year {self.year}")

        # Create mapping from ipc_int to position in the tensor
        ipc_int_to_pos = {node.ipc_int: i for i, node in enumerate(self.node_features)}

        # Node features (embeddings)
        node_embeddings = torch.tensor(
            [node.embedding for node in self.node_features], dtype=torch.float32
        )

        # Scalar features with one-hot encoded IPC category
        ipc_categories = ["a", "b", "c", "d", "e", "f", "g", "h"]
        category_to_index = {cat: i for i, cat in enumerate(ipc_categories)}

        scalar_with_category = []
        for node in self.node_features:
            ipc_category = node.ipc_code[0]  # First character is the category
            one_hot = [
                1.0
                if category_to_index[cat] == category_to_index[ipc_category]
                else 0.0
                for cat in ipc_categories
            ]
            scalar_with_category.append(node.scalar_features + one_hot)

        scalar_features = torch.tensor(scalar_with_category, dtype=torch.float32)

        # Targets
        targets = torch.tensor(
            [node.target for node in self.node_features], dtype=torch.float32
        )

        # Edge index (convert ipc_int to positions) and edge attributes
        edge_list = []
        edge_attrs = []

        for edge in self.edge_features:
            if edge.ipc1_int in ipc_int_to_pos and edge.ipc2_int in ipc_int_to_pos:
                source_pos = ipc_int_to_pos[edge.ipc1_int]
                target_pos = ipc_int_to_pos[edge.ipc2_int]
                edge_list.append([source_pos, target_pos])

                # Create edge attribute vector: [salton_similarity, weight, salton_similarity_prev,
                #                                 weight_prev, similarity_change, weight_change]
                # Use 0.0 for None values
                edge_attr_vector = [
                    edge.salton_similarity,
                    edge.weight,
                    edge.salton_similarity_prev
                    if edge.salton_similarity_prev is not None
                    else 0.0,
                    edge.weight_prev if edge.weight_prev is not None else 0.0,
                    edge.similarity_change
                    if edge.similarity_change is not None
                    else 0.0,
                    edge.weight_change if edge.weight_change is not None else 0.0,
                ]
                edge_attrs.append(edge_attr_vector)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(
                (0, 6), dtype=torch.float32
            )  # 6-dimensional edge features

        # Year tensor
        year_tensor = torch.tensor(
            [self.year] * len(self.node_features), dtype=torch.long
        )

        return Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=targets,
            scalar=scalar_features,
            year=year_tensor,
        )
