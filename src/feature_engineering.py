
import json
import logging
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(
        self,
        input_dir: str = "data/preprocessed",
        output_dir: str = "data/processed",
        artifacts_dir: str = "artifacts",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.model_name = model_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def serialize_properties(self, prop_str: str) -> str:
        try:
            props = json.loads(prop_str)
            return " ".join([f"[{k}] {v}" for k, v in props.items()])
        except Exception:
            return ""

    def process(self) -> dict:
        items_df = pd.read_csv(self.input_dir / "items.csv")
        users_df = pd.read_csv(self.input_dir / "users.csv")
        interactions_df = pd.read_csv(self.input_dir / "interactions.csv")

        user2id = {uid: int(i) for i, uid in enumerate(users_df["id"].unique())}
        item2id = {iid: int(i) for i, iid in enumerate(items_df["id"].unique())}
        id2item = {v: k for k, v in item2id.items()}

        with open(self.artifacts_dir / "user2id.json", "w") as f:
            json.dump(user2id, f)
        with open(self.artifacts_dir / "item2id.json", "w") as f:
            json.dump(item2id, f)
        with open(self.artifacts_dir / "id2item.json", "w") as f:
            json.dump(id2item, f)

        interactions_df["user_idx"] = interactions_df["user_id"].map(user2id)
        interactions_df["item_idx"] = interactions_df["item_id"].map(item2id)

        items_df["text_repr"] = items_df["properties"].apply(self.serialize_properties)
        embedder = SentenceTransformer(self.model_name)
        item_embeddings = embedder.encode(items_df["text_repr"].tolist(), convert_to_tensor=True, show_progress_bar=False)

        item_idx_map = {iid: i for i, iid in enumerate(items_df["id"])}
        item_features = torch.zeros((len(item2id), item_embeddings.shape[1]))

        for iid, idx in item2id.items():
            orig_idx = item_idx_map[iid]
            item_features[idx] = item_embeddings[orig_idx]

        data = HeteroData()
        data["user"].num_nodes = len(user2id)
        data["item"].x = item_features

        u_idx = interactions_df["user_idx"].to_numpy()
        i_idx = interactions_df["item_idx"].to_numpy()
        edge_index = torch.from_numpy(np.vstack([u_idx, i_idx])).long()

        data["user", "interacts", "item"].edge_index = edge_index
        data = T.ToUndirected()(data)

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=True,
            neg_sampling_ratio=1.0,
            edge_types=("user", "interacts", "item"),
            rev_edge_types=("item", "rev_interacts", "user"),
        )

        train_data, val_data, test_data = transform(data)

        torch.save(train_data, self.output_dir / "train_data.pt")
        torch.save(val_data, self.output_dir / "val_data.pt")
        torch.save(test_data, self.output_dir / "test_data.pt")
        torch.save(data, self.output_dir / "full_graph.pt")

        logger.info(f"Saved PyG Graph data tensors to {self.output_dir}")
        return {"processed_dir": str(self.output_dir)}

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["feature_engineering"]

    fe = FeatureEngineering(model_name=params["model_name"])
    fe.process()
