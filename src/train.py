
import json
import logging
import yaml
from pathlib import Path
import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, SAGEConv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NormalizedGraphSAGE(nn.Module):
    def __init__(self, num_users, item_feature_dim, hidden_channels):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, item_feature_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)

        self.conv1 = HeteroConv({
            ("user", "interacts", "item"): SAGEConv((-1, -1), hidden_channels),
            ("item", "rev_interacts", "user"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("user", "interacts", "item"): SAGEConv(hidden_channels, hidden_channels),
            ("item", "rev_interacts", "user"): SAGEConv(hidden_channels, hidden_channels),
        }, aggr="mean")

    def encode(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.normalize(x, p=2, dim=-1) for key, x in x_dict.items()}
        return x_dict

    def decode(self, z_user, z_item, edge_label_index):
        u_idx, i_idx = edge_label_index[0], edge_label_index[1]
        return (z_user[u_idx] * z_item[i_idx]).sum(dim=-1)

    def forward(self, graph_data):
        x_dict = {"user": self.user_emb.weight, "item": graph_data["item"].x}
        z_dict = self.encode(x_dict, graph_data.edge_index_dict)
        return self.decode(z_dict["user"], z_dict["item"], graph_data["user", "interacts", "item"].edge_label_index)

class ModelTrainer:
    def __init__(self, processed_dir="data/processed", artifacts_dir="artifacts", epochs=35, lr=0.005, hidden_channels=128, k=10):
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.epochs = epochs
        self.lr = lr
        self.hidden_channels = hidden_channels
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_ranking_metrics(self, model, train_graph, eval_graph):
        model.eval()
        x_dict = {"user": model.user_emb.weight, "item": eval_graph["item"].x}
        z_dict = model.encode(x_dict, eval_graph.edge_index_dict)
        z_user, z_item = z_dict["user"].cpu(), z_dict["item"].cpu()

        train_edges = train_graph["user", "interacts", "item"].edge_index.cpu()
        train_user_items = {}
        for u, i in zip(train_edges[0].numpy(), train_edges[1].numpy()):
            train_user_items.setdefault(u, set()).add(i)

        val_edges = eval_graph["user", "interacts", "item"].edge_label_index[:, eval_graph["user", "interacts", "item"].edge_label == 1].cpu()
        user_ground_truth = {}
        for u, i in zip(val_edges[0].numpy(), val_edges[1].numpy()):
            user_ground_truth.setdefault(u, set()).add(i)

        recalls, ndcgs = [], []
        for u, actual_items in user_ground_truth.items():
            u_emb = z_user[u].unsqueeze(0)
            scores = torch.matmul(u_emb, z_item.T).squeeze(0)

            if u in train_user_items:
                scores[list(train_user_items[u])] = -10000.0

            top_k_items = torch.topk(scores, k=self.k).indices.tolist()

            hits = len(set(top_k_items) & actual_items)
            recalls.append(hits / len(actual_items))

            dcg = sum([1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k_items) if item in actual_items])
            idcg = sum([1.0 / np.log2(r + 2) for r in range(min(self.k, len(actual_items)))])
            ndcgs.append(dcg / idcg if idcg > 0 else 0)

        return float(np.mean(recalls)), float(np.mean(ndcgs))

    def train_and_evaluate(self):
        train_data = torch.load(self.processed_dir / "train_data.pt", weights_only=False)
        val_data = torch.load(self.processed_dir / "val_data.pt", weights_only=False)
        full_graph = torch.load(self.processed_dir / "full_graph.pt", weights_only=False)

        with open(self.artifacts_dir / "user2id.json") as f:
            user2id = json.load(f)

        model = NormalizedGraphSAGE(len(user2id), full_graph["item"].x.shape[1], self.hidden_channels).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        train_data_dev = train_data.to(self.device)
        val_data_dev = val_data.to(self.device)

        history = {"train_loss": [], "val_loss": [], "val_recall": [], "val_ndcg": []}

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            train_preds = model(train_data_dev)
            train_loss = criterion(train_preds, train_data_dev["user", "interacts", "item"].edge_label.float())
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(val_data_dev)
                val_loss = criterion(val_preds, val_data_dev["user", "interacts", "item"].edge_label.float())

            val_recall, val_ndcg = self._compute_ranking_metrics(model, train_data, val_data)

            history["train_loss"].append(float(train_loss.item()))
            history["val_loss"].append(float(val_loss.item()))
            history["val_recall"].append(val_recall)
            history["val_ndcg"].append(val_ndcg)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {train_loss.item():.4f} | Val Recall@{self.k}: {val_recall:.4f} | Val NDCG@{self.k}: {val_ndcg:.4f}")

        
        torch.save(model.state_dict(), self.artifacts_dir / "model.pt")

        # 2. Save Comprehensive Metrics JSON (All Summary + Full History)
        best_ndcg_idx = int(np.argmax(history["val_ndcg"]))
        best_recall_idx = int(np.argmax(history["val_recall"]))

        all_metrics = {
            "parameters": {
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "hidden_channels": self.hidden_channels,
                "top_k": self.k,
            },
            "summary": {
                "best_val_ndcg": round(history["val_ndcg"][best_ndcg_idx], 4),
                "best_val_ndcg_epoch": best_ndcg_idx + 1,
                "best_val_recall": round(history["val_recall"][best_recall_idx], 4),
                "best_val_recall_epoch": best_recall_idx + 1,
                "final_train_loss": round(history["train_loss"][-1], 4),
                "final_val_loss": round(history["val_loss"][-1], 4),
                "final_val_recall": round(history["val_recall"][-1], 4),
                "final_val_ndcg": round(history["val_ndcg"][-1], 4),
            },
            "history": {
                "epoch": list(range(1, self.epochs + 1)),
                "train_loss": [round(x, 4) for x in history["train_loss"]],
                "val_loss": [round(x, 4) for x in history["val_loss"]],
                "val_recall_at_k": [round(x, 4) for x in history["val_recall"]],
                "val_ndcg_at_k": [round(x, 4) for x in history["val_ndcg"]],
            },
        }

        metrics_file = self.artifacts_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Saved complete metrics report to {metrics_file}")

        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.epochs + 1), history["train_loss"], label="Train Loss")
        plt.plot(range(1, self.epochs + 1), history["val_loss"], label="Val Loss")
        plt.title("BCE Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.epochs + 1), history["val_recall"], label=f"Val Recall@{self.k}")
        plt.plot(range(1, self.epochs + 1), history["val_ndcg"], label=f"Val NDCG@{self.k}")
        plt.title("Validation Metrics")
        plt.legend()
        plt.savefig(self.artifacts_dir / "training_plot.png")
        plt.close()

        
        model.eval()
        with torch.no_grad():
            full_dev = full_graph.to(self.device)
            z_dict = model.encode({"user": model.user_emb.weight, "item": full_dev["item"].x}, full_dev.edge_index_dict)
            z_user, z_item = z_dict["user"].cpu().numpy(), z_dict["item"].cpu().numpy()

        faiss_index = faiss.IndexFlatIP(z_item.shape[1])
        faiss_index.add(z_item.astype(np.float32))
        faiss.write_index(faiss_index, str(self.artifacts_dir / "item_faiss.index"))

        user_emb_dict = {u_str: z_user[u_idx].tolist() for u_str, u_idx in user2id.items()}
        with open(self.artifacts_dir / "user_embeddings.json", "w") as f:
            json.dump(user_emb_dict, f)

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    trainer = ModelTrainer(
        epochs=params["epochs"],
        lr=params["learning_rate"],
        hidden_channels=params["hidden_channels"],
        k=params["k"],
    )
    trainer.train_and_evaluate()
