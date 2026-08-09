
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import faiss
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecommenderInference:

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        preprocessed_dir: str = "data/preprocessed",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.preprocessed_dir = Path(preprocessed_dir)

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads FAISS index, user embeddings, ID mappings, and metadata."""
        logger.info("Loading inference artifacts...")

        
        faiss_path = self.artifacts_dir / "item_faiss.index"
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. Run train.py first."
            )
        self.faiss_index = faiss.read_index(str(faiss_path))

        
        user_emb_path = self.artifacts_dir / "user_embeddings.json"
        with open(user_emb_path, "r") as f:
            self.user_embeddings = json.load(f)

        
        with open(self.artifacts_dir / "id2item.json", "r") as f:
            self.id2item = {int(k): v for k, v in json.load(f).items()}

        with open(self.artifacts_dir / "user2id.json", "r") as f:
            self.user2id = json.load(f)

        
        items_path = self.preprocessed_dir / "items.csv"
        self.items_df = pd.read_csv(items_path)
        self.item_props_map = {}
        for _, row in self.items_df.iterrows():
            self.item_props_map[row["id"]] = json.loads(row["properties"])

        # 5. Load Interactions to Mask Seen Movies
        interactions_path = self.preprocessed_dir / "interactions.csv"
        interactions_df = pd.read_csv(interactions_path)
        self.user_seen_movies = {}
        for u_id, group in interactions_df.groupby("user_id"):
            self.user_seen_movies[u_id] = set(group["item_id"].tolist())

  
        all_vectors = np.array(list(self.user_embeddings.values()))
        self.global_user_avg = np.mean(all_vectors, axis=0, keepdims=True)

        logger.info("All inference artifacts successfully loaded into memory.")

    def recommend(
        self, user_str_id: str, top_k: int = 5, filter_seen: bool = True
    ) -> List[Dict]:
        """Generates top-K recommendations for a target user ID using FAISS vector search."""
      
        if user_str_id in self.user_embeddings:
            user_vec = np.array(
                [self.user_embeddings[user_str_id]], dtype=np.float32
            )
        else:
            logger.warning(
                f"User '{user_str_id}' not found in offline store. Using cold-start global average."
            )
            user_vec = self.global_user_avg.astype(np.float32)


        fetch_k = top_k + 50 if filter_seen else top_k
        scores, indices = self.faiss_index.search(user_vec, fetch_k)

        scores = scores[0]
        indices = indices[0]

        seen_items = self.user_seen_movies.get(user_str_id, set())

        recommendations = []
        for idx, score in zip(indices, scores):
            if idx == -1:
                continue

            item_str_id = self.id2item[idx]

            
            if filter_seen and item_str_id in seen_items:
                continue

            props = self.item_props_map.get(item_str_id, {})
            recommendations.append(
                {
                    "movie_id": item_str_id,
                    "score": round(float(score), 4),
                    "title": props.get("title", "Unknown"),
                    "genres": props.get("genres", "Unknown"),
                }
            )

            if len(recommendations) == top_k:
                break

        return recommendations


if __name__ == "__main__":
    recommender = RecommenderInference()

    
    sample_user = "user_187"
    print(f"\n=======================================================")
    print(f"Top 5 Movie Recommendations for User: {sample_user}")
    print(f"=======================================================")

    results = recommender.recommend(sample_user, top_k=5)
    for i, res in enumerate(results, 1):
        print(
            f"{i}. Score: {res['score']:.4f} | Title: {res['title']} | Genres: {res['genres']}"
        )
