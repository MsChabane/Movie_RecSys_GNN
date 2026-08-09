
import json
import logging
import yaml
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessing:
    def __init__(
        self,
        raw_dir: str = "data/raw",
        output_dir: str = "data/preprocessed",
        min_rating: float = 3.5,
        max_users: int = 2000,
        max_movies: int = 3000,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_rating = min_rating
        self.max_users = max_users
        self.max_movies = max_movies

    def process(self) -> dict:
        movies_path = self.raw_dir / "movies.csv"
        ratings_path = self.raw_dir / "ratings.csv"

        if not movies_path.exists() or not ratings_path.exists():
            raise FileNotFoundError(f"Raw data missing in {self.raw_dir}.")

        raw_movies = pd.read_csv(movies_path)
        raw_ratings = pd.read_csv(ratings_path)

        raw_movies = raw_movies.dropna(subset=["movieId", "title"]).drop_duplicates(subset=["movieId"])
        raw_ratings = raw_ratings.dropna(subset=["userId", "movieId", "rating"]).drop_duplicates(subset=["userId", "movieId"])

        pos_ratings = raw_ratings[raw_ratings["rating"] >= self.min_rating].copy()

        top_users = pos_ratings["userId"].value_counts().head(self.max_users).index
        top_movies = pos_ratings["movieId"].value_counts().head(self.max_movies).index

        pos_ratings = pos_ratings[pos_ratings["userId"].isin(top_users) & pos_ratings["movieId"].isin(top_movies)].copy()
        filtered_movies = raw_movies[raw_movies["movieId"].isin(top_movies)].copy()

        items_list = []
        for _, row in filtered_movies.iterrows():
            genres_formatted = str(row["genres"]).replace("|", ", ") if pd.notna(row.get("genres")) else "Unknown"
            props = {"title": str(row["title"]), "genres": genres_formatted}
            items_list.append({"id": f"movie_{int(row['movieId'])}", "properties": json.dumps(props)})

        items_df = pd.DataFrame(items_list)

        pos_ratings["user_id"] = "user_" + pos_ratings["userId"].astype(int).astype(str)
        pos_ratings["item_id"] = "movie_" + pos_ratings["movieId"].astype(int).astype(str)

        users_df = pd.DataFrame({"id": pos_ratings["user_id"].unique()})
        interactions_df = pos_ratings[["user_id", "item_id", "rating", "timestamp"]].reset_index(drop=True)

        items_out = self.output_dir / "items.csv"
        users_out = self.output_dir / "users.csv"
        interactions_out = self.output_dir / "interactions.csv"

        items_df.to_csv(items_out, index=False)
        users_df.to_csv(users_out, index=False)
        interactions_df.to_csv(interactions_out, index=False)

        logger.info(f"Saved preprocessed datasets to {self.output_dir}")
        return {"items_path": str(items_out), "users_path": str(users_out), "interactions_path": str(interactions_out)}

if __name__ == "__main__":
  
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocessing"]

    processor = DataPreprocessing(
        min_rating=params["min_rating"],
        max_users=params["max_users"],
        max_movies=params["max_movies"],
    )
    processor.process()
