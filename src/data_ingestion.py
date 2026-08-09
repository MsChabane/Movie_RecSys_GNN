
import os
import glob
import shutil
import logging
from pathlib import Path
import kagglehub


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, dataset_name: str = "parasharmanas/movie-recommendation-system"):
        self.dataset_name = dataset_name

      
        self.dirs = [
            "data/raw",
            "data/preprocessed",
            "data/processed",
           
        ]
        self._create_directories()

    def _create_directories(self):
        """Create all project directories if they do not exist."""
        for directory in self.dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")

    def download_data(self) -> dict:
        """Download Kaggle dataset and save raw CSV files to data/raw/."""
        logger.info(f"Downloading Kaggle dataset: {self.dataset_name}")
        download_path = kagglehub.dataset_download(self.dataset_name)

        
        movies_files = glob.glob(os.path.join(download_path, "**", "movies.csv"), recursive=True)
        ratings_files = glob.glob(os.path.join(download_path, "**", "ratings.csv"), recursive=True)

        if not movies_files or not ratings_files:
            raise FileNotFoundError("Target CSV files ('movies.csv', 'ratings.csv') were not found.")

        raw_dir = Path("data/raw")
        dest_movies = raw_dir / "movies.csv"
        dest_ratings = raw_dir / "ratings.csv"

      
        shutil.copy(movies_files[0], dest_movies)
        shutil.copy(ratings_files[0], dest_ratings)

        logger.info(f"Successfully copied raw files to:")
        logger.info(f" - {dest_movies}")
        logger.info(f" - {dest_ratings}")

        return {
            "movies_path": str(dest_movies),
            "ratings_path": str(dest_ratings)
        }

if __name__ == "__main__":
    ingestion = DataIngestion()
    paths = ingestion.download_data()
    print("\n[Data Ingestion Completed] Output paths:", paths)
