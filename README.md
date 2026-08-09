
# Movie Recommendation System (Heterogeneous GraphSAGE + FAISS)

A scalable, production-grade recommender system built using Graph Neural Networks (GNNs), vector similarity search, and modern MLOps practices. 

The architecture models user-item interactions as a bipartite graph, leveraging **Heterogeneous GraphSAGE** (via PyTorch Geometric) for structural representation learning and **SentenceTransformers** for semantic item feature serialization. The pipeline is fully orchestrated and version-controlled with **DVC**, served via a **FastAPI** REST backend, and monitored through a **Streamlit** evaluation dashboard.

---

## Core Technologies

* **Deep Learning & GNNs**: PyTorch, PyTorch Geometric (`torch_geometric`)
* **Semantic Feature Extraction**: SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector Search**: FAISS (Facebook AI Similarity Search)
* **Pipeline & Data Versioning**: DVC (Data Version Control)
* **REST API**: FastAPI, Uvicorn, Pydantic
* **Dashboard & Evaluation**: Streamlit, Matplotlib, Pandas
* **Ingestion**: KaggleHub API

---

## Architecture Overview

1. **Bipartite Graph Construction**:
   * **Nodes**: User nodes (trainable embeddings) and Item nodes (dense text embeddings derived from JSON property serialization).
   * **Edges**: Interaction edges (`interacts` / `rev_interacts`) created from positive ratings ($\ge 3.5$ stars).

2. **GraphSAGE Propagation**:
   * Message passing accumulates 2-hop neighborhood representations across user and item nodes.
   * Final latent representations are $L_2$-normalized to enforce Cosine Similarity geometry in the embedding space.

3. **Sub-Millisecond Inference**:
   * Offline pipeline extracts item node representations into a **FAISS Index** (`IndexFlatIP`).
   * Inference fetches user vectors and executes top-$K$ inner-product vector search while masking out historical interactions.

---

## Installation & Setup

### 1. Prerequisites
Python 3.10+ and `pip` are required.

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Pipeline Execution (DVC)

The entire workflow—from data download to model training and index generation—is managed as a reproducible Directed Acyclic Graph (DAG) using **DVC**.

### Pipeline Stages

1. **Data Ingestion** (`src/data_ingestion.py`): Downloads raw data (`movies.csv`, `ratings.csv`) to `data/raw/`.
2. **Preprocessing** (`src/preprocessing.py`): Cleans NaNs, filters ratings ($\ge 3.5$), serializes movie metadata into JSON attributes, and outputs cleaned CSVs to `data/preprocessed/`.
3. **Feature Engineering** (`src/feature_engineering.py`): Encodes movie properties using SentenceTransformers, constructs PyG `HeteroData`, splits positive edges (`RandomLinkSplit`), and saves graph tensors to `data/processed/`.
4. **Training** (`src/train.py`): Trains GraphSAGE, tracks metrics (Recall@K, NDCG@K, Loss), writes `artifacts/metrics.json`, builds the FAISS index, and exports user embeddings.

### Pipeline Commands

* **Run Full Pipeline**:
  ```bash
  dvc repro
  ```

* **Force Rerun a Specific Stage**:
  ```bash
  dvc repro -f train
  ```

* **Inspect Pipeline Graph**:
  ```bash
  dvc dag
  ```

* **View Performance Metrics**:
  ```bash
  dvc metrics show
  ```

---

## Hyperparameter Configuration (`params.yaml`)

All pipeline parameters are managed centrally in `params.yaml`. Modifying any parameter triggers DVC to rerun only the affected downstream stages upon running `dvc repro`.

```yaml
preprocessing:
  min_rating: 3.5
  max_users: 2000
  max_movies: 3000

feature_engineering:
  model_name: "all-MiniLM-L6-v2"
  val_ratio: 0.1
  test_ratio: 0.1
  neg_sampling_ratio: 1.0

train:
  epochs: 35
  learning_rate: 0.005
  hidden_channels: 128
  weight_decay: 0.0001
  k: 10
```

---

## Serving the REST API (FastAPI)

The API loads precomputed FAISS indices and user embeddings into memory during startup for low-latency serving.

### Start Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Key Endpoints

* `GET /health`: Health check and status of loaded FAISS index and user vectors.
* `GET /recommend/{user_id}?top_k=5&filter_seen=true`: Retrieve Top-$K$ recommendations for a user.
* `POST /recommend`: Accepts JSON payload for recommendation requests.
* `GET /docs`: Interactive Swagger API documentation UI.

### Example Request (`POST /recommend`)
```bash
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user_187",
  "top_k": 5,
  "filter_seen": true
}'
```

---

## Streamlit Evaluation Dashboard

The dashboard provides an interactive UI to inspect model metrics, training loss curves, hyperparameter configurations, and model predictions evaluated directly against held-out test data (`data/processed/test_data.pt`).

### Run Dashboard
```bash
streamlit run dashboard.py --server.port 8501
```

### Dashboard Features
* **Metrics & Parameters**: View summary metrics (`metrics.json`), loss history, ranking metric curves (Recall@10, NDCG@10), and active `params.yaml` parameters.
* **Test Set Comparison**: Select any user to evaluate Top-$K$ model predictions against positive ground truth edges from `test_data.pt`. Matches are automatically tagged with hit badges (`🎯 HIT!`).

---

## Offline Inference Usage

For standalone Python execution without running the web server:

```python
from src.inference import RecommenderInference

# Initialize engine (loads precomputed FAISS index & embeddings)
engine = RecommenderInference(
    artifacts_dir="artifacts",
    preprocessed_dir="data/preprocessed"
)

# Generate recommendations
recommendations = engine.recommend(
    user_str_id="user_187",
    top_k=5,
    filter_seen=True
)

for rec in recommendations:
    print(f"Score: {rec['score']:.4f} | Title: {rec['title']} | Genres: {rec['genres']}")
```
