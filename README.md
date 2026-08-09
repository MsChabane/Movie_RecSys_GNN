
MOVIE RECOMMENDATION SYSTEM WITH GRAPHSAGE
==========================================

This project implements a movie recommendation system using:

- MovieLens-style movie and rating data
- PyTorch Geometric
- Heterogeneous GraphSAGE
- Sentence Transformers for movie features
- FAISS for fast item similarity search
- DVC for data and pipeline versioning

The system creates a bipartite graph:

    User -- interacts/rates --> Movie

Positive ratings are converted into user-movie interaction edges.
Movie title and genres are serialized into text and converted into
dense feature vectors using a Sentence Transformer model

INSTALLATION
============

Install the dependencies:

    pip install -r requirements.txt

For Google Colab, use:

    !pip install -q kagglehub dvc sentence-transformers torch-geometric faiss-cpu PyYAML


DATASET
=======

The default dataset is downloaded from Kaggle:

    parasharmanas/movie-recommendation-system

The dataset should contain:

    movies.csv
    ratings.csv

The expected columns are:

movies.csv:

    movieId
    title
    genres

ratings.csv:

    userId
    movieId
    rating
    timestamp

Kaggle authentication may be required when downloading the dataset.
If authentication is required, configure the Kaggle API credentials before
running the ingestion stage.


PIPELINE STAGES
===============

1. DATA INGESTION

Downloads the Kaggle dataset and stores the raw files in:

    data/raw/

Run:

    python src/data_ingestion.py


2. PREPROCESSING

Loads the raw CSV files, then:

- Removes missing values
- Removes duplicate movies
- Removes duplicate user-movie ratings
- Filters positive ratings
- Selects active users and popular movies
- Converts movie data to the target schema
- Serializes movie properties as JSON

Output files:

    data/preprocessed/items.csv
    data/preprocessed/users.csv
    data/preprocessed/interactions.csv

Run:

    python src/preprocessing.py


3. FEATURE ENGINEERING

Loads the preprocessed data and:

- Creates string-to-integer ID mappings
- Serializes movie properties into text
- Generates movie embeddings using Sentence Transformers
- Creates the PyTorch Geometric heterogeneous graph
- Adds reverse user-movie edges
- Splits graph edges into train, validation, and test sets

Outputs:

    data/processed/train_data.pt
    data/processed/val_data.pt
    data/processed/test_data.pt
    data/processed/full_graph.pt

Mapping artifacts:

    artifacts/user2id.json
    artifacts/item2id.json
    artifacts/id2item.json

Run:

    python src/feature_engineering.py


4. TRAINING

The training stage:

- Loads the processed graph data
- Trains a heterogeneous GraphSAGE model
- Computes training and validation loss
- Computes Recall@10
- Computes NDCG@10
- Saves the best metrics and training history
- Creates a training plot
- Generates final user and movie embeddings
- Builds a FAISS item index

Outputs:

    artifacts/model.pt
    artifacts/metrics.json
    artifacts/training_plot.png
    artifacts/item_faiss.index
    artifacts/user_embeddings.json

Run:

    python src/train.py


5. INFERENCE

The inference stage loads:

- The FAISS movie index
- Precomputed user embeddings
- ID mappings
- Movie metadata
- User interaction history

It returns top-K movies for a user and filters movies that the user
has already rated.

Run:

    python src/inference.py


DVC PIPELINE
============

Initialize DVC in standalone mode:

    dvc init --no-scm -f

Run the complete pipeline:

    dvc repro

Display the pipeline graph:

    dvc dag

Display metrics:

    dvc metrics show

Check pipeline status:

    dvc status

The DVC stages are:

    data_ingestion
          |
          v
    preprocessing
          |
          v
    feature_engineering
          |
          v
    train


PARAMETERS
==========

Model and pipeline parameters are stored in params.yaml.

Example:

    preprocessing:
      min_rating: 3.5
      max_users: 2000
      max_movies: 3000

    feature_engineering:
      model_name: all-MiniLM-L6-v2
      val_ratio: 0.1
      test_ratio: 0.1
      neg_sampling_ratio: 1.0

    train:
      epochs: 35
      learning_rate: 0.005
      hidden_channels: 128
      weight_decay: 0.0001
      k: 10

After changing params.yaml, run:

    dvc repro

DVC will rerun only the stages affected by the changed parameters.


METRICS
=======

The metrics file is:

    artifacts/metrics.json

It contains:

- Training parameters
- Final training loss
- Final validation loss
- Best validation Recall@K
- Best validation NDCG@K
- Best metric epochs
- Per-epoch training loss
- Per-epoch validation loss
- Per-epoch Recall@K
- Per-epoch NDCG@K

View the file:

    python -m json.tool artifacts/metrics.json

Or in a notebook:

    import json

    with open("artifacts/metrics.json", "r") as file:
        metrics = json.load(file)

    print(json.dumps(metrics, indent=2))


DISPLAY THE TRAINING PLOT
========================

The plot is saved to:

    artifacts/training_plot.png

In Google Colab:

    from IPython.display import Image, display

    display(Image(filename="artifacts/training_plot.png"))

Using Matplotlib:

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    image = mpimg.imread("artifacts/training_plot.png")
    plt.figure(figsize=(16, 7))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


INFERENCE EXAMPLE
=================

Use the inference class:

    from src.inference import RecommenderInference

    recommender = RecommenderInference()

    recommendations = recommender.recommend(
        user_str_id="user_187",
        top_k=5,
        filter_seen=True
    )

    for recommendation in recommendations:
        print(recommendation)


COLD-START USERS
================

If a user is not present in user_embeddings.json, the inference module
uses the average user embedding as a fallback.

This is a basic cold-start strategy. A production system could improve
this by using:

- Popular movies
- Genre-based recommendations
- Recently trending movies
- New-user onboarding preferences
- A separate content-based recommendation model


IMPORTANT NOTES
===============

1. Graph data files are loaded using:

       torch.load(..., weights_only=False)

   This is required because PyTorch Geometric graph objects are stored
   in the files. Only load these files if they were created by a trusted
   source.

2. The current training approach uses full-graph training. This is
   suitable for small and medium-sized datasets. For very large graphs,
   use neighbor sampling with LinkNeighborLoader and install the
   required PyG sampling backend.

3. The FAISS index contains item embeddings in the same order as
   id2item.json.

4. User embeddings are stored in:

       artifacts/user_embeddings.json

5. Recommendations are generated from precomputed embeddings and do not
   run GraphSAGE during every inference request.


END-TO-END COLAB COMMANDS
=========================

Run the following commands from the project root:

    !pip install -r requirements.txt

    !dvc init --no-scm -f

    !dvc repro

    !dvc dag

    !dvc metrics show

    !python src/inference.py
