import json
import yaml
from pathlib import Path
import pandas as pd
import torch
import streamlit as st

# Import RecommenderInference
from src.inference import RecommenderInference

# Page Config
st.set_page_config(
    page_title="GNN Movie Recommender - Test Set Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

ARTIFACTS_DIR = Path("artifacts")
PREPROCESSED_DIR = Path("data/preprocessed")
PROCESSED_DIR = Path("data/processed")
PARAMS_FILE = Path("params.yaml")


# Load Metrics & Params
@st.cache_data
def load_metrics():
    path = ARTIFACTS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_params():
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            return yaml.safe_load(f)
    return None

@st.cache_resource
def load_inference_engine():
    return RecommenderInference(
        artifacts_dir=str(ARTIFACTS_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR)
    )

# Load ONLY test_data.pt and item metadata (NO train_data loaded)
@st.cache_data
def load_test_set_data():
    items_path = PREPROCESSED_DIR / "items.csv"
    test_path = PROCESSED_DIR / "test_data.pt"
    
    items_df = pd.read_csv(items_path)
    
    # Parse title & genres
    def parse_props(p):
        try:
            d = json.loads(p)
            return d.get("title", "Unknown"), d.get("genres", "Unknown")
        except:
            return "Unknown", "Unknown"
            
    parsed = items_df["properties"].apply(parse_props)
    items_df["title"] = [p[0] for p in parsed]
    items_df["genres"] = [p[1] for p in parsed]
    
    test_user_items = {}
    
    if test_path.exists():
        # Load PyG test split graph tensor
        test_data = torch.load(test_path, weights_only=False)
        
        # Load ID mappers
        with open(ARTIFACTS_DIR / "id2item.json") as f:
            id2item = {int(k): v for k, v in json.load(f).items()}
        with open(ARTIFACTS_DIR / "user2id.json") as f:
            user2id = json.load(f)
        id2user = {v: k for k, v in user2id.items()}
        
        # Extract Positive Test Edges (where edge_label == 1)
        test_edges = (
            test_data["user", "interacts", "item"]
            .edge_label_index[:, test_data["user", "interacts", "item"].edge_label == 1]
            .cpu()
            .numpy()
        )
        
        for u_idx, i_idx in zip(test_edges[0], test_edges[1]):
            u_str = id2user.get(u_idx)
            i_str = id2item.get(i_idx)
            if u_str and i_str:
                test_user_items.setdefault(u_str, set()).add(i_str)
                
    return items_df, test_user_items


metrics = load_metrics()
params = load_params()
engine = load_inference_engine()
items_df, test_user_items = load_test_set_data()

# Clean Sidebar
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("**GraphSAGE + FAISS Architecture**")
st.sidebar.divider()

if engine:
    st.sidebar.info(
        f"📊 **Users Indexed:** {len(engine.user_embeddings):,}\n\n"
        f"🎥 **Movies Indexed:** {engine.faiss_index.ntotal:,}"
    )

# Dashboard Tabs
tab1, tab2 = st.tabs([
    "📊 Metrics & Parameters", 
    "🧪 Test Set Comparison (Actual vs Predicted)"
])


# TAB 1: METRICS & PARAMETERS
with tab1:
    st.header("📈 Training Metrics & Configuration")
    
    if metrics and "summary" in metrics:
        summary = metrics["summary"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Train Loss", f"{summary.get('final_train_loss', 0):.4f}")
        c2.metric("Final Val Loss", f"{summary.get('final_val_loss', 0):.4f}")
        c3.metric("Final Recall@10", f"{summary.get('final_val_recall', 0):.4f}")
        c4.metric("Final NDCG@10", f"{summary.get('final_val_ndcg', 0):.4f}")
        
    st.divider()
    
    col_chart, col_params = st.columns([1.5, 1])
    
    with col_chart:
        st.subheader("📉 Epoch History")
        if metrics and "history" in metrics:
            hist = metrics["history"]
            
            st.markdown("**Loss Curve**")
            st.line_chart({
                "Train Loss": hist.get("train_loss", []),
                "Val Loss": hist.get("val_loss", [])
            })
            
            st.markdown("**Ranking Metrics Curve**")
            st.line_chart({
                "Recall@10": hist.get("val_recall_at_k", []),
                "NDCG@10": hist.get("val_ndcg_at_k", [])
            })
        else:
            st.warning("History not found in metrics.json")
            
    with col_params:
        st.subheader("⚙️ Hyperparameters (`params.yaml`)")
        if params:
            st.json(params)
        else:
            st.warning("params.yaml not found.")


# TAB 2: MODEL TESTING (COMPARE PREDICTIONS WITH test_data.pt ONLY)
with tab2:
    st.header("🧪 Test Set Evaluation")
    st.markdown("Compare actual ground truth movies in **`data/processed/test_data.pt`** with **Model Predictions**.")
    
    # Get users who have ground truth items in the test set
    test_users = sorted([u for u in test_user_items.keys() if len(test_user_items[u]) > 0])
    
    if not test_users:
        st.error("No test user interactions found in `test_data.pt`.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            selected_user = st.selectbox("Select Test User ID:", test_users, index=0)
        with c2:
            top_k = st.slider("Top-K Predictions:", min_value=1, max_value=20, value=5)
        with c3:
            filter_seen = st.checkbox("Filter Seen Movies", value=True)
            
        if st.button("🚀 Evaluate Model Predictions", type="primary", use_container_width=True):
            
            # 1. Fetch Actual Ground Truth Movies from test_data.pt ONLY
            actual_test_item_ids = test_user_items.get(selected_user, set())
            actual_test_movies = items_df[items_df["id"].isin(actual_test_item_ids)].reset_index(drop=True)
            
            # 2. Get Model Recommendations via RecommenderInference
            preds = engine.recommend(
                user_str_id=selected_user,
                top_k=top_k,
                filter_seen=filter_seen
            )
            pred_df = pd.DataFrame(preds)
            
            st.divider()
            
            col_act, col_pred = st.columns(2)
            
            # Display Test Ground Truth
            with col_act:
                st.subheader(f"🎯 Actual Test Ground Truth (`test_data.pt`) - {len(actual_test_movies)} Movies")
                st.caption("Held-out positive test movies for this user.")
                if not actual_test_movies.empty:
                    st.dataframe(actual_test_movies[["id", "title", "genres"]], use_container_width=True, height=400)
                else:
                    st.info("No test movies found for this user.")
                    
            # Display Model Predictions
            with col_pred:
                st.subheader(f"🔮 Model Top-{top_k} Predictions")
                st.caption("Predicted using GraphSAGE + FAISS Vector Search.")
                if not pred_df.empty:
                    # Check if predicted movie exists in test_data.pt
                    pred_df["Hit in Test Set?"] = pred_df["movie_id"].apply(
                        lambda x: "🎯 HIT!" if x in actual_test_item_ids else "✨ New Candidate"
                    )
                    disp_pred = pred_df[["Hit in Test Set?", "score", "title", "genres"]].rename(columns={"score": "Affinity Score"})
                    st.dataframe(disp_pred, use_container_width=True, height=400)
                    
                    # Hit Summary
                    hits_count = sum(1 for m_id in pred_df["movie_id"] if m_id in actual_test_item_ids)
                    if hits_count > 0:
                        st.success(f"🎉 **{hits_count} Hit(s)** found in predictions out of Top-{top_k} matching `test_data.pt`!")
                    else:
                        st.info("💡 Model predicted other unobserved candidate movies.")
                else:
                    st.warning("No predictions returned.")
