"""
Utility functions for the Plush For You recommender system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import math


def load_data_from_outputs(output_dir: str = "outputs") -> Tuple:
    """
    Load all preprocessed data from outputs directory.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Tuple of loaded data
    """
    output_path = Path(output_dir)
    
    # Load products
    products_df = pd.read_csv(output_path / "products_with_text.csv")
    
    # Load user profiles
    with open(output_path / "user_profiles.pkl", 'rb') as f:
        user_profiles = pickle.load(f)
    
    # Load TF-IDF matrix
    from scipy import sparse
    tfidf_matrix = sparse.load_npz(output_path / "tfidf_matrix.npz")
    
    # Load item to index mapping
    with open(output_path / "item_to_idx.pkl", 'rb') as f:
        item_to_idx = pickle.load(f)
    
    # Load co-visitation neighbors
    with open(output_path / "covis_neighbors.pkl", 'rb') as f:
        covis_neighbors = pickle.load(f)
    
    # Load popularity scores
    with open(output_path / "popularity_scores.pkl", 'rb') as f:
        popularity_scores = pickle.load(f)
    
    # Load TF-IDF vectorizer
    with open(output_path / "tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    return (products_df, user_profiles, tfidf_matrix, item_to_idx, 
            covis_neighbors, popularity_scores, vectorizer)


def save_sample_recommendations(recommendations: Dict, user_id: str, 
                               output_dir: str = "outputs/samples") -> None:
    """
    Save sample recommendations to JSON file.
    
    Args:
        recommendations: Recommendations dictionary
        user_id: User ID
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"recommendations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    print(f"Saved sample recommendations to {filepath}")


def create_temporal_split(events_df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create temporal split of events data.
    
    Args:
        events_df: Events DataFrame
        split_date: Split date in YYYY-MM-DD format
        
    Returns:
        Tuple of (train_events, test_events)
    """
    split_timestamp = pd.to_datetime(split_date)
    
    train_events = events_df[events_df['ts'] <= split_timestamp].copy()
    test_events = events_df[events_df['ts'] > split_timestamp].copy()
    
    print(f"Temporal split at {split_date}:")
    print(f"  Train events: {len(train_events)}")
    print(f"  Test events: {len(test_events)}")
    
    return train_events, test_events


def compute_hit_rate_at_k(recommendations: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute Hit Rate@K.
    
    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of ground truth item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Hit Rate@K
    """
    if k == 0:
        return 0.0
    
    top_k_recs = set(recommendations[:k])
    ground_truth_set = set(ground_truth)
    
    hits = len(top_k_recs.intersection(ground_truth_set))
    return hits / min(k, len(ground_truth_set)) if ground_truth_set else 0.0


def compute_ndcg_at_k(recommendations: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute NDCG@K.
    
    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of ground truth item IDs
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K
    """
    if k == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    ground_truth_set = set(ground_truth)
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(top_k_recs):
        if item in ground_truth_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
    
    # Compute IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(k, len(ground_truth_set))):
        idcg += 1.0 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_coverage_at_k(recommendations: List[str], all_items: List[str], k: int) -> float:
    """
    Compute Coverage@K (unique items shown).
    
    Args:
        recommendations: List of recommended item IDs
        all_items: List of all available items
        k: Number of top recommendations to consider
        
    Returns:
        Coverage@K
    """
    if k == 0 or not all_items:
        return 0.0
    
    top_k_recs = set(recommendations[:k])
    all_items_set = set(all_items)
    
    return len(top_k_recs.intersection(all_items_set)) / len(all_items_set)


def compute_freshness_at_k(recommendations: List[str], products_df: pd.DataFrame, k: int) -> float:
    """
    Compute Freshness@K (average item age in top-K).
    
    Args:
        recommendations: List of recommended item IDs
        products_df: Products DataFrame
        k: Number of top recommendations to consider
        
    Returns:
        Freshness@K (higher is better)
    """
    if k == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    freshness_scores = []
    
    for item_id in top_k_recs:
        # For now, use a simple freshness metric
        # In a real system, you might have item creation dates
        freshness_scores.append(1.0)  # Placeholder
    
    return np.mean(freshness_scores) if freshness_scores else 0.0


def evaluate_recommendations(recommendations: List[str], ground_truth: List[str], 
                           all_items: List[str], products_df: pd.DataFrame, 
                           k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Evaluate recommendations using multiple metrics.
    
    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of ground truth item IDs
        all_items: List of all available items
        products_df: Products DataFrame
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    for k in k_values:
        results[f"hit_rate@{k}"] = compute_hit_rate_at_k(recommendations, ground_truth, k)
        results[f"ndcg@{k}"] = compute_ndcg_at_k(recommendations, ground_truth, k)
        results[f"coverage@{k}"] = compute_coverage_at_k(recommendations, all_items, k)
        results[f"freshness@{k}"] = compute_freshness_at_k(recommendations, products_df, k)
    
    return results


def create_baseline_recommendations(products_df: pd.DataFrame, popularity_scores: Dict[str, float], 
                                  k: int = 20) -> List[str]:
    """
    Create baseline popularity-based recommendations.
    
    Args:
        products_df: Products DataFrame
        popularity_scores: Popularity scores dictionary
        k: Number of recommendations
        
    Returns:
        List of recommended item IDs
    """
    # Sort by popularity
    sorted_items = sorted(popularity_scores.items(), key=lambda x: -x[1])
    return [item_id for item_id, _ in sorted_items[:k]]


def create_content_baseline(user_items: List[str], all_items: List[str], 
                          tfidf_matrix: np.ndarray, item_to_idx: Dict[str, int], 
                          k: int = 20, popularity_scores: Dict[str, float] = None) -> List[str]:
    """
    Create content-based baseline recommendations.
    Falls back to popularity for cold start users.
    
    Args:
        user_items: List of user's interacted items
        all_items: List of all items
        tfidf_matrix: TF-IDF matrix
        item_to_idx: Item to index mapping
        k: Number of recommendations
        popularity_scores: Popularity scores for cold start fallback
        
    Returns:
        List of recommended item IDs
    """
    if not user_items:
        # Cold start: fall back to popularity
        if popularity_scores:
            sorted_items = sorted(popularity_scores.items(), key=lambda x: -x[1])
            return [item_id for item_id, _ in sorted_items[:k]]
        return []
    
    # Get user vector
    user_indices = [item_to_idx[item] for item in user_items if item in item_to_idx]
    if not user_indices:
        return []
    
    user_vector = tfidf_matrix[user_indices].mean(axis=0)
    
    # Compute similarities (convert to dense for cosine_similarity)
    user_vector_dense = np.asarray(user_vector).reshape(1, -1)
    from sklearn.metrics.pairwise import linear_kernel
    similarities = linear_kernel(user_vector_dense, tfidf_matrix).flatten()
    
    # Get top similar items
    item_scores = [(all_items[i], similarities[i]) for i in range(len(all_items))]
    item_scores.sort(key=lambda x: -x[1])
    
    return [item_id for item_id, _ in item_scores[:k]]


def print_evaluation_results(results: Dict, model_name: str = "Model") -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary of evaluation results
        model_name: Name of the model
    """
    print(f"\n=== {model_name} Evaluation Results ===")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    
    for metric, value in results.items():
        print(f"{metric:<20} {value:<10.4f}")


