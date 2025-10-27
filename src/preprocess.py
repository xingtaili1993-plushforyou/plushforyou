"""
Data preprocessing and feature engineering module for the Plush For You recommender system.
Combines preprocessing and feature engineering into one unified pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sentence_transformers import SentenceTransformer
import pickle
import math
import logging
import random
import torch

try:
    from .config import (
        RANDOM_SEED, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE, TFIDF_MAX_FEATURES,
        TFIDF_STOP_WORDS, SBERT_MODEL_NAME, SBERT_BATCH_SIZE, DECAY_RATE,
        COVIS_MAX_NEIGHBORS, COVIS_WINDOW_SIZE, EVENT_WEIGHTS
    )
except ImportError:
    from config import (
        RANDOM_SEED, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE, TFIDF_MAX_FEATURES,
        TFIDF_STOP_WORDS, SBERT_MODEL_NAME, SBERT_BATCH_SIZE, DECAY_RATE,
        COVIS_MAX_NEIGHBORS, COVIS_WINDOW_SIZE, EVENT_WEIGHTS
    )

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def create_item_text_features(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create text features for items by combining name, brand, color, tags, and description.
    
    Args:
        products_df: Products DataFrame
        
    Returns:
        DataFrame with text features added
    """
    print("Creating item text features...")
    
    def create_text_field(row):
        """Create a single text field from multiple item attributes."""
        fields = [
            str(row.get('name', '')),
            str(row.get('brand', '')),
            str(row.get('color', '')),
            ' '.join(row.get('tags', []) if isinstance(row.get('tags'), list) else []),
            str(row.get('description', ''))
        ]
        return ' '.join([field.lower().strip() for field in fields if field and field != 'nan'])
    
    products_df['text'] = products_df.apply(create_text_field, axis=1)
    
    print(f"Created text features for {len(products_df)} items")
    return products_df


def build_tfidf_features(products_df: pd.DataFrame, max_features: int = 50000) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    """
    Build TF-IDF features for items.
    
    Args:
        products_df: Products DataFrame with text field
        max_features: Maximum number of features
        
    Returns:
        Tuple of (vectorizer, tfidf_matrix)
    """
    print("Building TF-IDF features...")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES if max_features == 50000 else max_features,
        stop_words=TFIDF_STOP_WORDS,
        lowercase=True
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(products_df['text'])
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return vectorizer, tfidf_matrix


def build_sentence_embeddings(products_df: pd.DataFrame, model_name: str = SBERT_MODEL_NAME) -> Optional[np.ndarray]:
    """
    Build sentence embeddings for items using SentenceTransformer.
    
    Args:
        products_df: Products DataFrame with text field
        model_name: Name of the SentenceTransformer model
        
    Returns:
        Sentence embeddings matrix or None if failed
    """
    print(f"Building sentence embeddings using {model_name}...")
    print("This may take several minutes (~7 min for 25k items)...")
    
    try:
        # Load sentence transformer model
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        texts = products_df['text'].tolist()
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=SBERT_BATCH_SIZE)
        
        print(f"Sentence embeddings shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"Warning: Could not build sentence embeddings: {e}")
        print("Continuing without sentence embeddings...")
        return None


def compute_time_decay(days: float, decay_rate: float = DECAY_RATE) -> float:
    """
    Compute time decay factor.
    
    Args:
        days: Number of days
        decay_rate: Decay rate (lambda)
        
    Returns:
        Decay factor
    """
    return math.exp(-decay_rate * days)


def build_covisitation_matrix(events_df: pd.DataFrame, max_neighbors: int = COVIS_MAX_NEIGHBORS, window_size: int = COVIS_WINDOW_SIZE) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build co-visitation matrix from events data.
    
    Args:
        events_df: Events DataFrame
        max_neighbors: Maximum number of neighbors per item
        window_size: Time window size for co-visitation (default: 10 items)
        
    Returns:
        Dictionary mapping item_id to list of (neighbor_item_id, score) tuples
    """
    print("Building co-visitation matrix...")
    
    from collections import defaultdict
    
    # Sort events by user and timestamp
    events_df = events_df.sort_values(['user_id', 'ts'])
    
    # Build co-visitation counts
    covis_counts = defaultdict(lambda: defaultdict(float))
    
    # Group by user and process sessions
    for user_id, user_events in events_df.groupby('user_id'):
        items = user_events['item_id'].tolist()
        times = user_events['ts'].tolist()
        
        # Process each item pair within user sessions
        for i in range(len(items)):
            for j in range(max(0, i-window_size), i):  # Look at last window_size items
                if i != j:
                    item_a, item_b = items[i], items[j]
                    age_days = (pd.Timestamp.now() - times[j]).days
                    weight = compute_time_decay(age_days)
                    
                    # Weight by event type
                    event_type = user_events.iloc[i]['event_type']
                    if event_type == 'save':
                        weight *= 3.0
                    elif event_type == 'buy_click':
                        weight *= 2.0
                    else:  # product_click
                        weight *= 1.0
                    
                    covis_counts[item_a][item_b] += weight
                    covis_counts[item_b][item_a] += weight
    
    # Keep top neighbors for each item
    neighbors = {}
    for item, counts in covis_counts.items():
        sorted_neighbors = sorted(counts.items(), key=lambda x: -x[1])[:max_neighbors]
        neighbors[item] = sorted_neighbors
    
    print(f"Built co-visitation matrix for {len(neighbors)} items")
    return neighbors


def compute_popularity_scores(events_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute time-decayed popularity scores for items.
    
    Args:
        events_df: Events DataFrame
        
    Returns:
        Dictionary mapping item_id to popularity score
    """
    logger.info("Computing popularity scores...")
    
    popularity_scores = {}
    
    for item_id, item_events in events_df.groupby('item_id'):
        score = 0.0
        for _, event in item_events.iterrows():
            age_days = (pd.Timestamp.now() - event['ts']).days
            decay = compute_time_decay(age_days)
            weight = EVENT_WEIGHTS.get(event['event_type'], 1.0)
            score += weight * decay
        
        popularity_scores[item_id] = score
    
    print(f"Computed popularity scores for {len(popularity_scores)} items")
    return popularity_scores


def build_user_profiles(events_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build user profiles from interaction history.
    
    Args:
        events_df: Events DataFrame
        products_df: Products DataFrame
        
    Returns:
        Dictionary mapping user_id to profile features
    """
    print("Building user profiles...")
    
    user_profiles = {}
    
    for user_id, user_events in events_df.groupby('user_id'):
        profile = {
            'brand_affinity': {},
            'price_preferences': [],
            'interacted_items': [],
            'event_counts': {}
        }
        
        for _, event in user_events.iterrows():
            item_id = event['item_id']
            event_type = event['event_type']
            age_days = (pd.Timestamp.now() - event['ts']).days
            decay = compute_time_decay(age_days)
            
            # Get item details
            item_info = products_df[products_df['item_id'] == item_id]
            if len(item_info) > 0:
                item_info = item_info.iloc[0]
                
                # Brand affinity
                brand = item_info['brand']
                if brand not in profile['brand_affinity']:
                    profile['brand_affinity'][brand] = 0.0
                profile['brand_affinity'][brand] += decay
                
                # Price preferences
                if pd.notna(item_info['price']):
                    profile['price_preferences'].append(item_info['price'])
                
                # Track interacted items
                profile['interacted_items'].append({
                    'item_id': item_id,
                    'event_type': event_type,
                    'decay': decay
                })
            
            # Event counts
            if event_type not in profile['event_counts']:
                profile['event_counts'][event_type] = 0
            profile['event_counts'][event_type] += 1
        
        # Normalize brand affinity
        total_brand_score = sum(profile['brand_affinity'].values())
        if total_brand_score > 0:
            for brand in profile['brand_affinity']:
                profile['brand_affinity'][brand] /= total_brand_score
        
        # Compute price statistics
        if profile['price_preferences']:
            prices = np.array(profile['price_preferences'])
            profile['price_median'] = float(np.median(prices))
            profile['price_iqr'] = float(np.percentile(prices, 75) - np.percentile(prices, 25))
        else:
            profile['price_median'] = None
            profile['price_iqr'] = None
        
        user_profiles[user_id] = profile
    
    print(f"Built profiles for {len(user_profiles)} users")
    return user_profiles


def create_item_to_idx_mapping(products_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create mapping from item ID to matrix index.
    
    Args:
        products_df: Products DataFrame
        
    Returns:
        Dictionary mapping item_id to index
    """
    return {item_id: idx for idx, item_id in enumerate(products_df['item_id'])}


def save_all_artifacts(products_df: pd.DataFrame, vectorizer: TfidfVectorizer, 
                      tfidf_matrix: sparse.csr_matrix, sentence_embeddings: Optional[np.ndarray],
                      item_to_idx: Dict[str, int], covis_neighbors: Dict, 
                      popularity_scores: Dict, user_profiles: Dict,
                      output_dir: str = "outputs") -> None:
    """
    Save all preprocessed data, features, and models.
    
    Args:
        products_df: Products DataFrame
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix
        sentence_embeddings: Sentence embeddings matrix (optional)
        item_to_idx: Item to index mapping
        covis_neighbors: Co-visitation neighbors
        popularity_scores: Popularity scores
        user_profiles: User profiles
        output_dir: Output directory
    """
    print("\nSaving all artifacts...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save products with text features
    products_path = output_path / "products_with_text.csv"
    products_df.to_csv(products_path, index=False)
    print(f"[OK] Saved products with text features to {products_path}")
    
    # Save TF-IDF vectorizer
    vectorizer_path = output_path / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[OK] Saved TF-IDF vectorizer to {vectorizer_path}")
    
    # Save TF-IDF matrix as sparse matrix for memory efficiency
    tfidf_path = output_path / "tfidf_matrix.npz"
    sparse.save_npz(tfidf_path, tfidf_matrix)
    print(f"[OK] Saved TF-IDF matrix to {tfidf_path}")
    
    # Save sentence embeddings if available
    if sentence_embeddings is not None:
        embeddings_path = output_path / "sentence_embeddings.npz"
        np.savez_compressed(embeddings_path, embeddings=sentence_embeddings)
        print(f"[OK] Saved sentence embeddings to {embeddings_path}")
    
    # Save item to index mapping
    mapping_path = output_path / "item_to_idx.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(item_to_idx, f)
    print(f"[OK] Saved item to index mapping to {mapping_path}")
    
    # Save co-visitation neighbors
    covis_path = output_path / "covis_neighbors.pkl"
    with open(covis_path, 'wb') as f:
        pickle.dump(covis_neighbors, f)
    print(f"[OK] Saved co-visitation neighbors to {covis_path}")
    
    # Save popularity scores
    popularity_path = output_path / "popularity_scores.pkl"
    with open(popularity_path, 'wb') as f:
        pickle.dump(popularity_scores, f)
    print(f"[OK] Saved popularity scores to {popularity_path}")
    
    # Save user profiles
    profiles_path = output_path / "user_profiles.pkl"
    with open(profiles_path, 'wb') as f:
        pickle.dump(user_profiles, f)
    print(f"[OK] Saved user profiles to {profiles_path}")


def main():
    """Main preprocessing and feature engineering function."""
    print("=" * 70)
    print("Starting unified preprocessing and feature engineering pipeline...")
    print("=" * 70)
    
    # Load clean data
    print("\n[1/8] Loading clean data...")
    products_df = pd.read_csv("outputs/products_clean.csv")
    events_df = pd.read_csv("outputs/events_clean.csv")
    
    # Convert timestamp column to datetime and make timezone-naive
    events_df['ts'] = pd.to_datetime(events_df['ts']).dt.tz_localize(None)
    
    print(f"  - Products: {len(products_df)}")
    print(f"  - Events: {len(events_df)}")
    
    # Create text features
    print("\n[2/8] Creating text features...")
    products_df = create_item_text_features(products_df)
    
    # Build TF-IDF features
    print("\n[3/8] Building TF-IDF features...")
    vectorizer, tfidf_matrix = build_tfidf_features(products_df)
    
    # Build sentence embeddings (optional, takes ~7 minutes)
    print("\n[4/8] Building sentence embeddings...")
    sentence_embeddings = build_sentence_embeddings(products_df)
    
    # Create item to index mapping
    print("\n[5/8] Creating item to index mapping...")
    item_to_idx = create_item_to_idx_mapping(products_df)
    print(f"  - Mapped {len(item_to_idx)} items to indices")
    
    # Build co-visitation matrix
    print("\n[6/8] Building co-visitation matrix...")
    covis_neighbors = build_covisitation_matrix(events_df)
    
    # Compute popularity scores
    print("\n[7/8] Computing popularity scores...")
    popularity_scores = compute_popularity_scores(events_df)
    
    # Build user profiles
    print("\n[8/8] Building user profiles...")
    user_profiles = build_user_profiles(events_df, products_df)
    
    # Save all artifacts
    save_all_artifacts(
        products_df, vectorizer, tfidf_matrix, sentence_embeddings, item_to_idx,
        covis_neighbors, popularity_scores, user_profiles
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)
    print("\n=== Final Summary ===")
    print(f"Products with text features: {len(products_df)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    if sentence_embeddings is not None:
        print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
    print(f"Item to index mapping: {len(item_to_idx)} items")
    print(f"Co-visitation items: {len(covis_neighbors)}")
    print(f"Popularity scores: {len(popularity_scores)}")
    print(f"User profiles: {len(user_profiles)}")
    print("\n[OK] All artifacts saved to outputs/")


if __name__ == "__main__":
    main()
