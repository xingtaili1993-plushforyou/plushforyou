"""
Candidate generation module for the Plush For You recommender system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import linear_kernel


def generate_content_candidates(user_items: List[str], all_items: List[str], 
                               tfidf_matrix: np.ndarray, item_to_idx: Dict[str, int], 
                               top_k: int = 200) -> List[str]:
    """
    Generate content-based candidates for a user.
    
    Args:
        user_items: List of item IDs the user has interacted with
        all_items: List of all item IDs
        tfidf_matrix: TF-IDF matrix
        item_to_idx: Mapping from item ID to matrix index
        top_k: Number of top candidates to return
        
    Returns:
        List of candidate item IDs
    """
    if not user_items:
        return []
    
    # Get indices of user items
    user_indices = [item_to_idx[item] for item in user_items if item in item_to_idx]
    
    if not user_indices:
        return []
    
    # Compute average user vector
    user_vector = tfidf_matrix[user_indices].mean(axis=0)
    
    # Convert to dense array for cosine_similarity
    user_vector_dense = np.asarray(user_vector).reshape(1, -1)
    
    # Compute similarities with all items (using linear_kernel for sparse matrices)
    similarities = linear_kernel(user_vector_dense, tfidf_matrix).flatten()
    
    # Get top similar items (excluding user's own items)
    user_item_set = set(user_items)
    candidate_scores = []
    
    for i, item_id in enumerate(all_items):
        if item_id not in user_item_set:
            candidate_scores.append((item_id, similarities[i]))
    
    # Sort by similarity and return top-k
    candidate_scores.sort(key=lambda x: -x[1])
    return [item_id for item_id, _ in candidate_scores[:top_k]]


def generate_covisitation_candidates(user_items: List[str], covis_neighbors: Dict[str, List[Tuple[str, float]]], 
                                   top_k: int = 200) -> List[str]:
    """
    Generate co-visitation based candidates for a user.
    
    Args:
        user_items: List of item IDs the user has interacted with
        covis_neighbors: Co-visitation neighbors dictionary
        top_k: Number of top candidates to return
        
    Returns:
        List of candidate item IDs
    """
    if not user_items:
        return []
    
    # Collect co-visitation scores
    candidate_scores = defaultdict(float)
    
    for user_item in user_items:
        if user_item in covis_neighbors:
            for neighbor_item, score in covis_neighbors[user_item]:
                candidate_scores[neighbor_item] = max(candidate_scores[neighbor_item], score)
    
    # Remove user's own items
    user_item_set = set(user_items)
    filtered_scores = {item: score for item, score in candidate_scores.items() 
                      if item not in user_item_set}
    
    # Sort by score and return top-k
    sorted_candidates = sorted(filtered_scores.items(), key=lambda x: -x[1])
    return [item_id for item_id, _ in sorted_candidates[:top_k]]


def generate_popularity_candidates(popularity_scores: Dict[str, float], user_items: List[str], 
                                 top_k: int = 200) -> List[str]:
    """
    Generate popularity-based candidates for a user.
    
    Args:
        popularity_scores: Dictionary mapping item_id to popularity score
        user_items: List of item IDs the user has interacted with
        top_k: Number of top candidates to return
        
    Returns:
        List of candidate item IDs
    """
    # Remove user's own items
    user_item_set = set(user_items)
    filtered_scores = {item: score for item, score in popularity_scores.items() 
                      if item not in user_item_set}
    
    # Sort by popularity and return top-k
    sorted_candidates = sorted(filtered_scores.items(), key=lambda x: -x[1])
    return [item_id for item_id, _ in sorted_candidates[:top_k]]


def generate_candidates_for_user(user_id: str, user_profiles: Dict, products_df: pd.DataFrame,
                               tfidf_matrix: np.ndarray, item_to_idx: Dict[str, int],
                               covis_neighbors: Dict[str, List[Tuple[str, float]]],
                               popularity_scores: Dict[str, float], 
                               content_top_k: int = 200, covis_top_k: int = 200, 
                               popularity_top_k: int = 200) -> List[str]:
    """
    Generate candidates for a specific user using multiple strategies.
    
    Args:
        user_id: User ID
        user_profiles: User profiles dictionary
        products_df: Products DataFrame
        tfidf_matrix: TF-IDF matrix
        item_to_idx: Mapping from item ID to matrix index
        covis_neighbors: Co-visitation neighbors dictionary
        popularity_scores: Popularity scores dictionary
        content_top_k: Number of content-based candidates
        covis_top_k: Number of co-visitation candidates
        popularity_top_k: Number of popularity candidates
        
    Returns:
        List of candidate item IDs
    """
    # Get user profile
    user_profile = user_profiles.get(user_id, {})
    user_items = [item['item_id'] for item in user_profile.get('interacted_items', [])]
    
    # Get all items
    all_items = products_df['item_id'].tolist()
    
    # Generate candidates from different strategies
    content_candidates = generate_content_candidates(
        user_items, all_items, tfidf_matrix, item_to_idx, content_top_k
    )
    
    covis_candidates = generate_covisitation_candidates(
        user_items, covis_neighbors, covis_top_k
    )
    
    popularity_candidates = generate_popularity_candidates(
        popularity_scores, user_items, popularity_top_k
    )
    
    # Combine and deduplicate candidates with tie-breaking scores
    candidate_scores = {}
    
    # Content-based candidates (highest weight)
    for item in content_candidates:
        candidate_scores[item] = candidate_scores.get(item, 0) + 0.5
    
    # Co-visitation candidates (medium weight)
    for item in covis_candidates:
        candidate_scores[item] = candidate_scores.get(item, 0) + 0.3
    
    # Popularity candidates (lowest weight)
    for item in popularity_candidates:
        candidate_scores[item] = candidate_scores.get(item, 0) + 0.2
    
    # Sort by combined score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
    all_candidates = [item for item, score in sorted_candidates]
    
    # Logging for monitoring
    print(f"Generated {len(all_candidates)} candidates for user {user_id}")
    print(f"  Content: {len(content_candidates)}, Co-visitation: {len(covis_candidates)}, Popularity: {len(popularity_candidates)}")
    
    return all_candidates


def handle_cold_start_user(user_id: str, products_df: pd.DataFrame, 
                          popularity_scores: Dict[str, float], 
                          brands_df: pd.DataFrame = None, k: int = 200) -> List[str]:
    """
    Handle cold start users (new users with no interaction history).
    
    Args:
        user_id: User ID
        products_df: Products DataFrame
        popularity_scores: Popularity scores dictionary
        brands_df: Brands DataFrame (optional)
        k: Number of candidates to return
        
    Returns:
        List of candidate item IDs
    """
    # For cold start, we use:
    # 1. Global popularity
    # 2. Brand diversity
    # 3. Price range diversity
    
    # Get top popular items
    sorted_popularity = sorted(popularity_scores.items(), key=lambda x: -x[1])
    popular_items = [item_id for item_id, _ in sorted_popularity[:50]]
    
    # Add price diversity
    price_diversity_items = []
    if len(products_df) > 0:
        # Create price bins for diversity
        price_bins = np.linspace(products_df['price'].min(), products_df['price'].max(), 5)
        
        for i in range(len(price_bins) - 1):
            bin_min, bin_max = price_bins[i], price_bins[i + 1]
            bin_items = products_df[
                (products_df['price'] >= bin_min) & 
                (products_df['price'] < bin_max)
            ]['item_id'].tolist()
            
            if bin_items:
                # Get top popular item in this price range
                bin_popular = [item for item in popular_items if item in bin_items]
                if bin_popular:
                    price_diversity_items.append(bin_popular[0])
                else:
                    price_diversity_items.append(bin_items[0])
    
    # Add brand diversity
    brand_diversity_items = []
    if len(products_df) > 0:
        # Get top brands from products_df
        top_brands = products_df['brand'].value_counts().head(10).index.tolist()
        
        for brand in top_brands:
            brand_items = products_df[products_df['brand'] == brand]['item_id'].tolist()
            if brand_items:
                # Prefer popular items from this brand
                brand_popular = [item for item in popular_items if item in brand_items]
                if brand_popular:
                    brand_diversity_items.append(brand_popular[0])
                else:
                    brand_diversity_items.append(brand_items[0])
    
    # Combine and deduplicate, maintaining popularity order
    seen = set()
    all_candidates = []
    
    # Add popular items first (in order)
    for item in popular_items:
        if item not in seen:
            all_candidates.append(item)
            seen.add(item)
    
    # Add brand diverse items
    for item in brand_diversity_items:
        if item not in seen:
            all_candidates.append(item)
            seen.add(item)
    
    # Add price diverse items
    for item in price_diversity_items:
        if item not in seen:
            all_candidates.append(item)
            seen.add(item)
    
    # Return top k candidates
    final_candidates = all_candidates[:k]
    
    # Logging for monitoring
    print(f"Handling cold start for user {user_id}")
    print(f"Generated {len(final_candidates)} cold start candidates")
    
    return final_candidates
