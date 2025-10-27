"""
Re-ranking module for the Plush For You recommender system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import linear_kernel
import math
from collections import defaultdict


def mmr_rank(item_vecs: np.ndarray, scores: np.ndarray, topk: int = 20, lam: float = 0.7) -> List[int]:
    """
    Apply Maximal Marginal Relevance (MMR) ranking to diversify results.
    
    Args:
        item_vecs: Item vectors (n_items, n_features)
        scores: Initial scores for items
        topk: Number of items to return
        lam: Lambda parameter for MMR (0.7 = 70% relevance, 30% diversity)
        
    Returns:
        List of selected item indices
    """
    if len(scores) == 0:
        return []
    
    idxs = list(range(len(scores)))
    selected = []
    selected_vecs = []
    
    # Convert to dense for easier manipulation
    item_vecs_dense = np.asarray(item_vecs.todense()) if hasattr(item_vecs, 'todense') else item_vecs
    
    for _ in range(min(topk, len(scores))):
        if selected_vecs:
            # Compute similarity to already selected items (using linear_kernel for sparse matrices)
            sims = linear_kernel(
                item_vecs_dense[idxs], 
                np.asarray(np.mean(selected_vecs, axis=0)).reshape(1, -1)
            ).ravel()
        else:
            sims = np.zeros(len(idxs))
        
        # Compute MMR score
        mmr_scores = lam * scores[idxs] - (1 - lam) * sims
        
        # Select item with highest MMR score
        j = int(np.argmax(mmr_scores))
        selected.append(idxs[j])
        selected_vecs.append(item_vecs_dense[idxs[j]])
        idxs.pop(j)
    
    return selected


def apply_diversity_filters(ranked_items: List[str], products_df: pd.DataFrame, 
                          max_per_brand: int = 2, max_per_price_range: int = 3) -> List[str]:
    """
    Apply diversity filters to ensure variety in recommendations.
    
    Args:
        ranked_items: List of ranked item IDs
        products_df: Products DataFrame
        max_per_brand: Maximum items per brand
        max_per_price_range: Maximum items per price range
        
    Returns:
        Filtered list of item IDs
    """
    if not ranked_items:
        return []
    
    # Get item info for all ranked items
    item_info = products_df[products_df['item_id'].isin(ranked_items)].copy()
    if len(item_info) == 0:
        return ranked_items
    
    # Create price ranges
    price_ranges = pd.cut(item_info['price'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    item_info['price_range'] = price_ranges
    
    # Track counts
    brand_counts = defaultdict(int)
    price_range_counts = defaultdict(int)
    filtered_items = []
    
    for item_id in ranked_items:
        item_row = item_info[item_info['item_id'] == item_id]
        if len(item_row) == 0:
            continue
            
        brand = item_row.iloc[0]['brand']
        price_range = item_row.iloc[0]['price_range']
        
        # Check if we can add this item
        if (brand_counts[brand] < max_per_brand and 
            price_range_counts[price_range] < max_per_price_range):
            
            filtered_items.append(item_id)
            brand_counts[brand] += 1
            price_range_counts[price_range] += 1
    
    return filtered_items


def compute_heuristic_score(item_id: str, user_profile: Dict, products_df: pd.DataFrame,
                           tfidf_matrix: np.ndarray, item_to_idx: Dict[str, int],
                           covis_neighbors: Dict[str, List[Tuple[str, float]]],
                           popularity_scores: Dict[str, float]) -> float:
    """
    Compute heuristic score for an item based on multiple factors.
    Uses adaptive weighting based on signal strength.
    
    Args:
        item_id: Item ID
        user_profile: User profile dictionary
        products_df: Products DataFrame
        tfidf_matrix: TF-IDF matrix
        item_to_idx: Mapping from item ID to matrix index
        covis_neighbors: Co-visitation neighbors dictionary
        popularity_scores: Popularity scores dictionary
        
    Returns:
        Heuristic score
    """
    # Get item info
    item_info = products_df[products_df['item_id'] == item_id]
    if len(item_info) == 0:
        return 0.0
    
    item_info = item_info.iloc[0]
    
    # 1. Content similarity
    sim_content = 0.0
    if item_id in item_to_idx:
        item_idx = item_to_idx[item_id]
        user_items = [item['item_id'] for item in user_profile.get('interacted_items', [])]
        
        if user_items:
            user_indices = [item_to_idx[item] for item in user_items if item in item_to_idx]
            if user_indices:
                user_vector = tfidf_matrix[user_indices].mean(axis=0)
                user_vector_dense = np.asarray(user_vector).reshape(1, -1)
                similarity = linear_kernel(user_vector_dense, tfidf_matrix[item_idx:item_idx+1])[0, 0]
                sim_content = float(similarity)
    
    # 2. Co-visitation score
    score_covis = 0.0
    has_covis = False
    user_items = [item['item_id'] for item in user_profile.get('interacted_items', [])]
    for user_item in user_items:
        if user_item in covis_neighbors:
            has_covis = True
            for neighbor_item, score in covis_neighbors[user_item]:
                if neighbor_item == item_id:
                    score_covis = max(score_covis, score)
                    break
    
    # 3. Brand affinity
    brand_affinity = 0.0
    brand = item_info['brand']
    user_brand_affinity = user_profile.get('brand_affinity', {})
    if brand in user_brand_affinity:
        brand_affinity = user_brand_affinity[brand]
    
    # 4. Popularity score
    pop_score = popularity_scores.get(item_id, 0.0)
    if pop_score > 0:
        pop_score = min(pop_score / 100.0, 1.0)
    
    # 5. Price match
    price_match = 0.0
    user_price_median = user_profile.get('price_median')
    if user_price_median is not None and pd.notna(item_info['price']):
        price_diff = abs(item_info['price'] - user_price_median) / user_price_median
        price_match = math.exp(-price_diff)
    
    # 6. Freshness
    freshness = 1.0
    
    # ADAPTIVE WEIGHTING: If collaborative signals are weak/missing, rely more on content
    num_user_items = len(user_items)
    
    if num_user_items < 3 or not has_covis:
        # Sparse data: Use content-heavy weighting (90%)
        final_score = (
            0.90 * sim_content +      # Very high (sparse data)
            0.02 * score_covis +      # Minimal
            0.03 * brand_affinity +   # Minimal
            0.02 * pop_score +        # Minimal
            0.02 * price_match +      # Minimal
            0.01 * freshness          # Minimal
        )
    else:
        # More data available: Use balanced weighting
        final_score = (
            0.60 * sim_content +      # Still high but balanced
            0.15 * score_covis +      # Moderate
            0.10 * brand_affinity +   # Moderate
            0.08 * pop_score +        # Low
            0.05 * price_match +      # Low
            0.02 * freshness          # Low
        )
    
    return final_score


def rerank_candidates(candidates: List[str], user_id: str, user_profiles: Dict,
                     products_df: pd.DataFrame, tfidf_matrix: np.ndarray,
                     item_to_idx: Dict[str, int], covis_neighbors: Dict[str, List[Tuple[str, float]]],
                     popularity_scores: Dict[str, float], topk: int = 20) -> List[str]:
    """
    Re-rank candidates using heuristic scoring and MMR.
    
    Args:
        candidates: List of candidate item IDs
        user_id: User ID
        user_profiles: User profiles dictionary
        products_df: Products DataFrame
        tfidf_matrix: TF-IDF matrix
        item_to_idx: Mapping from item ID to matrix index
        covis_neighbors: Co-visitation neighbors dictionary
        popularity_scores: Popularity scores dictionary
        topk: Number of items to return
        
    Returns:
        List of re-ranked item IDs
    """
    if not candidates:
        return []
    
    # Get user profile
    user_profile = user_profiles.get(user_id, {})
    
    # Compute heuristic scores for all candidates
    candidate_scores = []
    for item_id in candidates:
        score = compute_heuristic_score(
            item_id, user_profile, products_df, tfidf_matrix,
            item_to_idx, covis_neighbors, popularity_scores
        )
        candidate_scores.append(score)
    
    # Convert to numpy arrays
    scores = np.array(candidate_scores)
    
    # Get item vectors for MMR
    valid_candidates = []
    item_indices = []
    valid_scores = []
    
    for i, item_id in enumerate(candidates):
        if item_id in item_to_idx:
            valid_candidates.append(item_id)
            item_indices.append(item_to_idx[item_id])
            valid_scores.append(candidate_scores[i])
    
    if not item_indices:
        # Fallback to simple ranking if no vectors available
        sorted_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in sorted_indices[:topk]]
    
    item_vecs = tfidf_matrix[item_indices]
    valid_scores = np.array(valid_scores)
    
    # Apply MMR ranking
    selected_indices = mmr_rank(item_vecs, valid_scores, topk)
    
    # Get MMR-ranked items
    mmr_items = [valid_candidates[i] for i in selected_indices]
    
    # Apply diversity filters
    diverse_items = apply_diversity_filters(mmr_items, products_df, max_per_brand=2, max_per_price_range=3)
    
    # If diversity filtering removed too many items, fall back to MMR results
    if len(diverse_items) < topk // 2:
        return mmr_items[:topk]
    
    return diverse_items[:topk]


def rerank_candidates_simple(candidates: List[str], user_id: str, user_profiles: Dict,
                            products_df: pd.DataFrame, tfidf_matrix: np.ndarray,
                            item_to_idx: Dict[str, int], covis_neighbors: Dict[str, List[Tuple[str, float]]],
                            popularity_scores: Dict[str, float], topk: int = 20) -> List[str]:
    """
    Simple re-ranking without MMR (faster but less diverse).
    
    Args:
        candidates: List of candidate item IDs
        user_id: User ID
        user_profiles: User profiles dictionary
        products_df: Products DataFrame
        tfidf_matrix: TF-IDF matrix
        item_to_idx: Mapping from item ID to matrix index
        covis_neighbors: Co-visitation neighbors dictionary
        popularity_scores: Popularity scores dictionary
        topk: Number of items to return
        
    Returns:
        List of re-ranked item IDs
    """
    if not candidates:
        return []
    
    # Get user profile
    user_profile = user_profiles.get(user_id, {})
    
    # Compute heuristic scores for all candidates
    candidate_scores = []
    for item_id in candidates:
        score = compute_heuristic_score(
            item_id, user_profile, products_df, tfidf_matrix,
            item_to_idx, covis_neighbors, popularity_scores
        )
        candidate_scores.append((item_id, score))
    
    # Sort by score and return top-k
    candidate_scores.sort(key=lambda x: -x[1])
    return [item_id for item_id, _ in candidate_scores[:topk]]
