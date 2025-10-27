"""
FastAPI service for the Plush For You recommender system.
"""

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from datetime import datetime
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from .candidate_gen import generate_candidates_for_user, handle_cold_start_user
from .rerank import rerank_candidates, rerank_candidates_simple, apply_diversity_filters

app = FastAPI(
    title="Plush For You Recommender API",
    description="A hybrid recommender system for dresses",
    version="1.0.0"
)

# Global variables to store loaded data
products_df = None
user_profiles = None
tfidf_matrix = None
item_to_idx = None
covis_neighbors = None
popularity_scores = None
vectorizer = None


@app.on_event("startup")
async def startup_event():
    """Load all required data and models on startup."""
    global products_df, user_profiles, tfidf_matrix, item_to_idx, covis_neighbors, popularity_scores, vectorizer
    
    logger.info("Loading recommender system data...")
    
    try:
        # Load products data
        products_df = pd.read_csv("outputs/products_with_text.csv")
        logger.info(f"Loaded {len(products_df)} products")
        
        # Load user profiles
        with open("outputs/user_profiles.pkl", 'rb') as f:
            user_profiles = pickle.load(f)
        logger.info(f"Loaded profiles for {len(user_profiles)} users")
        
        # Load TF-IDF matrix
        from scipy import sparse
        tfidf_matrix = sparse.load_npz("outputs/tfidf_matrix.npz")
        logger.info(f"Loaded TF-IDF matrix with shape {tfidf_matrix.shape}")
        
        # Load item to index mapping
        with open("outputs/item_to_idx.pkl", 'rb') as f:
            item_to_idx = pickle.load(f)
        logger.info(f"Loaded item to index mapping for {len(item_to_idx)} items")
        
        # Load co-visitation neighbors
        with open("outputs/covis_neighbors.pkl", 'rb') as f:
            covis_neighbors = pickle.load(f)
        logger.info(f"Loaded co-visitation neighbors for {len(covis_neighbors)} items")
        
        # Load popularity scores
        with open("outputs/popularity_scores.pkl", 'rb') as f:
            popularity_scores = pickle.load(f)
        logger.info(f"Loaded popularity scores for {len(popularity_scores)} items")
        
        # Load TF-IDF vectorizer
        with open("outputs/tfidf_vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("Loaded TF-IDF vectorizer")
        
        logger.info("All data loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise e


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Plush For You Recommender API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/for_you")
async def for_you(user_id: str, k: int = 20):
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: User ID
        k: Number of recommendations to return
        
    Returns:
        JSON response with recommendations
    """
    try:
        # Check if user exists in profiles
        if user_id not in user_profiles:
            logger.info(f"Cold start user: {user_id}")
            # Handle cold start
            candidates = handle_cold_start_user(user_id, products_df, popularity_scores)
        else:
            logger.info(f"Generating candidates for user: {user_id}")
            # Generate candidates
            candidates = generate_candidates_for_user(
                user_id, user_profiles, products_df, tfidf_matrix,
                item_to_idx, covis_neighbors, popularity_scores
            )
        
        if not candidates:
            return {
                "user_id": user_id,
                "items": [],
                "message": "No recommendations available"
            }
        
        # Re-rank candidates
        logger.info(f"Re-ranking {len(candidates)} candidates for user: {user_id}")
        ranked_items = rerank_candidates_simple(
            candidates, user_id, user_profiles, products_df,
            tfidf_matrix, item_to_idx, covis_neighbors, popularity_scores, k
        )
        
        # Apply diversity filters
        filtered_items = apply_diversity_filters(ranked_items, products_df)
        
        # Get final recommendations
        final_items = filtered_items[:k] if len(filtered_items) >= k else ranked_items[:k]
        
        # Format response
        recommendations = []
        for item_id in final_items:
            item_info = products_df[products_df['item_id'] == item_id]
            if len(item_info) > 0:
                item = item_info.iloc[0]
                recommendations.append({
                    "item_id": item_id,
                    "name": item['name'],
                    "brand": item['brand'],
                    "price": float(item['price']) if pd.notna(item['price']) else None,
                    "color": item['color'],
                    "url": item['url'],
                    "retailer": item['retailer']
                })
        
        return {
            "user_id": user_id,
            "items": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/debug/user/{user_id}")
async def debug_user(user_id: str):
    """
    Get debug information about a user's profile.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON response with user profile information
    """
    try:
        if user_id not in user_profiles:
            return {
                "user_id": user_id,
                "status": "cold_start",
                "message": "User not found in profiles"
            }
        
        user_profile = user_profiles[user_id]
        
        # Get top brands
        brand_affinity = user_profile.get('brand_affinity', {})
        top_brands = sorted(brand_affinity.items(), key=lambda x: -x[1])[:5]
        
        # Get price preferences
        price_median = user_profile.get('price_median')
        price_iqr = user_profile.get('price_iqr')
        
        # Get interaction summary
        interacted_items = user_profile.get('interacted_items', [])
        event_counts = user_profile.get('event_counts', {})
        
        return {
            "user_id": user_id,
            "status": "active",
            "top_brands": [{"brand": brand, "affinity": float(score)} for brand, score in top_brands],
            "price_preferences": {
                "median": float(price_median) if price_median else None,
                "iqr": float(price_iqr) if price_iqr else None
            },
            "interaction_summary": {
                "total_interactions": len(interacted_items),
                "event_counts": event_counts
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user debug info: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": all([
            products_df is not None,
            user_profiles is not None,
            tfidf_matrix is not None,
            item_to_idx is not None,
            covis_neighbors is not None,
            popularity_scores is not None,
            vectorizer is not None
        ])
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        return {
            "products_count": len(products_df) if products_df is not None else 0,
            "users_count": len(user_profiles) if user_profiles is not None else 0,
            "items_with_covis": len(covis_neighbors) if covis_neighbors is not None else 0,
            "items_with_popularity": len(popularity_scores) if popularity_scores is not None else 0,
            "tfidf_matrix_shape": tfidf_matrix.shape if tfidf_matrix is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
