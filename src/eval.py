"""
Evaluation module for the Plush For You recommender system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import our modules
try:
    # Try relative imports first (for when used as a module)
    from .utils import (
        load_data_from_outputs, create_temporal_split, evaluate_recommendations,
        create_baseline_recommendations, create_content_baseline, print_evaluation_results
    )
    from .candidate_gen import generate_candidates_for_user, handle_cold_start_user
    from .rerank import rerank_candidates_simple
    from .preprocess import build_user_profiles
except ImportError:
    # Fall back to absolute imports (for when run as script)
    from utils import (
        load_data_from_outputs, create_temporal_split, evaluate_recommendations,
        create_baseline_recommendations, create_content_baseline, print_evaluation_results
    )
    from candidate_gen import generate_candidates_for_user, handle_cold_start_user
    from rerank import rerank_candidates_simple
    from preprocess import build_user_profiles


def run_offline_evaluation(split_date: str = "2025-10-15", 
                          k_values: List[int] = [5, 10, 20],
                          num_test_users: int = 10) -> Dict:
    """
    Run offline evaluation of the recommender system.
    
    Args:
        split_date: Date to split train/test data
        k_values: List of k values to evaluate
        num_test_users: Number of test users to evaluate
        
    Returns:
        Dictionary of evaluation results
    """
    print("Starting offline evaluation...")
    
    # Load data (excluding user_profiles - we'll rebuild from train split)
    products_df, _, tfidf_matrix, item_to_idx, covis_neighbors, popularity_scores, vectorizer = load_data_from_outputs()
    
    # Load events data
    events_df = pd.read_csv("outputs/events_clean.csv")
    events_df['ts'] = pd.to_datetime(events_df['ts']).dt.tz_localize(None)
    
    # Create temporal split
    train_events, test_events = create_temporal_split(events_df, split_date)
    
    # CRITICAL: Rebuild user profiles using ONLY train_events
    # (prevents test leakage - excluding items user will interact with in future)
    print("Rebuilding user profiles from train split only...")
    user_profiles = build_user_profiles(train_events, products_df)
    
    # Get test users (users with interactions in test period)
    test_users = test_events['user_id'].unique()[:num_test_users]
    print(f"Evaluating on {len(test_users)} test users")
    
    # Get all items
    all_items = products_df['item_id'].tolist()
    
    # Initialize results
    results = {}
    for model_name in ['hybrid_model', 'popularity_baseline', 'content_baseline']:
        results[model_name] = {}
        for k in k_values:
            for metric in ['hit_rate', 'ndcg', 'coverage', 'freshness']:
                results[model_name][f'{metric}@{k}'] = []
    
    # Evaluate on test users
    for i, user_id in enumerate(test_users):
        if i % 10 == 0:
            print(f"Evaluating user {i+1}/{len(test_users)}: {user_id}")
        
        # Get ground truth (items user interacted with in test period)
        user_test_events = test_events[test_events['user_id'] == user_id]
        ground_truth = user_test_events['item_id'].unique().tolist()
        
        if not ground_truth:
            continue
        
        # Generate recommendations using hybrid model
        try:
            if user_id in user_profiles:
                candidates = generate_candidates_for_user(
                    user_id, user_profiles, products_df, tfidf_matrix,
                    item_to_idx, covis_neighbors, popularity_scores
                )
                hybrid_recs = rerank_candidates_simple(
                    candidates, user_id, user_profiles, products_df,
                    tfidf_matrix, item_to_idx, covis_neighbors, popularity_scores, max(k_values)
                )
            else:
                hybrid_recs = handle_cold_start_user(user_id, products_df, popularity_scores)
        except Exception as e:
            print(f"Error generating hybrid recommendations for user {user_id}: {e}")
            hybrid_recs = []
        
        # Generate baseline recommendations
        popularity_recs = create_baseline_recommendations(products_df, popularity_scores, max(k_values))
        
        # Generate content baseline (with cold start fallback)
        user_items = []
        if user_id in user_profiles:
            user_items = [item['item_id'] for item in user_profiles[user_id].get('interacted_items', [])]
        content_recs = create_content_baseline(user_items, all_items, tfidf_matrix, item_to_idx, max(k_values), popularity_scores)
        
        # Evaluate each model
        for model_name, recs in [('hybrid_model', hybrid_recs), ('popularity_baseline', popularity_recs), ('content_baseline', content_recs)]:
            if recs:
                eval_results = evaluate_recommendations(recs, ground_truth, all_items, products_df, k_values)
                for metric, value in eval_results.items():
                    results[model_name][metric].append(value)
    
    # Compute average metrics
    final_results = {}
    for model_name in results:
        final_results[model_name] = {}
        for metric in results[model_name]:
            values = results[model_name][metric]
            if values:
                final_results[model_name][metric] = np.mean(values)
            else:
                final_results[model_name][metric] = 0.0
    
    return final_results


def save_evaluation_results(results: Dict, output_dir: str = "outputs") -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_path / f"evaluation_results_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved evaluation results to {results_path}")


def print_evaluation_summary(results: Dict) -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print("OFFLINE EVALUATION RESULTS")
    print("=" * 60)
    
    for model_name in ['hybrid_model', 'popularity_baseline', 'content_baseline']:
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        # Create table header
        print(f"{'Metric':<15} {'K=5':<10} {'K=10':<10} {'K=20':<10}")
        print("-" * 50)
        
        # Print metrics
        for metric in ['hit_rate', 'ndcg', 'coverage', 'freshness']:
            values = []
            for k in [5, 10, 20]:
                value = results[model_name].get(f'{metric}@{k}', 0.0)
                values.append(f"{value:.4f}")
            print(f"{metric:<15} {values[0]:<10} {values[1]:<10} {values[2]:<10}")


def create_evaluation_report(results: Dict, output_dir: str = "outputs") -> None:
    """
    Create a markdown evaluation report.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"evaluation_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Offline Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results Summary\n\n")
        
        # Create comparison table
        f.write("| Model | Hit Rate@5 | Hit Rate@10 | Hit Rate@20 | NDCG@20 |\n")
        f.write("|-------|-----------|------------|------------|----------|\n")
        
        for model_name in ['hybrid_model', 'popularity_baseline', 'content_baseline']:
            model_display = model_name.replace('_', ' ').title()
            hr5 = results[model_name].get('hit_rate@5', 0.0)
            hr10 = results[model_name].get('hit_rate@10', 0.0)
            hr20 = results[model_name].get('hit_rate@20', 0.0)
            ndcg20 = results[model_name].get('ndcg@20', 0.0)
            
            f.write(f"| {model_display} | {hr5:.4f} | {hr10:.4f} | {hr20:.4f} | {ndcg20:.4f} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("The evaluation uses temporal train/test split to simulate real-world performance.\n\n")
        f.write("**Key Metrics:**\n")
        f.write("- **Hit Rate@K**: Fraction of test items found in top-K recommendations\n")
        f.write("- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)\n")
        f.write("- **Coverage@K**: Fraction of catalog shown in recommendations\n\n")
        f.write("**Expected Behavior:**\n")
        f.write("- With sparse data (<100 events), simpler baselines often perform best\n")
        f.write("- The hybrid model should demonstrate improved performance over the baseline methods.\n")
    
    print(f"Created evaluation report: {report_path}")


def generate_sample_recommendations(user_id: str, k: int = 10, output_dir: str = "outputs/samples") -> Dict:
    """
    Generate sample recommendations for a user and save to file.
    
    Args:
        user_id: User ID
        k: Number of recommendations
        output_dir: Output directory
        
    Returns:
        Dictionary with recommendations
    """
    # Load data
    products_df, user_profiles, tfidf_matrix, item_to_idx, covis_neighbors, popularity_scores, vectorizer = load_data_from_outputs()
    
    # Generate recommendations
    if user_id in user_profiles:
        candidates = generate_candidates_for_user(
            user_id, user_profiles, products_df, tfidf_matrix,
            item_to_idx, covis_neighbors, popularity_scores
        )
        recommendations = rerank_candidates_simple(
            candidates, user_id, user_profiles, products_df,
            tfidf_matrix, item_to_idx, covis_neighbors, popularity_scores, k
        )
    else:
        recommendations = handle_cold_start_user(user_id, products_df, popularity_scores, k=k)
    
    # Format results
    items = []
    for item_id in recommendations:
        item_info = products_df[products_df['item_id'] == item_id]
        if len(item_info) > 0:
            item = item_info.iloc[0]
            items.append({
                "item_id": item_id,
                "name": item['name'],
                "brand": item['brand'],
                "price": float(item['price']) if pd.notna(item['price']) else None,
                "color": item['color'],
                "url": item['url']
            })
    
    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "count": len(items),
        "items": items
    }
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    filename = f"sample_recommendations_{user_id}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved sample recommendations to {filepath}")
    
    return result


def main():
    """Main evaluation function."""
    print("Starting offline evaluation...")
    
    # Run evaluation
    results = run_offline_evaluation()
    
    # Print results
    print_evaluation_summary(results)
    
    # Save results
    save_evaluation_results(results)
    
    # Create report
    create_evaluation_report(results)
    
    print("\nOffline evaluation completed!")
    
    return results


if __name__ == "__main__":
    results = main()

