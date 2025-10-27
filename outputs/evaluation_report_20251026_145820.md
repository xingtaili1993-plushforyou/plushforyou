# Offline Evaluation Report

**Date**: 2025-10-26 14:58:20

## Results Summary

| Model | Hit Rate@5 | Hit Rate@10 | Hit Rate@20 | NDCG@20 |
|-------|------------|------------|------------|----------|
| Hybrid Model | 0.1000 | 0.0600 | 0.0577 | 0.0824 |
| Popularity Baseline | 0.0400 | 0.0500 | 0.0577 | 0.0639 |
| Content Baseline | 0.0200 | 0.0200 | 0.0342 | 0.0430 |

## Interpretation

The evaluation uses temporal train/test split to simulate real-world performance.

**Key Metrics:**
- **Hit Rate@K**: Fraction of test items found in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)
- **Coverage@K**: Fraction of catalog shown in recommendations

**Expected Behavior:**
- With sparse data (<100 events), simpler baselines often perform best
- The hybrid model should demonstrate improved performance over the baseline methods.
