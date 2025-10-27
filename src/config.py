"""
Configuration file for the Plush For You recommender system.
Centralizes all hyperparameters and constants for reproducibility.
"""

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
SAMPLES_DIR = "outputs/samples"

# ============================================================================
# EVENT WEIGHTS (for popularity and co-visitation)
# ============================================================================
EVENT_WEIGHTS = {
    'save': 3.0,           # Highest intent
    'buy_click': 2.0,      # Medium intent
    'product_click': 1.0   # Lowest intent
}

# ============================================================================
# TIME DECAY
# ============================================================================
DECAY_RATE = 0.03  # ~30-day half-life: exp(-0.03 * 30) ≈ 0.41

# ============================================================================
# TF-IDF PARAMETERS
# ============================================================================
TFIDF_MIN_DF = 3
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 50000
TFIDF_STOP_WORDS = 'english'

# ============================================================================
# SENTENCE EMBEDDINGS
# ============================================================================
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
SBERT_BATCH_SIZE = 32

# ============================================================================
# CO-VISITATION
# ============================================================================
COVIS_MAX_NEIGHBORS = 50
COVIS_WINDOW_SIZE = 10  # Look at last 10 items in user session

# ============================================================================
# CANDIDATE GENERATION
# ============================================================================
CANDIDATE_CONTENT_TOP_K = 200
CANDIDATE_COVIS_TOP_K = 200
CANDIDATE_POPULARITY_TOP_K = 200

# Tie-breaking weights for candidate deduplication
CANDIDATE_WEIGHT_CONTENT = 0.5
CANDIDATE_WEIGHT_COVIS = 0.3
CANDIDATE_WEIGHT_POPULARITY = 0.2

# ============================================================================
# COLD START
# ============================================================================
COLD_START_POPULAR_ITEMS = 50
COLD_START_PRICE_BINS = 5
COLD_START_TOP_BRANDS = 10
COLD_START_CANDIDATES_K = 200

# ============================================================================
# RE-RANKING WEIGHTS (Adaptive)
# ============================================================================

# For sparse data (< 3 items or no co-visitation)
RERANK_SPARSE_WEIGHTS = {
    'content': 0.90,
    'covis': 0.02,
    'brand': 0.03,
    'popularity': 0.02,
    'price': 0.02,
    'freshness': 0.01
}

# For richer data
RERANK_RICH_WEIGHTS = {
    'content': 0.60,
    'covis': 0.15,
    'brand': 0.10,
    'popularity': 0.08,
    'price': 0.05,
    'freshness': 0.02
}

# Threshold for sparse vs rich
RERANK_SPARSE_THRESHOLD = 3  # num user items

# ============================================================================
# DIVERSITY FILTERS
# ============================================================================
MAX_ITEMS_PER_BRAND = 2
MAX_ITEMS_PER_PRICE_RANGE = 3
MMR_LAMBDA = 0.7  # 70% relevance, 30% diversity

# ============================================================================
# BOT FILTERING
# ============================================================================
MAX_EVENTS_PER_HOUR = 50  # Threshold for bot detection

# ============================================================================
# EVALUATION
# ============================================================================
EVAL_SPLIT_DATE = "2025-10-15"  # Temporal split date
EVAL_K_VALUES = [5, 10, 20]
EVAL_NUM_TEST_USERS = 10

# ============================================================================
# API SETTINGS
# ============================================================================
API_HOST = "127.0.0.1"
API_PORT = 8000
API_DEFAULT_K = 20  # Default number of recommendations

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

