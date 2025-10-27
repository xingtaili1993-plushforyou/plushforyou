"""
Data loading and normalization module for the Plush For You recommender system.
"""

import json
import pandas as pd
from typing import Tuple
from pathlib import Path
import logging

try:
    from .config import MAX_EVENTS_PER_HOUR, OUTPUTS_DIR, DATA_DIR
except ImportError:
    from config import MAX_EVENTS_PER_HOUR, OUTPUTS_DIR, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dresses_data(data_path: str = None) -> pd.DataFrame:
    """
    Load and normalize dresses data from JSON file.
    
    Args:
        data_path: Path to the dresses JSON file
        
    Returns:
        DataFrame with normalized dress data
    """
    if data_path is None:
        data_path = Path(DATA_DIR) / "data" / "dresses.json"
    else:
        data_path = Path(data_path)
    
    logger.info(f"Loading dresses data from {data_path}")
    
    with data_path.open('r', encoding='utf-8') as f:
        dresses_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(dresses_data)
    
    # Extract and normalize fields
    products_df = pd.DataFrame({
        'item_id': df['_id'].apply(lambda x: x['$oid']),
        'name': df['name'],
        'brand': df['brand_name'],  # Use the actual brand_name field
        'price': df['price'],
        'original_price': df.get('original_price', df['price']),  # Use original_price if available, fallback to price
        'discount': df.get('discount', 0.0),  # Discount percentage
        'color': df['color'],
        'tags': df['tags'],
        'description': df['descriptions'].apply(lambda x: ' '.join([d.get('text', '') for d in x]) if isinstance(x, list) else str(x)),
        'url': df['url'],
        'retailer': df['retailer'],
        'categories': df.get('categories', [[]]).apply(lambda x: x if isinstance(x, list) else [])
    })
    
    # Basic cleaning
    products_df = products_df.dropna(subset=['name', 'brand'])
    products_df['brand'] = products_df['brand'].str.lower().str.strip()
    products_df['color'] = products_df['color'].str.lower().str.strip()
    products_df['tags'] = products_df['tags'].apply(
        lambda x: [tag.lower().strip() for tag in x] if isinstance(x, list) else []
    )
    
    print(f"Loaded {len(products_df)} dresses")
    return products_df


def load_brands_data(data_path: str = None) -> pd.DataFrame:
    """
    Load and normalize brands data from JSON file.
    
    Args:
        data_path: Path to the brands JSON file
        
    Returns:
        DataFrame with normalized brand data
    """
    if data_path is None:
        data_path = Path(DATA_DIR) / "data" / "brands.json"
    else:
        data_path = Path(data_path)
    
    logger.info(f"Loading brands data from {data_path}")
    
    with data_path.open('r', encoding='utf-8') as f:
        brands_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(brands_data)
    
    # Extract and normalize fields
    brands_df = pd.DataFrame({
        'brand_id': df['_id'].apply(lambda x: x['$oid']),
        'name': df['name'],
        'introduction': df['introduction']
    })
    
    # Basic cleaning
    brands_df = brands_df.dropna(subset=['name'])
    brands_df['name'] = brands_df['name'].str.lower().str.strip()
    
    print(f"Loaded {len(brands_df)} brands")
    return brands_df


def load_events_data(events_dir: str = None) -> pd.DataFrame:
    """
    Load and normalize events data from CSV files.
    
    Args:
        events_dir: Directory containing event CSV files
        
    Returns:
        DataFrame with unified event data
    """
    if events_dir is None:
        events_dir = Path(DATA_DIR) / "events"
    else:
        events_dir = Path(events_dir)
    
    logger.info(f"Loading events data from {events_dir}")
    
    events_list = []
    
    # Load save events
    save_path = events_dir / "save.csv"
    if save_path.exists():
        save_df = pd.read_csv(save_path)
        save_df['event_type'] = 'save'
        events_list.append(save_df)
        print(f"Loaded {len(save_df)} save events")
    
    # Load buy_click events
    buy_click_path = events_dir / "buy_click.csv"
    if buy_click_path.exists():
        buy_click_df = pd.read_csv(buy_click_path)
        buy_click_df['event_type'] = 'buy_click'
        events_list.append(buy_click_df)
        print(f"Loaded {len(buy_click_df)} buy_click events")
    
    # Load product_click events
    product_click_path = events_dir / "product_click.csv"
    if product_click_path.exists():
        product_click_df = pd.read_csv(product_click_path)
        product_click_df['event_type'] = 'product_click'
        events_list.append(product_click_df)
        print(f"Loaded {len(product_click_df)} product_click events")
    
    if not events_list:
        raise ValueError("No event files found!")
    
    # Combine all events
    events_df = pd.concat(events_list, ignore_index=True)
    
    # Normalize to unified schema: user_id, item_id, event_type, ts
    # Extract user_id and item_id from the complex column structure
    # The CSV has columns like "*.properties.distinct_id" and "*.properties.product_id"
    
    # Find the correct column names
    distinct_id_col = None
    product_id_col = None
    timestamp_col = None
    
    for col in events_df.columns:
        if 'distinct_id' in col and 'properties' in col:
            distinct_id_col = col
        # Prefer 'product_id' over 'seed_product_id'
        elif 'product_id' in col and 'properties' in col and 'seed' not in col:
            product_id_col = col
        elif 'timestamp' in col and not col.startswith('*'):
            timestamp_col = col
    
    if distinct_id_col is None or product_id_col is None or timestamp_col is None:
        print("Warning: Could not find required columns in events data")
        print(f"Available columns: {list(events_df.columns)}")
        # Create empty DataFrame with correct schema
        unified_events = pd.DataFrame(columns=['user_id', 'item_id', 'event_type', 'ts'])
    else:
        # Create unified events DataFrame efficiently (avoids fragmentation warnings)
        unified_events = pd.DataFrame({
            'user_id': events_df[distinct_id_col],
            'item_id': events_df[product_id_col],
            'event_type': events_df['event_type'],
            'ts': pd.to_datetime(events_df[timestamp_col])
        })
    
    # Remove rows with missing critical data
    unified_events = unified_events.dropna(subset=['user_id', 'item_id', 'ts'])
    
    # Sort by timestamp
    unified_events = unified_events.sort_values('ts')
    
    print(f"Total unified events: {len(unified_events)}")
    print(f"Unique users: {unified_events['user_id'].nunique()}")
    print(f"Unique items: {unified_events['item_id'].nunique()}")
    
    return unified_events


def filter_bot_users(events_df: pd.DataFrame, max_events_per_hour: int = MAX_EVENTS_PER_HOUR) -> pd.DataFrame:
    """
    Filter out potential bot users based on event frequency.
    
    Args:
        events_df: Events DataFrame
        max_events_per_hour: Maximum events per hour to consider a user as bot
        
    Returns:
        Filtered events DataFrame
    """
    print("Filtering potential bot users...")
    
    # Calculate events per hour for each user
    events_df['hour'] = events_df['ts'].dt.floor('h')
    user_hourly_counts = events_df.groupby(['user_id', 'hour']).size().reset_index(name='events_per_hour')
    
    # Find users with suspiciously high activity
    bot_users = user_hourly_counts[
        user_hourly_counts['events_per_hour'] > max_events_per_hour
    ]['user_id'].unique()
    
    print(f"Found {len(bot_users)} potential bot users")
    
    # Filter out bot users
    filtered_events = events_df[~events_df['user_id'].isin(bot_users)].copy()
    filtered_events = filtered_events.drop('hour', axis=1)
    
    print(f"Events after bot filtering: {len(filtered_events)}")
    
    return filtered_events


def save_clean_data(products_df: pd.DataFrame, events_df: pd.DataFrame, 
                   output_dir: str = OUTPUTS_DIR) -> None:
    """
    Save cleaned data to parquet files.
    
    Args:
        products_df: Clean products DataFrame
        events_df: Clean events DataFrame
        output_dir: Output directory for cleaned data
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save products data
    products_path = output_path / "products_clean.csv"
    products_df.to_csv(products_path, index=False)
    print(f"Saved clean products data to {products_path}")
    
    # Save events data
    events_path = output_path / "events_clean.csv"
    events_df.to_csv(events_path, index=False)
    print(f"Saved clean events data to {events_path}")


def main():
    """Main function to load and clean all data."""
    print("Starting data loading and cleaning process...")
    
    # Load data
    products_df = load_dresses_data()
    brands_df = load_brands_data()
    events_df = load_events_data()
    
    # Clean events data
    events_df = filter_bot_users(events_df)
    
    # Save cleaned data
    save_clean_data(products_df, events_df)
    
    print("Data loading and cleaning completed!")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Products: {len(products_df)}")
    print(f"Brands: {len(brands_df)}")
    print(f"Events: {len(events_df)}")
    print(f"Unique users: {events_df['user_id'].nunique()}")
    print(f"Unique items: {events_df['item_id'].nunique()}")
    
    # Event type distribution
    print("\nEvent type distribution:")
    print(events_df['event_type'].value_counts())
    
    return products_df, brands_df, events_df


if __name__ == "__main__":
    products_df, brands_df, events_df = main()
