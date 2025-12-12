"""
MemePulse vs FluWave Real-Time Cultural vs Health Trend Analyzer

A Streamlit dashboard that compares real-time meme activity from GIPHY 
with flu-related search trends from Google Trends.
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any, List, Callable, TypeVar
from pytrends.request import TrendReq
from textblob import TextBlob
import plotly.graph_objects as go
from scipy.stats import zscore
import time
from functools import wraps
from datetime import datetime
import pickle
import os

T = TypeVar('T')


# ============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# ============================================================================

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that implements retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: all exceptions)
        
    Returns:
        Decorated function with retry logic
        
    Requirements: 10.2
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Display retry attempt to user if using Streamlit context
                    if attempt > 0:
                        try:
                            st.info(f"🔄 Retry attempt {attempt}/{max_retries}...")
                        except:
                            # Not in Streamlit context, skip display
                            pass
                    
                    # Call the original function
                    result = func(*args, **kwargs)
                    
                    # Success - clear any retry messages
                    if attempt > 0:
                        try:
                            st.success(f"✅ Succeeded after {attempt} retry attempt(s)")
                        except:
                            pass
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise the exception
                    if attempt == max_retries:
                        break
                    
                    # Wait before retrying (exponential backoff)
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # All retries exhausted, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# PERSISTENT CACHE LAYER
# ============================================================================

def save_cache_to_file(data: pd.DataFrame, cache_file: str = '.cache_trends.pkl'):
    """Save data to a pickle file for persistent caching."""
    try:
        cache_data = {
            'data': data,
            'timestamp': datetime.now()
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception:
        pass  # Silently fail if can't save cache


def load_cache_from_file(cache_file: str = '.cache_trends.pkl') -> tuple:
    """Load data from pickle file. Returns (data, timestamp) or (None, None)."""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data.get('data'), cache_data.get('timestamp')
    except Exception:
        pass  # Silently fail if can't load cache
    return None, None


# ============================================================================
# INPUT VALIDATION LAYER
# ============================================================================

def validate_api_key(api_key: str) -> bool:
    """
    Validates GIPHY API key format.
    
    Args:
        api_key: API key string to validate
        
    Returns:
        True if valid, False otherwise
        
    Requirements: 10.1
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # GIPHY API keys are typically 32 characters alphanumeric
    if len(api_key) < 10:  # Minimum reasonable length
        return False
    
    # Check if it contains only valid characters (alphanumeric)
    if not api_key.replace('-', '').replace('_', '').isalnum():
        return False
    
    return True


def validate_country_code(country_code: str, valid_codes: List[str]) -> bool:
    """
    Validates country code against list of supported codes.
    
    Args:
        country_code: Country code to validate (can be empty for worldwide)
        valid_codes: List of valid country codes
        
    Returns:
        True if valid, False otherwise
        
    Requirements: 10.1
    """
    # Empty string is valid (represents worldwide)
    if country_code == "":
        return True
    
    # Check if code is in the valid list
    return country_code in valid_codes


def validate_limit(limit: int, min_val: int = 1, max_val: int = 100) -> bool:
    """
    Validates limit parameter for API calls.
    
    Args:
        limit: Limit value to validate
        min_val: Minimum allowed value (default: 1)
        max_val: Maximum allowed value (default: 100)
        
    Returns:
        True if valid, False otherwise
        
    Requirements: 10.1
    """
    if not isinstance(limit, int):
        return False
    
    return min_val <= limit <= max_val


# ============================================================================
# DATA FETCHING LAYER
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes (300 seconds)
@retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    exceptions=(requests.RequestException,)
)
def fetch_giphy_trending(api_key: str, limit: int = 50) -> Dict[str, Any]:
    """
    Fetches trending GIFs from GIPHY API with automatic retry on failure.
    
    Args:
        api_key: GIPHY API key
        limit: Number of trending GIFs to fetch (default: 50)
        
    Returns:
        dict with 'data' key containing list of GIF objects
        
    Raises:
        requests.RequestException: If API request fails after all retries
        
    Requirements: 1.1, 1.4, 10.2
    """
    url = "https://api.giphy.com/v1/gifs/trending"
    params = {
        "api_key": api_key,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch GIPHY trending data: {str(e)}")


@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds) to avoid rate limits
@retry_with_exponential_backoff(
    max_retries=2,  # Reduce retries to avoid hitting rate limits
    initial_delay=2.0,  # Increase initial delay
    backoff_factor=3.0,  # Increase backoff factor
    exceptions=(Exception,)
)
def fetch_flu_trends(keywords: List[str] = None, timeframe: str = 'today 12-m', geo: str = '') -> pd.DataFrame:
    """
    Fetches Google Trends search interest for flu-related terms with automatic retry on failure.
    Search interest is treated as a proxy for flu activity.
    Aggregation and alignment rules ensure compatibility with MVI.
    
    Args:
        keywords: List of search keywords (default: ['flu', 'fever', 'influenza'])
        timeframe: Time range for data (default: 'today 12-m' for past 12 months)
        geo: Country code (empty string for worldwide)
        
    Returns:
        DataFrame with datetime index and keyword columns containing search interest scores
        
    Raises:
        Exception: If PyTrends request fails or rate limit is hit after all retries
        
    Requirements: 2.1, 2.2, 2.4, 2.5, 10.2
    """
    if keywords is None:
        keywords = ['flu', 'fever', 'influenza']
    
    try:
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        # Initialize PyTrends
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Build payload
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
        
        # Get interest over time
        trends_df = pytrends.interest_over_time()
        
        # Remove 'isPartial' column if present
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df.drop(columns=['isPartial'])
        
        return trends_df
        
    except Exception as e:
        raise Exception(f"Failed to fetch Google Trends data: {str(e)}")


# ============================================================================
# ERROR MESSAGE DISPLAY
# ============================================================================

def display_error_message(error_type: str, error_details: str = "", show_retry: bool = False) -> None:
    """
    Displays user-friendly error messages for different error types.
    
    Args:
        error_type: Type of error ('api_failure', 'data_processing', 'insufficient_data', 'general')
        error_details: Additional details about the error (optional)
        show_retry: Whether to show retry information (default: False)
        
    Requirements: 10.1
    """
    error_messages = {
        'api_failure': {
            'title': '🔌 API Connection Error',
            'message': 'Unable to fetch data from the external API.',
            'suggestion': 'This could be due to network issues or API rate limits. The system will automatically retry.'
        },
        'giphy_failure': {
            'title': '🎬 GIPHY API Error',
            'message': 'Unable to fetch trending meme data from GIPHY.',
            'suggestion': 'Please check your API key configuration or try again later.'
        },
        'trends_failure': {
            'title': '📊 Google Trends Error',
            'message': 'Unable to fetch flu search trend data from Google Trends.',
            'suggestion': 'Google Trends may be temporarily unavailable. Showing cached data if available.'
        },
        'data_processing': {
            'title': '⚙️ Data Processing Error',
            'message': 'An error occurred while processing the data.',
            'suggestion': 'Some features may be unavailable, but other components will continue to work.'
        },
        'insufficient_data': {
            'title': '📉 Insufficient Data',
            'message': 'Not enough data available for this operation.',
            'suggestion': 'Some features require more historical data to function properly.'
        },
        'general': {
            'title': '⚠️ Error',
            'message': 'An unexpected error occurred.',
            'suggestion': 'Please try refreshing the page or contact support if the issue persists.'
        }
    }
    
    # Get error message template
    error_info = error_messages.get(error_type, error_messages['general'])
    
    # Build error message
    message_parts = [f"**{error_info['title']}**"]
    message_parts.append(f"\n{error_info['message']}")
    
    if error_details:
        message_parts.append(f"\n\n*Details:* {error_details}")
    
    message_parts.append(f"\n\n💡 {error_info['suggestion']}")
    
    if show_retry:
        message_parts.append("\n\n🔄 Retrying automatically...")
    
    # Display error message
    st.error('\n'.join(message_parts))


# ============================================================================
# METRIC COMPUTATION LAYER
# ============================================================================

def compute_mvi(giphy_data: Dict[str, Any]) -> int:
    """
    Computes the Meme Virality Index (MVI) as the total count of trending memes.
    
    Args:
        giphy_data: Dictionary returned from GIPHY API with 'data' key containing list of GIF objects
        
    Returns:
        Integer count of trending memes (MVI value)
    """
    # Handle empty or invalid data gracefully
    if not giphy_data:
        return 0
    
    if 'data' not in giphy_data:
        return 0
    
    data = giphy_data['data']
    
    # Handle None or non-list data
    if data is None or not isinstance(data, list):
        return 0
    
    return len(data)


def compute_fsi(trends_df: pd.DataFrame) -> float:
    """
    Computes the Flu Search Index (FSI) as the weekly average of flu keyword scores.
    
    Args:
        trends_df: DataFrame with datetime index and keyword columns (flu, fever, influenza)
                   containing search interest scores from Google Trends
        
    Returns:
        Float representing the weekly average FSI value
    """
    # Handle empty or invalid DataFrame
    if trends_df is None or trends_df.empty:
        return 0.0
    
    # Resample to weekly frequency (Monday-based weeks) and compute mean
    weekly_df = trends_df.resample('W-MON').mean()
    
    # Compute FSI as the average of all keyword columns for each week
    # Then take the overall mean across all weeks
    fsi = weekly_df.mean().mean()
    
    return float(fsi)


def compute_correlation(mvi_series: pd.Series, fsi_series: pd.Series) -> float:
    """
    Computes the Pearson correlation coefficient between MVI and FSI time series.
    
    Args:
        mvi_series: Time series of MVI values with datetime index
        fsi_series: Time series of FSI values with datetime index
        
    Returns:
        Float representing Pearson correlation coefficient between -1 and 1
        Returns 0.0 if insufficient data or computation fails
    """
    # Handle empty or invalid series
    if mvi_series is None or fsi_series is None:
        return 0.0
    
    if len(mvi_series) == 0 or len(fsi_series) == 0:
        return 0.0
    
    # Need at least 2 data points for correlation
    if len(mvi_series) < 2 or len(fsi_series) < 2:
        return 0.0
    
    try:
        # Align the series by their indices (inner join)
        aligned_mvi, aligned_fsi = mvi_series.align(fsi_series, join='inner')
        
        # Check if we have enough aligned data points
        if len(aligned_mvi) < 2:
            return 0.0
        
        # Compute Pearson correlation
        correlation = aligned_mvi.corr(aligned_fsi)
        
        # Handle NaN result (e.g., if one series has no variance)
        if pd.isna(correlation):
            return 0.0
        
        return float(correlation)
        
    except Exception:
        # Return 0.0 for any computation errors
        return 0.0


# ============================================================================
# SENTIMENT ANALYSIS LAYER
# ============================================================================

def analyze_sentiment(text: str) -> float:
    """
    Analyzes sentiment of text using TextBlob.
    
    Args:
        text: Input text (meme title)
        
    Returns:
        Sentiment score normalized to [-1, 1] range
        Returns 0.0 if text is empty, None, or analysis fails
    """
    # Handle empty or invalid text
    if not text or not isinstance(text, str):
        return 0.0
    
    # Handle empty string after stripping whitespace
    text = text.strip()
    if not text:
        return 0.0
    
    try:
        # Analyze sentiment using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        # TextBlob polarity is already in [-1, 1] range
        # Ensure it's within bounds (defensive programming)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return float(sentiment_score)
        
    except Exception:
        # Handle any errors gracefully (e.g., problematic text)
        return 0.0


def compute_weekly_sentiment(meme_data: List[Dict[str, Any]], date_key: str = 'trending_datetime') -> pd.Series:
    """
    Aggregates sentiment scores by week.
    
    Args:
        meme_data: List of meme dictionaries containing title and date information
        date_key: Key name for the date field in meme dictionaries (default: 'trending_datetime')
        
    Returns:
        Series with weekly average sentiment values, indexed by week start date
        Returns empty Series if no valid data
    """
    # Handle empty or invalid input
    if not meme_data or not isinstance(meme_data, list):
        return pd.Series(dtype=float)
    
    # Build list of (date, sentiment) tuples
    sentiment_records = []
    
    for meme in meme_data:
        if not isinstance(meme, dict):
            continue
        
        # Get title and date
        title = meme.get('title', '')
        date_str = meme.get(date_key, '')
        
        if not title or not date_str:
            continue
        
        try:
            # Parse date
            date = pd.to_datetime(date_str)
            
            # Compute sentiment
            sentiment = analyze_sentiment(title)
            
            sentiment_records.append({
                'date': date,
                'sentiment': sentiment
            })
            
        except Exception:
            # Skip records with problematic dates or sentiment computation
            continue
    
    # Handle case where no valid records were created
    if not sentiment_records:
        return pd.Series(dtype=float)
    
    # Create DataFrame
    df = pd.DataFrame(sentiment_records)
    df.set_index('date', inplace=True)
    
    # Resample to weekly frequency (Monday-based weeks) and compute mean
    weekly_sentiment = df['sentiment'].resample('W-MON').mean()
    
    return weekly_sentiment


# ============================================================================
# ANOMALY DETECTION LAYER
# ============================================================================

def detect_anomalies(mvi_series: pd.Series, threshold: float = 2.5) -> List[int]:
    """
    Detects anomalies in MVI time series using z-score method.
    
    Args:
        mvi_series: Time series of MVI values
        threshold: Z-score threshold for anomaly detection (default: 2.5)
        
    Returns:
        List of indices where anomalies were detected (z > threshold)
    """
    # Handle empty or invalid series
    if mvi_series is None or len(mvi_series) == 0:
        return []
    
    # Need at least 2 data points to compute standard deviation
    if len(mvi_series) < 2:
        return []
    
    try:
        # Compute z-scores
        mean = mvi_series.mean()
        std = mvi_series.std()
        
        # Handle case where std is 0 (all values are the same)
        if std == 0:
            return []
        
        # Calculate z-scores for each value
        z_scores = (mvi_series - mean) / std
        
        # Find indices where absolute z-score exceeds threshold
        anomaly_indices = []
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > threshold:
                anomaly_indices.append(i)
        
        return anomaly_indices
        
    except Exception:
        # Handle any computation errors gracefully
        return []


# ============================================================================
# PREDICTION MODEL LAYER
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds)
def train_predictor(fsi_history: pd.Series, mvi_history: pd.Series, 
                   sentiment_history: pd.Series) -> tuple:
    """
    Trains prediction model on historical FSI data using RandomForestRegressor.
    
    Args:
        fsi_history: Weekly FSI values (minimum 52 weeks required)
        mvi_history: Weekly MVI values
        sentiment_history: Weekly sentiment scores
    
    Returns:
        Tuple of (trained_model, predicted_next_week_fsi)
        
    Raises:
        ValueError: If insufficient data (< 52 weeks)
    """
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Validate input data
    if fsi_history is None or len(fsi_history) < 5:
        raise ValueError("Insufficient data: At least 5 weeks of FSI history required for prediction")
    
    if mvi_history is None or len(mvi_history) == 0:
        raise ValueError("MVI history is required for prediction")
    
    if sentiment_history is None or len(sentiment_history) == 0:
        raise ValueError("Sentiment history is required for prediction")
    
    # Align all series by their indices (inner join to ensure matching dates)
    aligned_data = pd.DataFrame({
        'fsi': fsi_history,
        'mvi': mvi_history,
        'sentiment': sentiment_history
    }).dropna()
    
    # Check if we still have enough data after alignment
    if len(aligned_data) < 5:
        raise ValueError(f"Insufficient aligned data: {len(aligned_data)} weeks available, 5 required")
    
    # Create feature matrix with lagged features
    features_list = []
    targets_list = []
    
    # We need at least 4 weeks of lag features, so start from index 4
    for i in range(4, len(aligned_data)):
        # Extract lagged features
        fsi_t1 = aligned_data['fsi'].iloc[i-1]
        fsi_t2 = aligned_data['fsi'].iloc[i-2]
        fsi_t3 = aligned_data['fsi'].iloc[i-3]
        fsi_t4 = aligned_data['fsi'].iloc[i-4]
        
        mvi_t1 = aligned_data['mvi'].iloc[i-1]
        mvi_t2 = aligned_data['mvi'].iloc[i-2]
        
        sentiment_avg = aligned_data['sentiment'].iloc[i-1]
        
        # Extract month for seasonality (1-12)
        month = aligned_data.index[i].month
        
        # Create feature vector
        features = [fsi_t1, fsi_t2, fsi_t3, fsi_t4, mvi_t1, mvi_t2, sentiment_avg, month]
        features_list.append(features)
        
        # Target is current FSI value
        targets_list.append(aligned_data['fsi'].iloc[i])
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(targets_list)
    
    # Check if we have enough training samples
    if len(X) < 1:
        raise ValueError(f"Insufficient training samples: {len(X)} samples available, at least 1 required")
    
    # Train RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Predict next week's FSI
    # Use the most recent data to create prediction features
    last_idx = len(aligned_data) - 1
    
    fsi_t1 = aligned_data['fsi'].iloc[last_idx]
    fsi_t2 = aligned_data['fsi'].iloc[last_idx - 1]
    fsi_t3 = aligned_data['fsi'].iloc[last_idx - 2]
    fsi_t4 = aligned_data['fsi'].iloc[last_idx - 3]
    
    mvi_t1 = aligned_data['mvi'].iloc[last_idx]
    mvi_t2 = aligned_data['mvi'].iloc[last_idx - 1]
    
    sentiment_avg = aligned_data['sentiment'].iloc[last_idx]
    
    # Predict next month (add 1 week to last date)
    next_date = aligned_data.index[last_idx] + pd.Timedelta(weeks=1)
    next_month = next_date.month
    
    # Create prediction feature vector
    prediction_features = np.array([[fsi_t1, fsi_t2, fsi_t3, fsi_t4, mvi_t1, mvi_t2, sentiment_avg, next_month]])
    
    # Make prediction
    predicted_fsi = model.predict(prediction_features)[0]
    
    return (model, float(predicted_fsi))


def get_feature_importance(model) -> Dict[str, float]:
    """
    Extracts feature importance scores from a trained RandomForestRegressor model.
    
    Args:
        model: Trained RandomForestRegressor model
        
    Returns:
        Dictionary mapping feature names to their importance scores
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Validate model type
    if not isinstance(model, RandomForestRegressor):
        raise TypeError("Model must be a RandomForestRegressor instance")
    
    # Check if model has been fitted
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model has not been fitted yet")
    
    # Feature names matching the order used in train_predictor
    feature_names = [
        'FSI_lag_1',
        'FSI_lag_2',
        'FSI_lag_3',
        'FSI_lag_4',
        'MVI_lag_1',
        'MVI_lag_2',
        'Sentiment_avg',
        'Month'
    ]
    
    # Extract feature importances
    importances = model.feature_importances_
    
    # Create dictionary mapping feature names to importance scores
    feature_importance_dict = {
        name: float(importance) 
        for name, importance in zip(feature_names, importances)
    }
    
    return feature_importance_dict


# ============================================================================
# UI STYLING LAYER
# ============================================================================

def inject_custom_css() -> None:
    """
    Injects custom CSS for neo-glassmorphism styling using st.markdown().
    Applies consistent glassmorphism effects across all UI elements including:
    - Semi-transparent backgrounds with blur
    - Cool-toned color palette (blues, purples, violet)
    - Rounded corners and soft shadows
    - Glow effects
    """
    css = """
    <style>
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Glassmorphism card base style */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* KPI Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(118, 75, 162, 0.5);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.2);
        padding: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0px 8px 40px rgba(118, 75, 162, 0.4);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(118, 75, 162, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(118, 75, 162, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0px 6px 25px rgba(118, 75, 162, 0.4);
        transform: translateY(-2px);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
    
    /* Charts */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Meme grid cards */
    .meme-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.2);
        padding: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
    }
    
    .meme-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 8px 40px rgba(118, 75, 162, 0.5);
    }
    
    .meme-card img {
        border-radius: 15px;
        width: 100%;
        height: auto;
    }
    
    .meme-title {
        color: #ffffff;
        font-weight: 600;
        margin-top: 0.5rem;
        text-shadow: 0 0 10px rgba(118, 75, 162, 0.3);
    }
    
    .sentiment-score {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* Glow effects for emphasis */
    .glow-text {
        text-shadow: 0 0 20px rgba(118, 75, 162, 0.8),
                     0 0 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Anomaly highlight */
    .anomaly-alert {
        background: rgba(255, 100, 100, 0.2);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 2px solid rgba(255, 100, 100, 0.5);
        padding: 1rem;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(255, 100, 100, 0.4);
    }
    
    /* Normal activity indicator */
    .normal-activity {
        background: rgba(100, 255, 150, 0.2);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 2px solid rgba(100, 255, 150, 0.5);
        padding: 1rem;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(100, 255, 150, 0.4);
    }
    
    /* Text colors */
    p, span, div {
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Sidebar (if used) */
    section[data-testid="stSidebar"] {
        background: rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(20px);
    }
    
    /* Loading indicator */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# UI RENDERING LAYER
# ============================================================================

def render_trend_chart(mvi_series: pd.Series, fsi_series: pd.Series, anomalies: List[int] = None) -> None:
    """
    Renders a line chart comparing weekly MVI and FSI trends over time.
    Uses Plotly for interactive visualization with normalized scales.
    Highlights anomalies if provided.
    
    Args:
        mvi_series: Time series of weekly MVI values with datetime index
        fsi_series: Time series of weekly FSI values with datetime index
        anomalies: Optional list of indices where anomalies were detected
        
    Requirements: 4.1, 4.3, 7.2
    """
    import plotly.graph_objects as go
    from scipy.stats import zscore
    
    # Handle empty or invalid series
    if mvi_series is None or fsi_series is None:
        st.warning("No trend data available to display.")
        return
    
    if len(mvi_series) == 0 or len(fsi_series) == 0:
        st.warning("Insufficient trend data to display.")
        return
    
    # Align series by their indices (inner join) - Requirement 4.3
    aligned_mvi, aligned_fsi = mvi_series.align(fsi_series, join='inner')
    
    if len(aligned_mvi) == 0:
        st.warning("No overlapping time periods between MVI and FSI data.")
        return
    
    # Normalize both series using z-score for fair comparison
    try:
        mvi_normalized = zscore(aligned_mvi)
        fsi_normalized = zscore(aligned_fsi)
    except Exception:
        # If normalization fails (e.g., constant values), use raw values
        mvi_normalized = aligned_mvi.values
        fsi_normalized = aligned_fsi.values
    
    # Create figure with plotly
    fig = go.Figure()
    
    # Add MVI trace
    fig.add_trace(go.Scatter(
        x=aligned_mvi.index,
        y=mvi_normalized,
        mode='lines+markers',
        name='Meme Virality Index (MVI)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#667eea'),
        hovertemplate='<b>MVI (normalized)</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Add FSI trace
    fig.add_trace(go.Scatter(
        x=aligned_fsi.index,
        y=fsi_normalized,
        mode='lines+markers',
        name='Flu Search Index (FSI)',
        line=dict(color='#764ba2', width=3),
        marker=dict(size=8, color='#764ba2'),
        hovertemplate='<b>FSI (normalized)</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Highlight anomalies if provided (Requirement 7.2)
    if anomalies and len(anomalies) > 0:
        # Filter valid anomaly indices
        valid_anomalies = [i for i in anomalies if 0 <= i < len(aligned_mvi)]
        
        if valid_anomalies:
            anomaly_dates = [aligned_mvi.index[i] for i in valid_anomalies]
            anomaly_values = [mvi_normalized[i] for i in valid_anomalies]
            
            # Add anomaly markers
            fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=15,
                    color='rgba(255, 100, 100, 0.8)',
                    symbol='star',
                    line=dict(color='red', width=2)
                ),
                hovertemplate='<b>Anomaly Detected</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
    
    # Update layout with glassmorphism styling
    fig.update_layout(
        title={
            'text': 'Meme vs Flu Trend Comparison (Normalized)',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Date',
        yaxis_title='Normalized Value (Z-Score)',
        hovermode='x unified',
        plot_bgcolor='rgba(255, 255, 255, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white', size=12),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white'
        ),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(color='white')
        ),
        height=500
    )
    
    # Display the chart
    st.plotly_chart(fig, width="stretch")


def render_sentiment_chart(sentiment_series: pd.Series) -> None:
    """
    Renders a line chart showing weekly sentiment trend over time.
    Uses Plotly for interactive visualization with consistent styling.
    
    Args:
        sentiment_series: Time series of weekly sentiment scores with datetime index
        
    Requirements: 4.2, 5.4
    """
    import plotly.graph_objects as go
    
    # Handle empty or invalid series
    if sentiment_series is None or len(sentiment_series) == 0:
        st.warning("No sentiment data available to display.")
        return
    
    # Create figure with plotly
    fig = go.Figure()
    
    # Add sentiment trace
    fig.add_trace(go.Scatter(
        x=sentiment_series.index,
        y=sentiment_series.values,
        mode='lines+markers',
        name='Weekly Sentiment',
        line=dict(color='#64ffaa', width=3),
        marker=dict(size=8, color='#64ffaa'),
        fill='tozeroy',
        fillcolor='rgba(100, 255, 170, 0.2)',
        hovertemplate='<b>Sentiment</b><br>Date: %{x}<br>Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.3)",
        annotation_text="Neutral",
        annotation_position="right"
    )
    
    # Update layout with glassmorphism styling
    fig.update_layout(
        title={
            'text': 'Weekly Sentiment Trend',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        hovermode='x unified',
        plot_bgcolor='rgba(255, 255, 255, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white', size=12),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white',
            range=[-1.1, 1.1]  # Sentiment is in [-1, 1] range
        ),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(color='white')
        ),
        height=400
    )
    
    # Display the chart
    st.plotly_chart(fig, width="stretch")


def render_kpi_cards(mvi: int, fsi: float, correlation: float, sentiment: float) -> None:
    """
    Renders KPI metric cards displaying MVI, FSI, correlation, and sentiment.
    Uses Streamlit columns for layout with glassmorphism styling applied via CSS.
    
    Args:
        mvi: Meme Virality Index value (integer count of trending memes)
        fsi: Flu Search Index value (float representing weekly average)
        correlation: Pearson correlation coefficient between MVI and FSI (-1 to 1)
        sentiment: Current sentiment score (-1 to 1)
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    # Create 4 columns for the KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    # MVI Card (Requirement 3.1)
    with col1:
        st.metric(
            label="🔥 Meme Virality Index",
            value=f"{mvi:,}",
            help="Total count of trending memes from GIPHY"
        )
    
    # FSI Card (Requirement 3.2)
    with col2:
        st.metric(
            label="🦠 Flu Search Index",
            value=f"{fsi:.2f}",
            help="Weekly average of flu-related search interest"
        )
    
    # Correlation Card (Requirement 3.3)
    with col3:
        # Format correlation with sign and color indication
        correlation_display = f"{correlation:+.3f}"
        
        st.metric(
            label="📊 Correlation",
            value=correlation_display,
            help="Pearson correlation between MVI and FSI (-1 to +1)"
        )
    
    # Sentiment Card (Requirement 3.4)
    with col4:
        # Format sentiment with appropriate emoji
        if sentiment > 0.1:
            sentiment_emoji = "😊"
        elif sentiment < -0.1:
            sentiment_emoji = "😔"
        else:
            sentiment_emoji = "😐"
        
        st.metric(
            label=f"{sentiment_emoji} Sentiment Score",
            value=f"{sentiment:+.3f}",
            help="Average sentiment of trending meme titles (-1 to +1)"
        )


def render_header() -> None:
    """
    Renders the dashboard header with title and subtitle.
    Applies glassmorphism styling for a polished, modern appearance.
    
    Requirements: All (UI requirement)
    """
    # Create a custom header with glassmorphism styling
    header_html = """
    <div style="
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.2);
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="
            color: #ffffff;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 0 20px rgba(118, 75, 162, 0.8);
            letter-spacing: 2px;
        ">
            🦠 MemePulse vs FluWave
        </h1>
        <p style="
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            font-weight: 400;
            margin: 1rem 0 0 0;
            text-shadow: 0 0 10px rgba(102, 126, 234, 0.6);
        ">
            Real-Time Cultural vs Health Trend Analyzer
        </p>
        <p style="
            color: rgba(255, 255, 255, 0.7);
            font-size: 1rem;
            margin: 0.5rem 0 0 0;
        ">
            Comparing meme virality with flu search trends to uncover hidden patterns
        </p>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)


def render_meme_grid(memes: List[Dict[str, Any]]) -> None:
    """
    Renders a grid of the top 10 trending memes with thumbnails, titles, and sentiment scores.
    Uses Streamlit columns for grid layout with glassmorphism styling applied via CSS.
    
    Args:
        memes: List of meme dictionaries containing id, title, images, and other metadata
               Expected structure from GIPHY API response
    
    Requirements: 1.3, 9.3
    """
    # Handle empty or invalid input
    if not memes or not isinstance(memes, list):
        st.warning("No meme data available to display.")
        return
    
    # Limit to top 10 memes (Requirement 1.3)
    top_memes = memes[:10]
    
    if len(top_memes) == 0:
        st.warning("No memes to display.")
        return
    
    # Display memes in a grid layout (2 columns per row for better visibility)
    # This creates a 2x5 grid for 10 memes
    for i in range(0, len(top_memes), 2):
        cols = st.columns(2)
        
        for col_idx, col in enumerate(cols):
            meme_idx = i + col_idx
            
            # Check if we have a meme for this position
            if meme_idx >= len(top_memes):
                break
            
            meme = top_memes[meme_idx]
            
            with col:
                # Create a container with glassmorphism styling
                # Use HTML/CSS for custom card styling
                
                # Extract meme data
                title = meme.get('title', 'Untitled Meme')
                
                # Get thumbnail URL from images object
                # GIPHY API structure: images -> fixed_height -> url (or fixed_width)
                thumbnail_url = None
                if 'images' in meme and isinstance(meme['images'], dict):
                    # Try fixed_height first, then fixed_width, then original
                    if 'fixed_height' in meme['images']:
                        thumbnail_url = meme['images']['fixed_height'].get('url')
                    elif 'fixed_width' in meme['images']:
                        thumbnail_url = meme['images']['fixed_width'].get('url')
                    elif 'original' in meme['images']:
                        thumbnail_url = meme['images']['original'].get('url')
                
                # Compute sentiment score for the title
                sentiment_score = analyze_sentiment(title)
                
                # Format sentiment with emoji
                if sentiment_score > 0.1:
                    sentiment_emoji = "😊"
                    sentiment_color = "#64ffaa"
                elif sentiment_score < -0.1:
                    sentiment_emoji = "😔"
                    sentiment_color = "#ff6464"
                else:
                    sentiment_emoji = "😐"
                    sentiment_color = "#ffffff"
                
                # Create meme card with custom HTML for glassmorphism styling
                card_html = f"""
                <div class="meme-card">
                    {"<img src='" + thumbnail_url + "' alt='Meme thumbnail' />" if thumbnail_url else "<div style='height: 200px; background: rgba(255,255,255,0.1); border-radius: 15px; display: flex; align-items: center; justify-content: center;'><span style='color: rgba(255,255,255,0.5);'>No Image</span></div>"}
                    <div class="meme-title">{title[:100]}{"..." if len(title) > 100 else ""}</div>
                    <div class="sentiment-score" style="color: {sentiment_color};">
                        {sentiment_emoji} Sentiment: {sentiment_score:+.3f}
                    </div>
                </div>
                """
                
                st.markdown(card_html, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="MemePulse vs FluWave Dashboard",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS for neo-glassmorphism styling
    inject_custom_css()
    
    # Render header with glassmorphism styling
    render_header()
    
    # Add refresh button (Requirement 4.5)
    col1, col2, col3 = st.columns([6, 1, 1])
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and fetch fresh data from APIs"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        if st.button("ℹ️ About", help="About this dashboard"):
            st.info("""
            **MemePulse vs FluWave Dashboard**
            
            This dashboard compares real-time meme activity from GIPHY with flu-related search trends from Google Trends.
            
            **Features:**
            - Real-time meme virality tracking
            - Flu search trend analysis
            - Sentiment analysis of meme titles
            - Correlation analysis between cultural and health trends
            - Anomaly detection for unusual viral events
            - Predictive modeling for flu search trends
            
            **Data Sources:**
            - GIPHY Trending API
            - Google Trends (via PyTrends)
            
            **Refresh:** Data is cached for performance. Click "Refresh Data" to fetch the latest information.
            """)
    
    st.markdown("---")
    
    # Initialize session state for data caching if not exists (Requirement 10.5)
    if 'giphy_data' not in st.session_state:
        st.session_state.giphy_data = None
    if 'trends_data' not in st.session_state:
        st.session_state.trends_data = None
    if 'mvi' not in st.session_state:
        st.session_state.mvi = 0
    if 'fsi' not in st.session_state:
        st.session_state.fsi = 0.0
    if 'correlation' not in st.session_state:
        st.session_state.correlation = 0.0
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = 0.0
    if 'weekly_sentiment' not in st.session_state:
        st.session_state.weekly_sentiment = pd.Series(dtype=float)
    if 'anomalies' not in st.session_state:
        st.session_state.anomalies = []
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = {}
    if 'prediction_model' not in st.session_state:
        st.session_state.prediction_model = None
    
    # Cache timestamps for fallback UI (Requirement 10.5)
    if 'giphy_data_timestamp' not in st.session_state:
        st.session_state.giphy_data_timestamp = None
    if 'trends_data_timestamp' not in st.session_state:
        st.session_state.trends_data_timestamp = None
    
    # Country selection for Google Trends (Requirement 2.4)
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = ''  # Default to worldwide
    
    # Fetch data (with error handling, isolation, and caching - Requirements 10.4, 10.5)
    try:
        # Get GIPHY API key from secrets
        if 'GIPHY' in st.secrets and 'API_KEY' in st.secrets['GIPHY']:
            api_key = st.secrets['GIPHY']['API_KEY']
            
            # Validate API key (Requirement 10.1)
            if not validate_api_key(api_key):
                display_error_message('giphy_failure', 'Invalid API key format')
                if st.session_state.giphy_data is None:
                    st.session_state.giphy_data = {'data': []}
                    st.session_state.mvi = 0
            else:
                # Validate limit parameter
                limit = 50
                if not validate_limit(limit, min_val=1, max_val=100):
                    limit = 50  # Use default if invalid
                
                with st.spinner("Fetching trending memes from GIPHY..."):
                    new_giphy_data = fetch_giphy_trending(api_key, limit=limit)
                    st.session_state.giphy_data = new_giphy_data
                    st.session_state.giphy_data_timestamp = datetime.now()
                    st.session_state.mvi = compute_mvi(st.session_state.giphy_data)
        else:
            display_error_message('giphy_failure', 'API key not found in secrets configuration')
            # Use cached data if available (Requirement 10.5)
            if st.session_state.giphy_data is None:
                st.session_state.giphy_data = {'data': []}
                st.session_state.mvi = 0
    except requests.RequestException as e:
        display_error_message('giphy_failure', str(e))
        # Use cached data if available (Requirement 10.5)
        if st.session_state.giphy_data is None:
            st.session_state.giphy_data = {'data': []}
            st.session_state.mvi = 0
        else:
            # Show cached data timestamp
            if st.session_state.giphy_data_timestamp:
                st.info(f"📦 Showing cached GIPHY data from {st.session_state.giphy_data_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        display_error_message('data_processing', f"GIPHY data processing error: {str(e)}")
        # Use cached data if available (Requirement 10.5)
        if st.session_state.giphy_data is None:
            st.session_state.giphy_data = {'data': []}
            st.session_state.mvi = 0
        else:
            # Show cached data timestamp
            if st.session_state.giphy_data_timestamp:
                st.info(f"📦 Showing cached GIPHY data from {st.session_state.giphy_data_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Always try to load from persistent file cache first
    cached_data, cached_timestamp = load_cache_from_file()
    if cached_data is not None and not cached_data.empty:
        st.session_state.trends_data = cached_data
        st.session_state.trends_data_timestamp = cached_timestamp
        st.session_state.fsi = compute_fsi(st.session_state.trends_data)
    
    # Try to fetch fresh data from Google Trends
    try:
        with st.spinner("Fetching flu trends from Google Trends..."):
            new_trends_data = fetch_flu_trends(geo=st.session_state.selected_country)
            st.session_state.trends_data = new_trends_data
            st.session_state.trends_data_timestamp = datetime.now()
            st.session_state.fsi = compute_fsi(st.session_state.trends_data)
            # Save to persistent cache for future use
            save_cache_to_file(new_trends_data)
    except Exception as e:
        # Silently fall back to cached data if available
        # Only show warning if we have no cached data at all
        if st.session_state.trends_data is None or st.session_state.trends_data.empty:
            st.session_state.trends_data = pd.DataFrame()
            st.session_state.fsi = 0.0
        else:
            # Show info that we're using cached data
            if st.session_state.trends_data_timestamp:
                st.info(f"📦 Using cached flu trend data from {st.session_state.trends_data_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Compute metrics if we have data (with error isolation - Requirement 10.4)
    if st.session_state.giphy_data and st.session_state.giphy_data.get('data'):
        try:
            with st.spinner("Analyzing sentiment of meme titles..."):
                # Compute weekly sentiment
                st.session_state.weekly_sentiment = compute_weekly_sentiment(
                    st.session_state.giphy_data['data']
                )
                
                # Compute current sentiment (average of all meme titles)
                if st.session_state.weekly_sentiment is not None and len(st.session_state.weekly_sentiment) > 0:
                    st.session_state.sentiment = st.session_state.weekly_sentiment.mean()
                else:
                    st.session_state.sentiment = 0.0
        except Exception as e:
            display_error_message('data_processing', f"Sentiment analysis error: {str(e)}")
            st.session_state.weekly_sentiment = pd.Series(dtype=float)
            st.session_state.sentiment = 0.0
        
        # Detect anomalies in MVI (we need weekly MVI series for this)
        try:
            with st.spinner("Detecting anomalies in meme activity..."):
                meme_data = st.session_state.giphy_data['data']
                if meme_data:
                    # Create DataFrame with dates
                    dates = []
                    for meme in meme_data:
                        date_str = meme.get('trending_datetime', '')
                        if date_str:
                            try:
                                dates.append(pd.to_datetime(date_str))
                            except:
                                pass
                    
                    if dates:
                        # Create weekly MVI series
                        mvi_df = pd.DataFrame({'date': dates, 'count': 1})
                        mvi_df.set_index('date', inplace=True)
                        weekly_mvi = mvi_df.resample('W-MON').sum()['count']
                        
                        # Detect anomalies
                        st.session_state.anomalies = detect_anomalies(weekly_mvi)
        except Exception as e:
            display_error_message('data_processing', f"Anomaly detection error: {str(e)}")
            st.session_state.anomalies = []
    
    # Compute correlation if we have both MVI and FSI data (with error isolation - Requirement 10.4)
    if not st.session_state.trends_data.empty:
        try:
            with st.spinner("Computing correlation between meme and flu trends..."):
                # Create weekly MVI series for correlation
                meme_data = st.session_state.giphy_data.get('data', [])
                if meme_data:
                    dates = []
                    for meme in meme_data:
                        date_str = meme.get('trending_datetime', '')
                        if date_str:
                            try:
                                dates.append(pd.to_datetime(date_str))
                            except:
                                pass
                    
                    if dates:
                        mvi_df = pd.DataFrame({'date': dates, 'count': 1})
                        mvi_df.set_index('date', inplace=True)
                        weekly_mvi = mvi_df.resample('W-MON').sum()['count']
                        
                        # Compute weekly FSI
                        weekly_fsi = st.session_state.trends_data.resample('W-MON').mean().mean(axis=1)
                        
                        # Compute correlation
                        st.session_state.correlation = compute_correlation(weekly_mvi, weekly_fsi)
        except Exception as e:
            display_error_message('data_processing', f"Correlation computation error: {str(e)}")
            st.session_state.correlation = 0.0
    
    # Try to train prediction model if we have enough data (with error isolation - Requirement 10.4)
    try:
        if not st.session_state.trends_data.empty and st.session_state.giphy_data.get('data'):
            with st.spinner("Training prediction model for flu trends..."):
                # Create weekly series
                meme_data = st.session_state.giphy_data['data']
                dates = []
                for meme in meme_data:
                    date_str = meme.get('trending_datetime', '')
                    if date_str:
                        try:
                            dates.append(pd.to_datetime(date_str))
                        except:
                            pass
                
                if dates:
                    mvi_df = pd.DataFrame({'date': dates, 'count': 1})
                    mvi_df.set_index('date', inplace=True)
                    weekly_mvi = mvi_df.resample('W-MON').sum()['count']
                    
                    weekly_fsi = st.session_state.trends_data.resample('W-MON').mean().mean(axis=1)
                    
                    # Try to train predictor
                    model, prediction = train_predictor(
                        weekly_fsi,
                        weekly_mvi,
                        st.session_state.weekly_sentiment
                    )
                    
                    st.session_state.prediction_model = model
                    st.session_state.prediction = prediction
                    st.session_state.feature_importance = get_feature_importance(model)
    except ValueError as e:
        # Not enough data for prediction - this is expected, not an error
        display_error_message('insufficient_data', str(e))
        st.session_state.prediction = None
        st.session_state.feature_importance = {}
    except Exception as e:
        display_error_message('data_processing', f"Prediction model error: {str(e)}")
        st.session_state.prediction = None
        st.session_state.feature_importance = {}
    
    # Create tabbed navigation (Requirement 8.1)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔥 Meme Data",
        "🦠 Flu Search Trends",
        "💡 Insights"
    ])
    
    # Tab 1: Overview (Requirement 8.2)
    with tab1:
        st.header("Dashboard Overview")
        st.markdown("Key metrics and summary visualizations comparing meme virality with flu search trends.")
        
        # Display KPI cards
        render_kpi_cards(
            st.session_state.mvi,
            st.session_state.fsi,
            st.session_state.correlation,
            st.session_state.sentiment
        )
        
        st.markdown("---")
        
        # Display trend comparison chart (with error isolation - Requirement 10.4)
        if not st.session_state.trends_data.empty and st.session_state.giphy_data.get('data'):
            try:
                # Create weekly series for visualization
                meme_data = st.session_state.giphy_data['data']
                dates = []
                for meme in meme_data:
                    date_str = meme.get('trending_datetime', '')
                    if date_str:
                        try:
                            dates.append(pd.to_datetime(date_str))
                        except:
                            pass
                
                if dates:
                    mvi_df = pd.DataFrame({'date': dates, 'count': 1})
                    mvi_df.set_index('date', inplace=True)
                    weekly_mvi = mvi_df.resample('W-MON').sum()['count']
                    
                    weekly_fsi = st.session_state.trends_data.resample('W-MON').mean().mean(axis=1)
                    
                    render_trend_chart(weekly_mvi, weekly_fsi, st.session_state.anomalies)
            except Exception as e:
                display_error_message('data_processing', f"Chart rendering error: {str(e)}")
        else:
            st.info("Insufficient data to display trend comparison chart.")
        
        # Display sentiment chart (with error isolation - Requirement 10.4)
        if st.session_state.weekly_sentiment is not None and len(st.session_state.weekly_sentiment) > 0:
            st.markdown("---")
            try:
                render_sentiment_chart(st.session_state.weekly_sentiment)
            except Exception as e:
                display_error_message('data_processing', f"Sentiment chart rendering error: {str(e)}")
        else:
            st.info("No sentiment data available.")
    
    # Tab 2: Meme Data (Requirement 8.3)
    with tab2:
        st.header("Trending Meme Data")
        st.markdown("Explore the top trending memes with sentiment analysis.")
        
        # Display meme grid (with error isolation - Requirement 10.4)
        if st.session_state.giphy_data and st.session_state.giphy_data.get('data'):
            try:
                render_meme_grid(st.session_state.giphy_data['data'])
            except Exception as e:
                display_error_message('data_processing', f"Meme grid rendering error: {str(e)}")
        else:
            st.warning("No meme data available to display.")
    
    # Tab 3: Flu Search Trends (Requirement 8.4)
    with tab3:
        st.header("Flu Search Trends")
        st.markdown("Google Trends data for flu-related search trends.")
        
        # Country selector (Requirement 2.4)
        st.subheader("🌍 Geographic Selection")
        
        # Define country options with codes
        country_options = {
            "Worldwide": "",
            "United States": "US",
            "United Kingdom": "GB",
            "Canada": "CA",
            "Australia": "AU",
            "Germany": "DE",
            "France": "FR",
            "Italy": "IT",
            "Spain": "ES",
            "Japan": "JP",
            "South Korea": "KR",
            "India": "IN",
            "Brazil": "BR",
            "Mexico": "MX",
            "Netherlands": "NL",
            "Sweden": "SE",
            "Norway": "NO",
            "Denmark": "DK",
            "Finland": "FI",
            "Poland": "PL",
            "Switzerland": "CH",
            "Austria": "AT",
            "Belgium": "BE",
            "Ireland": "IE",
            "Portugal": "PT",
            "Greece": "GR",
            "Czech Republic": "CZ",
            "New Zealand": "NZ",
            "Singapore": "SG",
            "Hong Kong": "HK",
            "Taiwan": "TW"
        }
        
        # Get current country name from code
        current_country_name = "Worldwide"
        for name, code in country_options.items():
            if code == st.session_state.selected_country:
                current_country_name = name
                break
        
        # Create selectbox for country selection
        selected_country_name = st.selectbox(
            "Select a country or region:",
            options=list(country_options.keys()),
            index=list(country_options.keys()).index(current_country_name),
            help="Choose a geographic region to view flu search trends. Worldwide shows global data."
        )
        
        # Update session state if country changed
        new_country_code = country_options[selected_country_name]
        
        # Validate country code (Requirement 10.1)
        valid_country_codes = list(country_options.values())
        if validate_country_code(new_country_code, valid_country_codes):
            if new_country_code != st.session_state.selected_country:
                st.session_state.selected_country = new_country_code
                # Clear cache to force refetch with new country
                st.cache_data.clear()
                st.rerun()
        else:
            st.error(f"Invalid country code: {new_country_code}. Using worldwide data.")
        
        st.markdown("---")
        
        # Display flu trend data (with error isolation - Requirement 10.4)
        if not st.session_state.trends_data.empty:
            st.subheader("Search Interest Over Time")
            
            try:
                # Create a chart for flu trends
                fig = go.Figure()
                
                for column in st.session_state.trends_data.columns:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.trends_data.index,
                        y=st.session_state.trends_data[column],
                        mode='lines+markers',
                        name=column.capitalize(),
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                
                fig.update_layout(
                    title={
                        'text': 'Flu-Related Search Interest',
                        'font': {'size': 20, 'color': 'white'}
                    },
                    xaxis_title='Date',
                    yaxis_title='Search Interest (0-100)',
                    hovermode='x unified',
                    plot_bgcolor='rgba(255, 255, 255, 0.05)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(color='white', size=12),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        color='white'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        color='white'
                    ),
                    legend=dict(
                        bgcolor='rgba(255, 255, 255, 0.1)',
                        bordercolor='rgba(255, 255, 255, 0.2)',
                        borderwidth=1,
                        font=dict(color='white')
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                display_error_message('data_processing', f"Flu trends chart rendering error: {str(e)}")
            
            # Display prediction if available
            st.markdown("---")
            st.subheader("📈 Next Week Prediction")
            
            if st.session_state.prediction is not None:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        label="Predicted FSI (Next Week)",
                        value=f"{st.session_state.prediction:.2f}",
                        help="Predicted Flu Search Index for next week"
                    )
                
                with col2:
                    st.info("💡 This prediction is based on historical FSI, MVI, sentiment data, and seasonal patterns.")
            else:
                st.warning("⚠️ Not enough historical data to generate predictions. At least 52 weeks of data required.")
        else:
            st.warning("No flu trend data available to display.")
    
    # Tab 4: Insights (Requirement 8.5)
    with tab4:
        st.header("Insights & Analysis")
        st.markdown("Deep dive into correlations, anomalies, and model insights.")
        
        # Correlation Analysis
        st.subheader("📊 Correlation Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="Pearson Correlation",
                value=f"{st.session_state.correlation:+.3f}",
                help="Correlation between MVI and FSI (-1 to +1)"
            )
            
            # Interpretation
            if abs(st.session_state.correlation) > 0.7:
                strength = "Strong"
                color = "green"
            elif abs(st.session_state.correlation) > 0.4:
                strength = "Moderate"
                color = "orange"
            else:
                strength = "Weak"
                color = "red"
            
            direction = "positive" if st.session_state.correlation > 0 else "negative"
            
            st.markdown(f"**Interpretation:** <span style='color: {color};'>{strength} {direction} correlation</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **What does this mean?**
            
            - **Positive correlation**: When meme activity increases, flu search interest tends to increase too
            - **Negative correlation**: When meme activity increases, flu search interest tends to decrease
            - **Near zero**: Little to no relationship between meme activity and flu searches
            
            A strong correlation suggests that cultural meme trends and public health interest may be related.
            """)
        
        st.markdown("---")
        
        # Anomaly Detection Results
        st.subheader("🚨 Anomaly Detection")
        
        if st.session_state.anomalies and len(st.session_state.anomalies) > 0:
            st.markdown(
                f"""<div class="anomaly-alert">
                <strong>⚠️ Anomalies Detected!</strong><br>
                Found {len(st.session_state.anomalies)} unusual spike(s) in meme activity.
                These anomalies are highlighted in the trend chart on the Overview tab.
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div class="normal-activity">
                <strong>✅ Normal Activity</strong><br>
                No unusual spikes detected in meme activity. All metrics are within expected ranges.
                </div>""",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Feature Importance (with error isolation - Requirement 10.4)
        st.subheader("🎯 Prediction Model Feature Importance")
        
        if st.session_state.feature_importance:
            st.markdown("**Which factors matter most for predicting flu search trends?**")
            
            try:
                # Create a bar chart for feature importance
                features = list(st.session_state.feature_importance.keys())
                importances = list(st.session_state.feature_importance.values())
                
                # Sort by importance
                sorted_indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
                sorted_features = [features[i] for i in sorted_indices]
                sorted_importances = [importances[i] for i in sorted_indices]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=sorted_importances,
                        y=sorted_features,
                        orientation='h',
                        marker=dict(
                            color=sorted_importances,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Importance")
                        )
                    )
                ])
                
                fig.update_layout(
                    title={
                        'text': 'Feature Importance in Prediction Model',
                        'font': {'size': 18, 'color': 'white'}
                    },
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    plot_bgcolor='rgba(255, 255, 255, 0.05)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(color='white', size=12),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        color='white'
                    ),
                    yaxis=dict(
                        showgrid=False,
                        color='white'
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
                
                st.markdown("""
                **Understanding Feature Importance:**
                - Higher values indicate features that have more influence on predictions
                - Lagged FSI values show how past flu trends affect future trends
                - MVI and sentiment scores reveal the impact of cultural trends on health interest
                """)
            except Exception as e:
                display_error_message('data_processing', f"Feature importance chart rendering error: {str(e)}")
        else:
            st.info("Feature importance data will be available once the prediction model is trained with sufficient data (52+ weeks).")


if __name__ == "__main__":
    main()
