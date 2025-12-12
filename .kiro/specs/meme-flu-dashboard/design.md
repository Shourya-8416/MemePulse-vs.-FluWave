# Design Document: MemePulse vs FluWave Dashboard

## Overview

The MemePulse vs FluWave dashboard is a single-page Streamlit application that integrates real-time data from GIPHY's Trending API and Google Trends to provide comparative analytics between cultural meme trends and public health search patterns. The application features a neo-glassmorphism UI design, real-time data fetching, sentiment analysis, predictive modeling, and anomaly detection capabilities.

The application is structured as a monolithic Streamlit app (`app.py`) that orchestrates data fetching, processing, analysis, and visualization in a cohesive user interface. The design prioritizes simplicity, visual appeal, and real-time responsiveness while maintaining robust error handling.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app.py)              │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Data       │  │  Analysis    │  │     UI       │      │
│  │   Layer      │  │   Layer      │  │   Layer      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ GIPHY API    │  │ Sentiment    │  │ Tabs/Sections│      │
│  │ PyTrends     │  │ Correlation  │  │ KPI Cards    │      │
│  │ Caching      │  │ Prediction   │  │ Charts       │      │
│  │              │  │ Anomaly Det. │  │ Styling      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
┌──────────────────┐                    ┌──────────────────┐
│  External APIs   │                    │   User Browser   │
│  - GIPHY         │                    │                  │
│  - Google Trends │                    │                  │
└──────────────────┘                    └──────────────────┘
```

### Component Breakdown

1. **Data Layer**: Responsible for fetching and caching data from external APIs
2. **Analysis Layer**: Performs computations, sentiment analysis, predictions, and anomaly detection
3. **UI Layer**: Renders the Streamlit interface with custom styling and interactive elements

## Components and Interfaces

### 1. Data Fetching Components

#### GiphyDataFetcher
**Purpose**: Fetch trending meme data from GIPHY API

**Interface**:
```python
def fetch_giphy_trending(api_key: str, limit: int = 50) -> dict:
    """
    Fetches trending GIFs from GIPHY API
    
    Args:
        api_key: GIPHY API key
        limit: Number of trending GIFs to fetch
        
    Returns:
        dict with 'data' key containing list of GIF objects
        
    Raises:
        requests.RequestException: If API request fails
    """
```

#### GoogleTrendsFetcher
**Purpose**: Fetch flu-related search trend data from Google Trends

**Important Note**: Google Trends does not provide actual flu case numbers. Instead, it provides search interest scores (0–100) for flu-related keywords, which serve as a proxy indicator of public flu activity.

**Interface**:
```python
def fetch_flu_trends(keywords: list[str], timeframe: str = 'today 3-m', geo: str = '') -> pd.DataFrame:
    """
    Fetches Google Trends search interest for flu-related terms.
    Search interest is treated as a proxy for flu activity.
    Aggregation and alignment rules ensure compatibility with MVI.
    
    Args:
        keywords: List of search keywords (e.g., ['flu', 'fever', 'influenza'])
        timeframe: Time range for data (default: 'today 3-m' for past 3 months)
        geo: Country code (empty string for worldwide)
        
    Returns:
        DataFrame with datetime index and keyword columns containing search interest scores
        
    Raises:
        Exception: If PyTrends request fails or rate limit is hit
    """
```

### 2. Metric Computation Components

#### MetricsCalculator
**Purpose**: Compute MVI, FSI, and correlation metrics

**Weekly Aggregation Rules (Critical for Correctness)**:

**MVI (Meme Virality Index)**:
- Convert trending_datetime to ISO week
- Count unique GIF IDs per week
- This produces weekly MVI values

**FSI (Flu Search Index)**:
- Resample Google Trends data to weekly frequency: `trends_df.resample("W-MON").mean()`
- Compute FSI as the average of the flu-related keyword columns per week

**Alignment Rule**: Both series must share the same week index before correlation or plotting.

**Normalization Rule**: Before plotting MVI and FSI together or computing correlation:
- Apply z-score normalization to both indices:
  - `MVI_z = zscore(MVI)`
  - `FSI_z = zscore(FSI)`
- This ensures fair comparison across different scales

**Interface**:
```python
def compute_mvi(giphy_data: dict) -> pd.Series:
    """
    Returns weekly count of trending memes
    Converts trending_datetime to ISO week and counts unique GIF IDs
    """
    
def compute_fsi(trends_df: pd.DataFrame) -> pd.Series:
    """
    Returns weekly average of flu keyword scores
    Resamples to weekly frequency and averages keyword columns
    """
    
def normalize_series(series: pd.Series) -> pd.Series:
    """
    Applies z-score normalization
    Returns (series - mean) / std
    """
    
def compute_correlation(mvi_series: pd.Series, fsi_series: pd.Series) -> float:
    """
    Returns Pearson correlation coefficient
    Requires both series to have aligned indices
    """
```

### 3. Sentiment Analysis Component

#### SentimentAnalyzer
**Purpose**: Analyze sentiment of meme titles

**Interface**:
```python
def analyze_sentiment(text: str) -> float:
    """
    Analyzes sentiment of text using TextBlob or VADER
    
    Args:
        text: Input text (meme title)
        
    Returns:
        Sentiment score normalized to [-1, 1] range
    """
    
def compute_weekly_sentiment(meme_data: list, dates: list) -> pd.Series:
    """
    Aggregates sentiment scores by week
    
    Returns:
        Series with weekly sentiment averages
    """
```

### 4. Prediction Component

#### FluPredictor
**Purpose**: Predict next week's FSI using RandomForestRegressor

**Constraints**:
- A prediction model requires historical weekly data
- **Minimum requirement**: At least 52 weeks of FSI history (1 year)
- If fewer weeks exist → skip prediction and show: "Not enough data to generate prediction."

**Feature Set (Mandatory)**:
- Lagged FSI: t-1, t-2, t-3, t-4
- Lagged MVI: t-1, t-2
- Weekly sentiment average
- Month-of-year (seasonality feature)

**Interface**:
```python
def train_predictor(fsi_history: pd.Series, mvi_history: pd.Series, 
                   sentiment_history: pd.Series) -> tuple[RandomForestRegressor, float]:
    """
    Trains prediction model on historical FSI data
    
    Args:
        fsi_history: Weekly FSI values (minimum 52 weeks)
        mvi_history: Weekly MVI values
        sentiment_history: Weekly sentiment scores
    
    Returns:
        Tuple of (trained_model, predicted_next_week_fsi)
        
    Raises:
        ValueError: If insufficient data (< 52 weeks)
    """
    
def get_feature_importance(model: RandomForestRegressor) -> dict:
    """Returns feature importance scores for all features"""
```

### 5. Anomaly Detection Component

#### AnomalyDetector
**Purpose**: Identify unusual spikes in MVI

**Algorithm**: Use z-score method:
- `z = (value - mean) / std`
- If `z > 2.5` → mark as anomaly
- This method is simple, interpretable, and fits real-time dashboards

**Interface**:
```python
def detect_anomalies(mvi_series: pd.Series, threshold: float = 2.5) -> list[int]:
    """
    Detects anomalies using z-score method
    
    Args:
        mvi_series: Time series of MVI values
        threshold: Z-score threshold for anomaly detection (default: 2.5)
        
    Returns:
        List of indices where anomalies were detected (z > threshold)
    """
```

### 6. UI Rendering Components

#### StyleInjector
**Purpose**: Inject custom CSS for neo-glassmorphism styling

**Glassmorphism Parameters (Explicit Values)**:
- Background: `rgba(255, 255, 255, 0.15)`
- Backdrop blur: `blur(20px)`
- Border-radius: `20px`
- Shadow: `0px 4px 30px rgba(0, 0, 0, 0.2)`
- Card border: `1px solid rgba(255, 255, 255, 0.2)`

**Interface**:
```python
def inject_custom_css() -> None:
    """
    Injects CSS using st.markdown() for glassmorphism effects
    Applies consistent styling parameters across all UI elements
    """
```

#### DashboardRenderer
**Purpose**: Render dashboard sections and components

**Interface**:
```python
def render_header() -> None:
    """Renders title and subtitle"""
    
def render_kpi_cards(mvi: int, fsi: float, correlation: float, sentiment: float) -> None:
    """Renders KPI metric cards"""
    
def render_trend_chart(mvi_series: pd.Series, fsi_series: pd.Series) -> None:
    """Renders line chart comparing MVI and FSI"""
    
def render_meme_grid(memes: list) -> None:
    """Renders top 10 memes in grid layout"""
    
def render_tabs() -> None:
    """Renders tabbed navigation interface"""
```

## Data Models

### MemeData
```python
@dataclass
class MemeData:
    id: str
    title: str
    url: str
    thumbnail_url: str
    trending_datetime: str
    sentiment_score: float
```

### TrendData
```python
@dataclass
class TrendData:
    date: datetime
    flu_score: float
    fever_score: float
    influenza_score: float
    fsi: float  # Computed average
```

### DashboardState
```python
@dataclass
class DashboardState:
    mvi: int
    fsi: float
    correlation: float
    sentiment_score: float
    memes: list[MemeData]
    trends: pd.DataFrame
    anomalies: list[int]
    prediction: float
    feature_importance: dict
    last_updated: datetime
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: MVI computation correctness
*For any* list of memes returned from the GIPHY API, the computed MVI should equal the length of that list.
**Validates: Requirements 1.2**

### Property 2: FSI computation correctness
*For any* DataFrame containing flu keyword trend data, the computed FSI should equal the weekly average of the three keyword columns (flu, fever, influenza).
**Validates: Requirements 2.3**

### Property 3: Country-specific trend fetching
*For any* valid country code, the Google Trends fetch function should pass that country code to the PyTrends API as the geo parameter.
**Validates: Requirements 2.4**

### Property 4: Top N meme display
*For any* list of memes with length >= 10, the display function should render exactly 10 memes with their titles and thumbnail URLs present in the output.
**Validates: Requirements 1.3**

### Property 5: Metric rendering completeness
*For any* set of computed metrics (MVI, FSI, correlation, sentiment), all metric values should appear in the rendered KPI card output.
**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 6: Time axis consistency
*For any* two time series (MVI and FSI), before correlation or visualization, the condition `MVI_weekly.index == FSI_weekly.index` must always be true.
**Validates: Requirements 4.3**

### Property 7: Sentiment score computation
*For any* meme title string, the sentiment analysis function should return a sentiment score (handling errors gracefully for problematic inputs).
**Validates: Requirements 5.1**

### Property 8: Sentiment aggregation
*For any* list of sentiment scores with associated dates, the weekly aggregation function should return a Series with weekly average sentiment values.
**Validates: Requirements 5.2**

### Property 9: Sentiment normalization
*For any* sentiment score computed by the system, the value should be normalized to the range [-1, 1].
**Validates: Requirements 5.5**

### Property 10: Feature importance availability
*For any* trained RandomForestRegressor model, the system should be able to extract and display feature importance values for all features used in training.
**Validates: Requirements 6.3**

### Property 11: Prediction value display
*For any* prediction generated by the model, the predicted FSI value should appear in the rendered output.
**Validates: Requirements 6.2, 6.5**

### Property 12: Anomaly detection
*For any* MVI time series containing values that exceed the threshold (e.g., 2 standard deviations from mean), the anomaly detection algorithm should identify those indices as anomalies.
**Validates: Requirements 7.1**

### Property 13: Anomaly visualization
*For any* detected anomalies, the visualization should include both highlighting in the chart and a notification message in the UI.
**Validates: Requirements 7.2, 7.3**

### Property 14: Error isolation
*For any* component that encounters a processing error, other dashboard components should continue to function and display their data correctly.
**Validates: Requirements 10.4**

### Property 15: Error message display
*For any* API request failure, the dashboard should display a user-friendly error message that explains the issue.
**Validates: Requirements 10.1**

## Error Handling

### API Failure Handling

1. **Retry Logic**: Implement exponential backoff retry mechanism for failed API requests
   - Initial retry after 1 second
   - Double wait time for each subsequent retry (max 3 retries)
   - Display retry attempt count to user

2. **PyTrends Rate Limit Handling**: PyTrends may temporarily block requests
   - Automatically retry 3 times using exponential backoff
   - If still failing → load cached Google Trends data
   - Show message: "Google Trends is temporarily unavailable. Showing last available data."

3. **Graceful Degradation**: When APIs fail, display cached data if available with timestamp indicating data age

4. **User Feedback**: Show clear error messages with actionable information:
   - "Unable to fetch GIPHY data. Retrying..."
   - "Google Trends temporarily unavailable. Showing cached data from [timestamp]"

### Data Processing Errors

1. **Sentiment Analysis Failures**: Skip problematic titles and continue processing remaining memes
2. **Prediction Model Failures**: Display message when insufficient data available (< 4 weeks of history)
3. **Anomaly Detection Failures**: Fall back to displaying raw data without anomaly highlighting

### Input Validation

1. **Country Code Validation**: Validate country codes against PyTrends supported list
2. **Date Range Validation**: Ensure timeframe parameters are valid for PyTrends API
3. **API Key Validation**: Check for presence of GIPHY API key before making requests

## Testing Strategy

### Unit Testing Approach

The application will use **pytest** as the testing framework. Unit tests will focus on:

1. **Data Fetching Functions**:
   - Test GIPHY API integration with mocked responses
   - Test PyTrends integration with mocked responses
   - Test error handling for API failures

2. **Metric Computation Functions**:
   - Test MVI calculation with various list sizes
   - Test FSI calculation with sample DataFrames
   - Test correlation computation with known data

3. **Sentiment Analysis**:
   - Test sentiment scoring with sample meme titles
   - Test error handling for problematic text inputs
   - Test weekly aggregation logic

4. **Prediction Model**:
   - Test model training with sufficient data
   - Test handling of insufficient data scenarios
   - Test feature importance extraction

5. **Anomaly Detection**:
   - Test detection with known outliers
   - Test handling of normal data (no anomalies)

### Property-Based Testing Approach

The application will use **Hypothesis** for property-based testing. Each property-based test will:

- Be tagged with a comment explicitly referencing the correctness property from this design document
- Use the format: `# Feature: meme-flu-dashboard, Property {number}: {property_text}`

**Hypothesis Test Settings**:
- Use reduced iterations in CI: `@settings(max_examples=25)  # CI-safe`
- Use full iterations locally: `@settings(max_examples=100)  # Development mode`

Property-based tests will cover:

1. **Property 1 (MVI Computation)**: Generate random lists of meme objects and verify MVI equals list length
2. **Property 2 (FSI Computation)**: Generate random trend DataFrames and verify FSI calculation
3. **Property 3 (Country Parameter)**: Generate random country codes and verify they're passed to API
4. **Property 5 (Metric Rendering)**: Generate random metric values and verify all appear in output
5. **Property 6 (Time Axis)**: Generate random time series and verify alignment
6. **Property 7 (Sentiment Computation)**: Generate random text strings and verify sentiment scores are returned
7. **Property 8 (Sentiment Aggregation)**: Generate random sentiment data and verify weekly aggregation
8. **Property 9 (Sentiment Normalization)**: Generate random sentiment scores and verify normalization to [-1, 1]
9. **Property 12 (Anomaly Detection)**: Generate time series with artificial spikes and verify detection
10. **Property 14 (Error Isolation)**: Simulate component failures and verify other components continue

### Integration Testing

While the focus is on unit and property-based tests, key integration points to validate:

1. End-to-end data flow from API fetch to UI rendering
2. Interaction between prediction model and visualization components
3. Caching behavior across multiple dashboard refreshes

### Test Configuration

- Minimum 100 iterations for each property-based test
- Use pytest fixtures for common test data (sample memes, trend data)
- Mock external API calls to avoid rate limiting and ensure test reliability
- Use pytest-cov for coverage reporting (target: >80% coverage)

## Implementation Notes

### Technology Stack

- **Framework**: Streamlit 1.28+
- **Data Processing**: pandas, numpy
- **API Integration**: requests (GIPHY), pytrends (Google Trends)
- **Sentiment Analysis**: TextBlob or vaderSentiment
- **Machine Learning**: scikit-learn (RandomForestRegressor)
- **Visualization**: plotly or matplotlib
- **Testing**: pytest, hypothesis

### Caching Strategy

Use Streamlit's `@st.cache_data` decorator for:
- GIPHY API responses (TTL: 5 minutes)
- Google Trends data (TTL: 15 minutes)
- Trained prediction models (TTL: 1 hour)

### Performance Considerations

1. Limit GIPHY API calls to 50 trending memes to reduce response time
2. Use PyTrends rate limiting to avoid API blocks
3. Cache sentiment analysis results to avoid recomputation
4. Lazy load meme thumbnails in UI

### Security Considerations

1. **GIPHY API Key Storage**: API key must be stored in `.streamlit/secrets.toml`:
   ```toml
   [GIPHY]
   API_KEY = "your_key"
   ```
   Access via: `api_key = st.secrets["GIPHY"]["API_KEY"]`
   This ensures security and deployability.

2. Validate all user inputs (country codes, date ranges)
3. Sanitize meme titles before sentiment analysis to prevent injection attacks
4. Implement rate limiting on API calls to prevent abuse

### Deployment Considerations

1. Application can be deployed on Streamlit Cloud, Heroku, or similar platforms
2. Ensure environment variables are properly configured for API keys
3. Monitor API usage to stay within rate limits
4. Consider implementing request queuing for high-traffic scenarios
