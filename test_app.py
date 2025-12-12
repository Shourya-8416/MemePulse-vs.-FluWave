"""
Unit tests for MemePulse vs FluWave Dashboard

Tests for data fetching functions with mocked API responses.
"""

import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch, MagicMock
from app import fetch_giphy_trending, fetch_flu_trends, analyze_sentiment, compute_weekly_sentiment, render_kpi_cards


# ============================================================================
# GIPHY API TESTS
# ============================================================================

class TestFetchGiphyTrending:
    """Test suite for GIPHY API integration"""
    
    def test_fetch_giphy_trending_success(self):
        """Test successful GIPHY API call with valid response"""
        # Mock response data
        mock_response_data = {
            'data': [
                {
                    'id': 'abc123',
                    'title': 'Funny Cat GIF',
                    'url': 'https://giphy.com/gifs/abc123',
                    'images': {
                        'fixed_height': {
                            'url': 'https://media.giphy.com/media/abc123/200.gif'
                        }
                    }
                },
                {
                    'id': 'def456',
                    'title': 'Dancing Dog GIF',
                    'url': 'https://giphy.com/gifs/def456',
                    'images': {
                        'fixed_height': {
                            'url': 'https://media.giphy.com/media/def456/200.gif'
                        }
                    }
                }
            ],
            'pagination': {
                'total_count': 2,
                'count': 2,
                'offset': 0
            }
        }
        
        # Mock the requests.get call
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Call the function
            result = fetch_giphy_trending(api_key='test_api_key', limit=50)
            
            # Assertions
            assert result == mock_response_data
            assert 'data' in result
            assert len(result['data']) == 2
            assert result['data'][0]['id'] == 'abc123'
            assert result['data'][1]['id'] == 'def456'
            
            # Verify the API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[0][0] == 'https://api.giphy.com/v1/gifs/trending'
            assert call_args[1]['params']['api_key'] == 'test_api_key'
            assert call_args[1]['params']['limit'] == 50
    
    def test_fetch_giphy_trending_custom_limit(self):
        """Test GIPHY API call with custom limit parameter"""
        mock_response_data = {'data': [{'id': 'test'}]}
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Call with custom limit
            result = fetch_giphy_trending(api_key='test_key', limit=10)
            
            # Verify limit parameter was passed correctly
            call_args = mock_get.call_args
            assert call_args[1]['params']['limit'] == 10
    
    def test_fetch_giphy_trending_http_error(self):
        """Test GIPHY API call handles HTTP errors correctly"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            # Should raise RequestException
            with pytest.raises(requests.RequestException) as exc_info:
                fetch_giphy_trending(api_key='test_key')
            
            assert "Failed to fetch GIPHY trending data" in str(exc_info.value)
    
    def test_fetch_giphy_trending_connection_error(self):
        """Test GIPHY API call handles connection errors"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection failed")
            
            # Should raise RequestException
            with pytest.raises(requests.RequestException) as exc_info:
                fetch_giphy_trending(api_key='test_key')
            
            assert "Failed to fetch GIPHY trending data" in str(exc_info.value)
    
    def test_fetch_giphy_trending_timeout_error(self):
        """Test GIPHY API call handles timeout errors"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")
            
            # Should raise RequestException
            with pytest.raises(requests.RequestException) as exc_info:
                fetch_giphy_trending(api_key='test_key')
            
            assert "Failed to fetch GIPHY trending data" in str(exc_info.value)
    
    def test_fetch_giphy_trending_empty_response(self):
        """Test GIPHY API call with empty data array"""
        mock_response_data = {
            'data': [],
            'pagination': {'total_count': 0, 'count': 0, 'offset': 0}
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = fetch_giphy_trending(api_key='test_key')
            
            assert result == mock_response_data
            assert 'data' in result
            assert len(result['data']) == 0


# ============================================================================
# GOOGLE TRENDS (PYTRENDS) TESTS
# ============================================================================

class TestFetchFluTrends:
    """Test suite for Google Trends (PyTrends) integration"""
    
    def test_fetch_flu_trends_success(self):
        """Test successful Google Trends API call with valid response"""
        # Create mock DataFrame with flu trend data
        mock_dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
        mock_trends_data = pd.DataFrame({
            'flu': [45, 52, 48, 55, 60, 58, 62, 65, 70, 68, 72, 75],
            'fever': [40, 48, 45, 50, 55, 53, 58, 60, 65, 63, 68, 70],
            'influenza': [35, 42, 40, 45, 50, 48, 52, 55, 60, 58, 62, 65],
            'isPartial': [False] * 12
        }, index=mock_dates)
        
        # Mock PyTrends
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            # Call the function
            result = fetch_flu_trends(keywords=['flu', 'fever', 'influenza'])
            
            # Assertions
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 12
            assert 'flu' in result.columns
            assert 'fever' in result.columns
            assert 'influenza' in result.columns
            assert 'isPartial' not in result.columns  # Should be removed
            
            # Verify PyTrends was called correctly
            mock_trend_req.assert_called_once_with(hl='en-US', tz=360)
            mock_pytrends.build_payload.assert_called_once()
            call_args = mock_pytrends.build_payload.call_args
            assert call_args[0][0] == ['flu', 'fever', 'influenza']
    
    def test_fetch_flu_trends_default_keywords(self):
        """Test Google Trends call with default keywords"""
        mock_trends_data = pd.DataFrame({
            'flu': [50],
            'fever': [45],
            'influenza': [40]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            # Call without keywords parameter
            result = fetch_flu_trends()
            
            # Should use default keywords
            call_args = mock_pytrends.build_payload.call_args
            assert call_args[0][0] == ['flu', 'fever', 'influenza']
    
    def test_fetch_flu_trends_custom_timeframe(self):
        """Test Google Trends call with custom timeframe"""
        mock_trends_data = pd.DataFrame({
            'flu': [50],
            'fever': [45],
            'influenza': [40]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            # Call with custom timeframe
            result = fetch_flu_trends(timeframe='today 1-m')
            
            # Verify timeframe parameter
            call_args = mock_pytrends.build_payload.call_args
            assert call_args[1]['timeframe'] == 'today 1-m'
    
    def test_fetch_flu_trends_custom_geo(self):
        """Test Google Trends call with custom geographic region"""
        mock_trends_data = pd.DataFrame({
            'flu': [50],
            'fever': [45],
            'influenza': [40]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            # Call with country code
            result = fetch_flu_trends(geo='US')
            
            # Verify geo parameter
            call_args = mock_pytrends.build_payload.call_args
            assert call_args[1]['geo'] == 'US'
    
    def test_fetch_flu_trends_removes_ispartial_column(self):
        """Test that isPartial column is removed from results"""
        mock_trends_data = pd.DataFrame({
            'flu': [50, 55],
            'fever': [45, 50],
            'influenza': [40, 45],
            'isPartial': [False, True]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            result = fetch_flu_trends()
            
            # isPartial should be removed
            assert 'isPartial' not in result.columns
            assert 'flu' in result.columns
            assert 'fever' in result.columns
            assert 'influenza' in result.columns
    
    def test_fetch_flu_trends_api_error(self):
        """Test Google Trends call handles API errors"""
        # Clear cache before test
        from app import fetch_flu_trends
        fetch_flu_trends.clear()
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.build_payload.side_effect = Exception("Rate limit exceeded")
            mock_trend_req.return_value = mock_pytrends
            
            # Should raise Exception after retries are exhausted
            with pytest.raises(Exception) as exc_info:
                fetch_flu_trends()
            
            assert "Failed to fetch Google Trends data" in str(exc_info.value)
    
    def test_fetch_flu_trends_connection_error(self):
        """Test Google Trends call handles connection errors"""
        # Clear cache before test
        from app import fetch_flu_trends
        fetch_flu_trends.clear()
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.side_effect = ConnectionError("Network error")
            mock_trend_req.return_value = mock_pytrends
            
            # Should raise Exception after retries are exhausted
            with pytest.raises(Exception) as exc_info:
                fetch_flu_trends()
            
            assert "Failed to fetch Google Trends data" in str(exc_info.value)
    
    def test_fetch_flu_trends_without_ispartial(self):
        """Test Google Trends response without isPartial column"""
        # Clear cache before test
        from app import fetch_flu_trends
        fetch_flu_trends.clear()
        
        mock_trends_data = pd.DataFrame({
            'flu': [50, 55],
            'fever': [45, 50],
            'influenza': [40, 45]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data.copy()
            mock_trend_req.return_value = mock_pytrends
            
            # Should not raise error even without isPartial column
            result = fetch_flu_trends()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'flu' in result.columns


# ============================================================================
# SENTIMENT ANALYSIS TESTS
# ============================================================================

class TestAnalyzeSentiment:
    """Test suite for sentiment analysis function"""
    
    def test_analyze_sentiment_positive_text(self):
        """Test sentiment analysis with positive text"""
        text = "This is amazing and wonderful!"
        sentiment = analyze_sentiment(text)
        
        # Should return positive sentiment
        assert sentiment > 0
        assert -1.0 <= sentiment <= 1.0
    
    def test_analyze_sentiment_negative_text(self):
        """Test sentiment analysis with negative text"""
        text = "This is terrible and awful!"
        sentiment = analyze_sentiment(text)
        
        # Should return negative sentiment
        assert sentiment < 0
        assert -1.0 <= sentiment <= 1.0
    
    def test_analyze_sentiment_neutral_text(self):
        """Test sentiment analysis with neutral text"""
        text = "This is a thing."
        sentiment = analyze_sentiment(text)
        
        # Should return sentiment close to 0
        assert -1.0 <= sentiment <= 1.0
    
    def test_analyze_sentiment_empty_string(self):
        """Test sentiment analysis with empty string"""
        sentiment = analyze_sentiment("")
        assert sentiment == 0.0
    
    def test_analyze_sentiment_whitespace_only(self):
        """Test sentiment analysis with whitespace-only string"""
        sentiment = analyze_sentiment("   \n\t  ")
        assert sentiment == 0.0
    
    def test_analyze_sentiment_none_input(self):
        """Test sentiment analysis with None input"""
        sentiment = analyze_sentiment(None)
        assert sentiment == 0.0
    
    def test_analyze_sentiment_non_string_input(self):
        """Test sentiment analysis with non-string input"""
        sentiment = analyze_sentiment(123)
        assert sentiment == 0.0
    
    def test_analyze_sentiment_normalized_range(self):
        """Test that sentiment scores are always in [-1, 1] range"""
        test_texts = [
            "I love this so much!",
            "I hate this completely!",
            "This is okay.",
            "Absolutely fantastic wonderful amazing!",
            "Terrible horrible awful disgusting!"
        ]
        
        for text in test_texts:
            sentiment = analyze_sentiment(text)
            assert -1.0 <= sentiment <= 1.0, f"Sentiment {sentiment} out of range for text: {text}"
    
    def test_analyze_sentiment_special_characters(self):
        """Test sentiment analysis with special characters"""
        text = "Great!!! 😊 #awesome @user"
        sentiment = analyze_sentiment(text)
        
        # Should handle special characters gracefully
        assert -1.0 <= sentiment <= 1.0


class TestComputeWeeklySentiment:
    """Test suite for weekly sentiment aggregation function"""
    
    def test_compute_weekly_sentiment_basic(self):
        """Test basic weekly sentiment aggregation"""
        meme_data = [
            {
                'title': 'Happy cat is happy',
                'trending_datetime': '2024-01-01 10:00:00'
            },
            {
                'title': 'Sad dog is sad',
                'trending_datetime': '2024-01-02 15:00:00'
            },
            {
                'title': 'Excited puppy!',
                'trending_datetime': '2024-01-08 12:00:00'
            }
        ]
        
        result = compute_weekly_sentiment(meme_data)
        
        # Should return a Series
        assert isinstance(result, pd.Series)
        
        # Should have weekly aggregated data
        assert len(result) >= 1
        
        # All values should be in [-1, 1] range
        for value in result:
            assert -1.0 <= value <= 1.0
    
    def test_compute_weekly_sentiment_empty_list(self):
        """Test weekly sentiment with empty list"""
        result = compute_weekly_sentiment([])
        
        assert isinstance(result, pd.Series)
        assert len(result) == 0
    
    def test_compute_weekly_sentiment_none_input(self):
        """Test weekly sentiment with None input"""
        result = compute_weekly_sentiment(None)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 0
    
    def test_compute_weekly_sentiment_invalid_data(self):
        """Test weekly sentiment with invalid data"""
        meme_data = [
            {'title': 'Valid meme'},  # Missing date
            {'trending_datetime': '2024-01-01'},  # Missing title
            'not a dict',  # Invalid type
            None  # None value
        ]
        
        result = compute_weekly_sentiment(meme_data)
        
        # Should handle invalid data gracefully
        assert isinstance(result, pd.Series)
    
    def test_compute_weekly_sentiment_custom_date_key(self):
        """Test weekly sentiment with custom date key"""
        meme_data = [
            {
                'title': 'Happy meme',
                'custom_date': '2024-01-01 10:00:00'
            }
        ]
        
        result = compute_weekly_sentiment(meme_data, date_key='custom_date')
        
        assert isinstance(result, pd.Series)
        assert len(result) >= 1
    
    def test_compute_weekly_sentiment_multiple_weeks(self):
        """Test weekly sentiment aggregation across multiple weeks"""
        meme_data = [
            {'title': 'Great!', 'trending_datetime': '2024-01-01'},
            {'title': 'Awesome!', 'trending_datetime': '2024-01-02'},
            {'title': 'Terrible!', 'trending_datetime': '2024-01-08'},
            {'title': 'Horrible!', 'trending_datetime': '2024-01-09'},
            {'title': 'Amazing!', 'trending_datetime': '2024-01-15'}
        ]
        
        result = compute_weekly_sentiment(meme_data)
        
        # Should have multiple weeks
        assert len(result) >= 2
        
        # All values should be in valid range
        for value in result:
            assert -1.0 <= value <= 1.0
    
    def test_compute_weekly_sentiment_invalid_dates(self):
        """Test weekly sentiment with invalid date strings"""
        meme_data = [
            {'title': 'Valid meme', 'trending_datetime': 'not-a-date'},
            {'title': 'Another meme', 'trending_datetime': '2024-01-01'}
        ]
        
        result = compute_weekly_sentiment(meme_data)
        
        # Should skip invalid dates and process valid ones
        assert isinstance(result, pd.Series)
    
    def test_compute_weekly_sentiment_aggregation_logic(self):
        """Test that weekly aggregation correctly averages sentiment scores"""
        # Create memes in the same week with known sentiments
        meme_data = [
            {'title': 'Neutral text', 'trending_datetime': '2024-01-01'},
            {'title': 'Neutral text', 'trending_datetime': '2024-01-02'},
            {'title': 'Neutral text', 'trending_datetime': '2024-01-03'}
        ]
        
        result = compute_weekly_sentiment(meme_data)
        
        # Should have one week of data
        assert len(result) >= 1
        
        # Average of neutral sentiments should be close to 0
        assert abs(result.iloc[0]) < 0.5


# ============================================================================
# PREDICTION MODEL TESTS
# ============================================================================

class TestPredictionModel:
    """Test suite for prediction model functions"""
    
    def test_train_predictor_success(self):
        """Test successful model training with sufficient data"""
        from app import train_predictor
        
        # Create 52 weeks of synthetic data
        dates = pd.date_range(start='2023-01-01', periods=52, freq='W-MON')
        
        # Create FSI data with some trend
        fsi_values = [50 + i * 0.5 + (i % 4) * 2 for i in range(52)]
        fsi_history = pd.Series(fsi_values, index=dates)
        
        # Create MVI data
        mvi_values = [100 + i * 1.2 + (i % 3) * 5 for i in range(52)]
        mvi_history = pd.Series(mvi_values, index=dates)
        
        # Create sentiment data
        sentiment_values = [0.1 + (i % 10) * 0.05 for i in range(52)]
        sentiment_history = pd.Series(sentiment_values, index=dates)
        
        # Train model
        model, prediction = train_predictor(fsi_history, mvi_history, sentiment_history)
        
        # Assertions
        assert model is not None
        assert isinstance(prediction, float)
        assert prediction > 0  # FSI should be positive
    
    def test_train_predictor_insufficient_data(self):
        """Test that model raises error with insufficient data"""
        from app import train_predictor
        
        # Create only 30 weeks of data (less than 52 required)
        dates = pd.date_range(start='2023-01-01', periods=30, freq='W-MON')
        
        fsi_history = pd.Series([50] * 30, index=dates)
        mvi_history = pd.Series([100] * 30, index=dates)
        sentiment_history = pd.Series([0.5] * 30, index=dates)
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            train_predictor(fsi_history, mvi_history, sentiment_history)
        
        assert "Insufficient data" in str(exc_info.value)
        assert "52 weeks" in str(exc_info.value)
    
    def test_train_predictor_none_fsi(self):
        """Test that model raises error with None FSI history"""
        from app import train_predictor
        
        dates = pd.date_range(start='2023-01-01', periods=52, freq='W-MON')
        mvi_history = pd.Series([100] * 52, index=dates)
        sentiment_history = pd.Series([0.5] * 52, index=dates)
        
        with pytest.raises(ValueError) as exc_info:
            train_predictor(None, mvi_history, sentiment_history)
        
        assert "Insufficient data" in str(exc_info.value)
    
    def test_train_predictor_none_mvi(self):
        """Test that model raises error with None MVI history"""
        from app import train_predictor
        
        dates = pd.date_range(start='2023-01-01', periods=52, freq='W-MON')
        fsi_history = pd.Series([50] * 52, index=dates)
        sentiment_history = pd.Series([0.5] * 52, index=dates)
        
        with pytest.raises(ValueError) as exc_info:
            train_predictor(fsi_history, None, sentiment_history)
        
        assert "MVI history is required" in str(exc_info.value)
    
    def test_train_predictor_none_sentiment(self):
        """Test that model raises error with None sentiment history"""
        from app import train_predictor
        
        dates = pd.date_range(start='2023-01-01', periods=52, freq='W-MON')
        fsi_history = pd.Series([50] * 52, index=dates)
        mvi_history = pd.Series([100] * 52, index=dates)
        
        with pytest.raises(ValueError) as exc_info:
            train_predictor(fsi_history, mvi_history, None)
        
        assert "Sentiment history is required" in str(exc_info.value)
    
    def test_get_feature_importance_success(self):
        """Test feature importance extraction from trained model"""
        from app import train_predictor, get_feature_importance
        
        # Create 52 weeks of synthetic data
        dates = pd.date_range(start='2023-01-01', periods=52, freq='W-MON')
        
        fsi_history = pd.Series([50 + i * 0.5 for i in range(52)], index=dates)
        mvi_history = pd.Series([100 + i * 1.2 for i in range(52)], index=dates)
        sentiment_history = pd.Series([0.1 + i * 0.01 for i in range(52)], index=dates)
        
        # Train model
        model, _ = train_predictor(fsi_history, mvi_history, sentiment_history)
        
        # Get feature importance
        importance = get_feature_importance(model)
        
        # Assertions
        assert isinstance(importance, dict)
        assert len(importance) == 8  # Should have 8 features
        
        # Check that all expected features are present
        expected_features = [
            'FSI_lag_1', 'FSI_lag_2', 'FSI_lag_3', 'FSI_lag_4',
            'MVI_lag_1', 'MVI_lag_2', 'Sentiment_avg', 'Month'
        ]
        for feature in expected_features:
            assert feature in importance
            assert isinstance(importance[feature], float)
            assert 0 <= importance[feature] <= 1  # Importance should be normalized
        
        # Sum of all importances should be approximately 1.0
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.01
    
    def test_get_feature_importance_invalid_model(self):
        """Test that feature importance raises error with invalid model"""
        from app import get_feature_importance
        
        # Pass a non-model object
        with pytest.raises(TypeError) as exc_info:
            get_feature_importance("not a model")
        
        assert "RandomForestRegressor" in str(exc_info.value)
    
    def test_get_feature_importance_unfitted_model(self):
        """Test that feature importance raises error with unfitted model"""
        from app import get_feature_importance
        from sklearn.ensemble import RandomForestRegressor
        
        # Create unfitted model
        model = RandomForestRegressor()
        
        with pytest.raises(ValueError) as exc_info:
            get_feature_importance(model)
        
        assert "not been fitted" in str(exc_info.value)


# ============================================================================
# UI RENDERING TESTS
# ============================================================================

class TestRenderKPICards:
    """Test suite for KPI card rendering function"""
    
    def test_render_kpi_cards_basic(self):
        """Test that render_kpi_cards executes without errors"""
        # Mock Streamlit functions
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            # Mock columns to return mock column objects
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_col3 = MagicMock()
            mock_col4 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]
            
            # Call the function with sample data
            render_kpi_cards(mvi=150, fsi=55.5, correlation=0.75, sentiment=0.3)
            
            # Verify columns were created
            mock_columns.assert_called_once_with(4)
            
            # Verify metric was called 4 times (once for each KPI)
            assert mock_metric.call_count == 4
    
    def test_render_kpi_cards_with_various_values(self):
        """Test KPI cards with various metric values"""
        test_cases = [
            # (mvi, fsi, correlation, sentiment)
            (100, 50.0, 0.5, 0.2),
            (0, 0.0, 0.0, 0.0),
            (1000, 100.0, -0.8, -0.5),
            (50, 25.5, 0.95, 0.9),
            (200, 75.3, -0.3, -0.1)
        ]
        
        for mvi, fsi, correlation, sentiment in test_cases:
            with patch('app.st.columns') as mock_columns, \
                 patch('app.st.metric') as mock_metric:
                
                # Mock columns
                mock_cols = [MagicMock() for _ in range(4)]
                mock_columns.return_value = mock_cols
                
                # Should not raise any errors
                render_kpi_cards(mvi=mvi, fsi=fsi, correlation=correlation, sentiment=sentiment)
                
                # Verify function executed
                mock_columns.assert_called_once_with(4)
                assert mock_metric.call_count == 4
    
    def test_render_kpi_cards_mvi_formatting(self):
        """Test that MVI is formatted with comma separators"""
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            mock_cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = mock_cols
            
            # Call with large MVI value
            render_kpi_cards(mvi=1500, fsi=50.0, correlation=0.5, sentiment=0.2)
            
            # Check that first metric call (MVI) has formatted value
            first_call = mock_metric.call_args_list[0]
            assert '1,500' in str(first_call)
    
    def test_render_kpi_cards_fsi_formatting(self):
        """Test that FSI is formatted with 2 decimal places"""
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            mock_cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = mock_cols
            
            # Call with FSI value
            render_kpi_cards(mvi=100, fsi=55.567, correlation=0.5, sentiment=0.2)
            
            # Check that second metric call (FSI) has 2 decimal places
            second_call = mock_metric.call_args_list[1]
            assert '55.57' in str(second_call)
    
    def test_render_kpi_cards_correlation_formatting(self):
        """Test that correlation is formatted with sign and 3 decimal places"""
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            mock_cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = mock_cols
            
            # Test positive correlation
            render_kpi_cards(mvi=100, fsi=50.0, correlation=0.756, sentiment=0.2)
            third_call = mock_metric.call_args_list[2]
            assert '+0.756' in str(third_call)
            
            # Reset mocks
            mock_metric.reset_mock()
            
            # Test negative correlation
            render_kpi_cards(mvi=100, fsi=50.0, correlation=-0.456, sentiment=0.2)
            third_call = mock_metric.call_args_list[2]
            assert '-0.456' in str(third_call)
    
    def test_render_kpi_cards_sentiment_formatting(self):
        """Test that sentiment is formatted with sign and 3 decimal places"""
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            mock_cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = mock_cols
            
            # Test positive sentiment
            render_kpi_cards(mvi=100, fsi=50.0, correlation=0.5, sentiment=0.345)
            fourth_call = mock_metric.call_args_list[3]
            assert '+0.345' in str(fourth_call)
            
            # Reset mocks
            mock_metric.reset_mock()
            
            # Test negative sentiment
            render_kpi_cards(mvi=100, fsi=50.0, correlation=0.5, sentiment=-0.234)
            fourth_call = mock_metric.call_args_list[3]
            assert '-0.234' in str(fourth_call)
    
    def test_render_kpi_cards_all_metrics_displayed(self):
        """Test that all four metrics are displayed (Requirements 3.1, 3.2, 3.3, 3.4)"""
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            mock_cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = mock_cols
            
            render_kpi_cards(mvi=150, fsi=55.5, correlation=0.75, sentiment=0.3)
            
            # Verify all 4 metrics were called
            assert mock_metric.call_count == 4
            
            # Verify labels contain expected text
            call_args = [str(call) for call in mock_metric.call_args_list]
            
            # Check for MVI label (Requirement 3.1)
            assert any('Meme Virality Index' in arg for arg in call_args)
            
            # Check for FSI label (Requirement 3.2)
            assert any('Flu Search Index' in arg for arg in call_args)
            
            # Check for Correlation label (Requirement 3.3)
            assert any('Correlation' in arg for arg in call_args)
            
            # Check for Sentiment label (Requirement 3.4)
            assert any('Sentiment Score' in arg for arg in call_args)
    
    def test_render_kpi_cards_extreme_values(self):
        """Test KPI cards with extreme values"""
        with patch('app.st.columns') as mock_columns, \
             patch('app.st.metric') as mock_metric:
            
            mock_cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = mock_cols
            
            # Test with extreme values
            render_kpi_cards(mvi=999999, fsi=100.0, correlation=1.0, sentiment=1.0)
            
            # Should not raise errors
            assert mock_metric.call_count == 4
            
            # Reset and test with minimum values
            mock_metric.reset_mock()
            render_kpi_cards(mvi=0, fsi=0.0, correlation=-1.0, sentiment=-1.0)
            
            assert mock_metric.call_count == 4



# ============================================================================
# ERROR HANDLING AND RETRY LOGIC TESTS
# ============================================================================

class TestRetryLogic:
    """Test suite for retry logic with exponential backoff"""
    
    def test_retry_decorator_success_on_first_attempt(self):
        """Test that retry decorator doesn't retry on successful first attempt"""
        from app import retry_with_exponential_backoff
        
        call_count = {'count': 0}
        
        @retry_with_exponential_backoff(max_retries=3)
        def successful_function():
            call_count['count'] += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count['count'] == 1  # Should only be called once
    
    def test_retry_decorator_success_after_retries(self):
        """Test that retry decorator retries and eventually succeeds"""
        from app import retry_with_exponential_backoff
        
        call_count = {'count': 0}
        
        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def eventually_successful_function():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = eventually_successful_function()
        
        assert result == "success"
        assert call_count['count'] == 3  # Should be called 3 times
    
    def test_retry_decorator_exhausts_retries(self):
        """Test that retry decorator raises exception after max retries"""
        from app import retry_with_exponential_backoff
        
        call_count = {'count': 0}
        
        @retry_with_exponential_backoff(max_retries=2, initial_delay=0.01)
        def always_failing_function():
            call_count['count'] += 1
            raise ValueError("Permanent failure")
        
        with pytest.raises(ValueError) as exc_info:
            always_failing_function()
        
        assert "Permanent failure" in str(exc_info.value)
        assert call_count['count'] == 3  # Initial attempt + 2 retries
    
    def test_retry_decorator_with_specific_exceptions(self):
        """Test that retry decorator only catches specified exceptions"""
        from app import retry_with_exponential_backoff
        
        @retry_with_exponential_backoff(
            max_retries=2,
            initial_delay=0.01,
            exceptions=(ValueError,)
        )
        def function_with_type_error():
            raise TypeError("This should not be retried")
        
        # TypeError should not be caught and retried
        with pytest.raises(TypeError) as exc_info:
            function_with_type_error()
        
        assert "This should not be retried" in str(exc_info.value)
    
    def test_retry_decorator_exponential_backoff_timing(self):
        """Test that retry decorator uses exponential backoff"""
        from app import retry_with_exponential_backoff
        import time
        
        call_times = []
        
        @retry_with_exponential_backoff(
            max_retries=2,
            initial_delay=0.1,
            backoff_factor=2.0
        )
        def timed_function():
            call_times.append(time.time())
            raise ValueError("Test failure")
        
        try:
            timed_function()
        except ValueError:
            pass
        
        # Should have 3 calls (initial + 2 retries)
        assert len(call_times) == 3
        
        # Check that delays are approximately correct (with some tolerance)
        # First delay should be ~0.1s, second delay should be ~0.2s
        if len(call_times) >= 2:
            first_delay = call_times[1] - call_times[0]
            assert 0.08 < first_delay < 0.15  # Allow some tolerance
        
        if len(call_times) >= 3:
            second_delay = call_times[2] - call_times[1]
            assert 0.18 < second_delay < 0.25  # Allow some tolerance


class TestErrorMessageDisplay:
    """Test suite for error message display function"""
    
    def test_display_error_message_api_failure(self):
        """Test error message display for API failures"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('api_failure', 'Connection timeout')
            
            # Verify error was called
            mock_error.assert_called_once()
            
            # Check that message contains expected content
            call_args = str(mock_error.call_args)
            assert 'API Connection Error' in call_args
            assert 'Connection timeout' in call_args
    
    def test_display_error_message_giphy_failure(self):
        """Test error message display for GIPHY failures"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('giphy_failure', 'Invalid API key')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'GIPHY API Error' in call_args
            assert 'Invalid API key' in call_args
    
    def test_display_error_message_trends_failure(self):
        """Test error message display for Google Trends failures"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('trends_failure', 'Rate limit exceeded')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'Google Trends Error' in call_args
            assert 'Rate limit exceeded' in call_args
    
    def test_display_error_message_data_processing(self):
        """Test error message display for data processing errors"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('data_processing', 'Invalid data format')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'Data Processing Error' in call_args
            assert 'Invalid data format' in call_args
    
    def test_display_error_message_insufficient_data(self):
        """Test error message display for insufficient data"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('insufficient_data', 'Need 52 weeks')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'Insufficient Data' in call_args
            assert 'Need 52 weeks' in call_args
    
    def test_display_error_message_general(self):
        """Test error message display for general errors"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('general', 'Unknown error')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'Error' in call_args
            assert 'Unknown error' in call_args
    
    def test_display_error_message_with_retry_flag(self):
        """Test error message display with retry flag"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('api_failure', 'Test error', show_retry=True)
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'Retrying automatically' in call_args
    
    def test_display_error_message_without_details(self):
        """Test error message display without error details"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('api_failure')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert 'API Connection Error' in call_args
    
    def test_display_error_message_unknown_type(self):
        """Test error message display with unknown error type falls back to general"""
        from app import display_error_message
        
        with patch('app.st.error') as mock_error:
            display_error_message('unknown_type', 'Some error')
            
            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            # Should fall back to general error message
            assert 'Error' in call_args


class TestErrorIsolation:
    """Test suite for error isolation in components"""
    
    def test_fetch_giphy_with_retry_decorator(self):
        """Test that fetch_giphy_trending has retry decorator applied"""
        from app import fetch_giphy_trending
        
        # Check that the function has been wrapped
        assert hasattr(fetch_giphy_trending, '__wrapped__')
    
    def test_fetch_flu_trends_with_retry_decorator(self):
        """Test that fetch_flu_trends has retry decorator applied"""
        from app import fetch_flu_trends
        
        # Check that the function has been wrapped
        assert hasattr(fetch_flu_trends, '__wrapped__')
    
    def test_compute_mvi_handles_empty_data(self):
        """Test that compute_mvi handles empty data without crashing"""
        from app import compute_mvi
        
        # Should not raise exception
        result = compute_mvi({})
        assert result == 0
        
        result = compute_mvi({'data': []})
        assert result == 0
        
        result = compute_mvi(None)
        assert result == 0
    
    def test_compute_fsi_handles_empty_data(self):
        """Test that compute_fsi handles empty data without crashing"""
        from app import compute_fsi
        
        # Should not raise exception
        result = compute_fsi(pd.DataFrame())
        assert result == 0.0
        
        result = compute_fsi(None)
        assert result == 0.0
    
    def test_analyze_sentiment_handles_errors(self):
        """Test that analyze_sentiment handles errors gracefully"""
        from app import analyze_sentiment
        
        # Should not raise exception for invalid inputs
        result = analyze_sentiment(None)
        assert result == 0.0
        
        result = analyze_sentiment("")
        assert result == 0.0
        
        result = analyze_sentiment(123)
        assert result == 0.0
    
    def test_detect_anomalies_handles_empty_data(self):
        """Test that detect_anomalies handles empty data without crashing"""
        from app import detect_anomalies
        
        # Should not raise exception
        result = detect_anomalies(pd.Series([]))
        assert result == []
        
        result = detect_anomalies(None)
        assert result == []
        
        result = detect_anomalies(pd.Series([1]))  # Single value
        assert result == []
