"""
Test for country selector functionality (Task 16.1)
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from app import fetch_flu_trends


class TestCountrySelector:
    """Tests for country selector functionality"""
    
    def test_fetch_flu_trends_with_country_code(self):
        """Test that fetch_flu_trends accepts and uses geo parameter"""
        mock_trends_data = pd.DataFrame({
            'flu': [50, 55, 60],
            'fever': [45, 50, 55],
            'influenza': [40, 45, 50],
            'isPartial': [False, False, False]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            # Test with US country code
            result = fetch_flu_trends(geo='US')
            
            # Verify build_payload was called with correct geo parameter
            mock_pytrends.build_payload.assert_called_once()
            call_args = mock_pytrends.build_payload.call_args
            assert call_args[1]['geo'] == 'US'
            
            # Verify result is a DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
    
    def test_fetch_flu_trends_with_worldwide_default(self):
        """Test that fetch_flu_trends defaults to worldwide (empty string)"""
        mock_trends_data = pd.DataFrame({
            'flu': [50, 55, 60],
            'fever': [45, 50, 55],
            'influenza': [40, 45, 50],
            'isPartial': [False, False, False]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='W'))
        
        with patch('app.TrendReq') as mock_trend_req:
            mock_pytrends = MagicMock()
            mock_pytrends.interest_over_time.return_value = mock_trends_data
            mock_trend_req.return_value = mock_pytrends
            
            # Test with default (worldwide)
            result = fetch_flu_trends(geo='')
            
            # Verify build_payload was called with empty geo parameter
            mock_pytrends.build_payload.assert_called_once()
            call_args = mock_pytrends.build_payload.call_args
            assert call_args[1]['geo'] == ''
            
            # Verify result is a DataFrame
            assert isinstance(result, pd.DataFrame)
    
    def test_fetch_flu_trends_with_different_countries(self):
        """Test that fetch_flu_trends works with various country codes"""
        mock_trends_data = pd.DataFrame({
            'flu': [50, 55],
            'fever': [45, 50],
            'influenza': [40, 45],
            'isPartial': [False, False]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='W'))
        
        country_codes = ['GB', 'CA', 'AU']
        
        for country_code in country_codes:
            with patch('app.TrendReq') as mock_trend_req:
                mock_pytrends = MagicMock()
                mock_pytrends.interest_over_time.return_value = mock_trends_data
                mock_trend_req.return_value = mock_pytrends
                
                # Clear cache before each test to ensure fresh call
                import streamlit as st
                st.cache_data.clear()
                
                # Test with country code
                result = fetch_flu_trends(geo=country_code)
                
                # Verify build_payload was called with correct geo parameter
                call_args = mock_pytrends.build_payload.call_args
                assert call_args[1]['geo'] == country_code
                
                # Verify result is a DataFrame
                assert isinstance(result, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
