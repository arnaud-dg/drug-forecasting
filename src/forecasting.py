"""
Drug sales Forecasting is a project dedicated to apply time series forecasting 
to predict the sales of pharmaceutical products in France.
This project is based on the Nixtla library, which provides a unified interface.

This module is for time series forecasting functionality.
This module handles the initialization and execution of various forecasting models,
including data preparation and statistics calculation.

Author: Arnaud Duigou
Date : Nov 2024
"""

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    DynamicOptimizedTheta,
    SeasonalNaive,
    AutoARIMA
)
import os

# Enable Nixtla ID as column setting
os.environ['NIXTLA_ID_AS_COL'] = '1'

class ForecastDataPreparator:
    """
    Class for preparing data for forecasting operations.
    """
    
    @staticmethod
    def prepare_data(df: pl.DataFrame, filters: dict) -> pl.DataFrame:
        """
        Prepare data for forecasting by applying filters and formatting.

        Args:
            df (pl.DataFrame): Input DataFrame
            filters (dict): Dictionary of filters to apply

        Returns:
            pl.DataFrame: Prepared DataFrame for forecasting
        """
        if not isinstance(df, pl.DataFrame):
            return None
            
        # Apply filters
        filtered_df = df
        for col, value in filters.items():
            if value and col in df.columns:
                filtered_df = filtered_df.filter(pl.col(col) == value)
                
        if filtered_df.height == 0:
            return None
            
        # Group and aggregate data
        forecast_data = (
            filtered_df
            .groupby('Date')
            .agg(pl.sum('Value').alias('y'))
            .sort('Date')
        )
        
        # Add required columns for StatsForecast
        forecast_data = forecast_data.with_columns([
            pl.lit('series1').alias('unique_id'),
            pl.col('Date').alias('ds')
        ])
        
        return forecast_data

class ForecastModelManager:
    """
    Class for managing forecast models and generating predictions.
    """
    
    @staticmethod
    def initialize_models(season_length: int = 12) -> StatsForecast:
        """
        Initialize StatsForecast models.

        Args:
            season_length (int): Season length for models

        Returns:
            StatsForecast: Initialized forecaster object
        """
        models = [
            HoltWinters(season_length=season_length),
            DynamicOptimizedTheta(season_length=season_length),
            SeasonalNaive(season_length=season_length),
            AutoARIMA(season_length=season_length)
        ]
        
        return StatsForecast(
            models=models,
            freq='MS',
            n_jobs=-1
        )
    
    @staticmethod
    def generate_forecasts(
        df: pl.DataFrame, 
        filters: dict, 
        forecast_horizon: int = 12, 
        confidence_level: list = [95]
    ) -> pl.DataFrame:
        """
        Generate forecasts using StatsForecast models.

        Args:
            df (pl.DataFrame): Input DataFrame
            filters (dict): Filter settings
            forecast_horizon (int): Number of periods to forecast
            confidence_level (list): Confidence levels for prediction intervals

        Returns:
            pl.DataFrame: DataFrame containing forecasts and confidence intervals
        """
        # Prepare data
        prepared_data = ForecastDataPreparator.prepare_data(df, filters)
        if prepared_data is None:
            return None
        
        # Convert to pandas for StatsForecast
        pandas_data = prepared_data.to_pandas()
        
        # Initialize and fit model
        forecaster = ForecastModelManager.initialize_models()
        forecaster.fit(pandas_data)
        
        # Generate forecasts and fitted values
        future_forecasts = pl.from_pandas(
            forecaster.forecast(h=forecast_horizon, level=confidence_level)
        )
        fitted_values = pl.from_pandas(
            forecaster.forecast(h=1, level=confidence_level, fitted=True)
        )
        
        # Combine and format results
        results_df = ForecastModelManager._format_results(
            fitted_values,
            future_forecasts,
            prepared_data,
            confidence_level
        )
        
        return results_df.sort('Date')
    
    @staticmethod
    def _format_results(
        fitted_values: pl.DataFrame,
        future_forecasts: pl.DataFrame,
        historical_data: pl.DataFrame,
        confidence_level: list
    ) -> pl.DataFrame:
        """
        Format and combine forecast results.

        Args:
            fitted_values (pl.DataFrame): Historical fitted values
            future_forecasts (pl.DataFrame): Future predictions
            historical_data (pl.DataFrame): Original historical data
            confidence_level (list): Confidence levels used

        Returns:
            pl.DataFrame: Formatted results
        """
        # Combine forecasts
        combined_forecasts = pl.concat([fitted_values, future_forecasts])
        results_df = combined_forecasts.with_columns([
            pl.col('ds').alias('Date')
        ])
        
        # Join with historical data
        historical = historical_data.select([
            pl.col('Date'), 
            pl.col('y').alias('Value')
        ])
        results_df = results_df.join(historical, on='Date', how='left')
        
        # Rename columns
        main_model = 'HoltWinters'
        results_df = results_df.rename({
            main_model: 'Forecast',
            f'{main_model}-lo-{confidence_level[0]}': 'Lower_CI',
            f'{main_model}-hi-{confidence_level[0]}': 'Upper_CI'
        })
        
        # Use actual values for historical dates
        results_df = results_df.with_columns(
            pl.when(pl.col('Value').is_not_null())
            .then(pl.col('Value'))
            .otherwise(pl.col('Forecast'))
            .alias('Forecast')
        )
        
        return results_df

def calculate_forecast_statistics(forecast_df: pl.DataFrame) -> dict:
    """
    Calculate summary statistics for forecasts.

    Args:
        forecast_df (pl.DataFrame): DataFrame containing forecasts

    Returns:
        dict: Dictionary of forecast statistics
    """
    if forecast_df is None:
        return None
    
    # Calculate statistics for future values only
    forecast_only = forecast_df.filter(pl.col('Value').is_null())
    
    return forecast_only.select([
        pl.col('Forecast').mean().alias('mean_forecast'),
        pl.col('Forecast').min().alias('min_forecast'),
        pl.col('Forecast').max().alias('max_forecast'),
        pl.col('Lower_CI').mean().alias('lower_ci_mean'),
        pl.col('Upper_CI').mean().alias('upper_ci_mean')
    ]).to_dict(as_series=False)