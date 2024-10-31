import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    DynamicOptimizedTheta,
    SeasonalNaive,
    AutoARIMA
)
from datetime import datetime

import os
os.environ['NIXTLA_ID_AS_COL'] = '1'

#########################################################################################
#####################            Statsforecast - Nixtla             #####################
#########################################################################################

def prepare_forecast_data(df: pl.DataFrame, filters: dict) -> pl.DataFrame:
    """Prepare data for forecasting by applying filters and formatting"""
    if not isinstance(df, pl.DataFrame):
        return None
        
    # Apply filters
    filtered_df = df
    for col, value in filters.items():
        if value:
            filtered_df = filtered_df.filter(pl.col(col) == value)
            
    if filtered_df.height == 0:
        return None
        
    # Group and aggregate data if needed
    forecast_data = (filtered_df
        .groupby('Date')
        .agg(pl.sum('Value').alias('y'))
        .sort('Date')
    )
    
    # Create unique_id column required by StatsForecast
    forecast_data = forecast_data.with_columns([
        pl.lit('series1').alias('unique_id'),
        pl.col('Date').alias('ds')
    ])
    
    return forecast_data

def get_forecast_model(season_length: int = 12):
    """Initialize StatsForecast model"""
    models = [
        HoltWinters(season_length=season_length),
        DynamicOptimizedTheta(season_length=season_length),
        SeasonalNaive(season_length=season_length),
        SeasonalNaive(season_length=season_length),
        AutoARIMA(season_length=season_length)
    ]
    
    return StatsForecast(
        models=models,
        freq='MS',
        n_jobs=-1
    )

def generate_forecasts(df: pl.DataFrame, filters: dict, forecast_horizon: int = 12, confidence_level: list = [95]) -> pl.DataFrame:
    """Generate forecasts using StatsForecasts"""
    # Prepare data
    forecast_data = prepare_forecast_data(df, filters)
    if forecast_data is None:
        return None
    
    # Convert to pandas for StatsForecast
    temp_pd_data = forecast_data.to_pandas()
    
    # Initialize and fit model
    forecaster = get_forecast_model()
    forecaster.fit(temp_pd_data)
    
    # Generate forecasts
    future_forecasts = pl.from_pandas(
        forecaster.forecast(h=forecast_horizon, level=confidence_level)
    )
    
    # Get fitted values (historical predictions)
    fitted_values = pl.from_pandas(
        forecaster.forecast(h=1, level=confidence_level, fitted=True)
    )
    
    # Combine results
    combined_forecasts = pl.concat([fitted_values, future_forecasts])
    
    # Format results
    results_df = combined_forecasts.with_columns([
        pl.col('ds').alias('Date')
    ])
    
    # Join with historical data
    historical_data = forecast_data.select([
        pl.col('Date'), 
        pl.col('y').alias('Value')
    ])
    results_df = results_df.join(historical_data, on='Date', how='left')
    
    # Rename columns for clarity
    main_model = 'HoltWinters'  # Using HoltWinters as primary model
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
    
    return results_df.sort('Date')

def get_forecast_statistics(forecast_df: pl.DataFrame) -> dict:
    """Calculate forecast statistics using Polars"""
    if forecast_df is None:
        return None
    
    forecast_only = forecast_df.filter(pl.col('Value').is_null())
    
    stats = forecast_only.select([
        pl.col('Forecast').mean().alias('mean_forecast'),
        pl.col('Forecast').min().alias('min_forecast'),
        pl.col('Forecast').max().alias('max_forecast'),
        pl.col('Lower_CI').mean().alias('lower_ci_mean'),
        pl.col('Upper_CI').mean().alias('upper_ci_mean')
    ]).to_dict(as_series=False)
    
    return stats