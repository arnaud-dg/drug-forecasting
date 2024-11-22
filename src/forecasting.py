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
from utilsforecast.losses import mse
from utilsforecast.evaluation import evaluate
from statsforecast.models import (
    HoltWinters,
    DynamicOptimizedTheta,
    SeasonalNaive,
    AutoARIMA,
    Naive,
    ETS
)
import os
import numpy as np

import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean, 
    RollingMean,
    ExponentiallyWeightedMean
)
from mlforecast.target_transforms import (
    Differences,
    LocalStandardScaler
)

#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
from hierarchicalforecast.utils import aggregate

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

###################################################################################
#######################           Evaluation             ##########################
###################################################################################

def evaluate_cross_validation(df, metric):
    models = [c for c in df.columns if c not in ('unique_id', 'ds', 'cutoff', 'y')]
    evals = []
    # Calculate loss for every unique_id and cutoff.    
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df.filter(pl.col('cutoff') == cutoff), metrics=[metric], models=models)
        evals.append(eval_)
    evals = pl.concat(evals).drop('metric')
    # Calculate the mean of each 'unique_id' group
    evals = evals.group_by(['unique_id'], maintain_order=True).mean() 

    # For each row in evals (excluding 'unique_id'), find the model with the lowest value
    best_model = [min(row, key=row.get) for row in evals.drop('unique_id').rows(named=True)]

    # Add a 'best_model' column to evals dataframe with the best model for each 'unique_id'
    evals = evals.with_columns(pl.Series(best_model).alias('best_model')).sort(by=['unique_id'])
    return evals

def get_best_model_forecast(forecasts_df, evaluation_df):
    """
    Process forecast data to get best model predictions with confidence intervals
    
    Parameters:
    -----------
    forecasts_df : polars.DataFrame
        DataFrame containing forecasts from different models
    evaluation_df : polars.DataFrame
        DataFrame containing evaluation results and best model selection
        
    Returns:
    --------
    polars.DataFrame
        Processed DataFrame with best model forecasts
    """
    # First, detect the confidence interval level from column names
    interval_cols = [col for col in forecasts_df.columns if "-lo-" in col or "-hi-" in col]
    if interval_cols:
        confidence_level = interval_cols[0].split("-")[2]
    else:
        raise ValueError("No confidence interval columns found")
    
    # Create patterns for string replacement
    lo_pattern = f"-lo-{confidence_level}"
    hi_pattern = f"-hi-{confidence_level}"
    
    # Melt the forecasts dataframe
    df = (
        forecasts_df
        .melt(
            id_vars=["unique_id", "ds"],
            value_vars=forecasts_df.columns[2:],
            variable_name="model",
            value_name="best_model_forecast"
        )
        .join(
            evaluation_df[['unique_id', 'best_model']],
            on='unique_id',
            how="left"
        )
    )
    
    # Clean up model names and filter
    df = (
        df
        .with_columns(
            pl.col('model')
            .str.replace(f"{lo_pattern}|{hi_pattern}", "")
            .alias("clean_model")
        )
        .filter(pl.col('clean_model') == pl.col('best_model'))
        .drop('clean_model', 'best_model')
    )
    
    # Rename models to best_model while preserving confidence interval suffixes
    df = (
        df
        .with_columns(
            pl.when(pl.col('model').str.contains(f"(lo|hi)-{confidence_level}$"))
            .then(pl.concat_str([
                pl.lit('best_model'),
                pl.col('model').str.extract(f"(-(?:lo|hi)-{confidence_level})$")
            ]))
            .otherwise(pl.lit('best_model'))
            .alias('model')
        )
        .pivot(
            values='best_model_forecast',
            index=['unique_id', 'ds'],
            columns='model',
            aggregate_function='first'
        )
        .sort(by=['unique_id', 'ds'])
    )
    
    return df


###################################################################################
#######################           ML Forecast            ##########################
###################################################################################

def get_month_index(dates):
    """Extract month from date column using polars"""
    return dates.dt.month()

def create_ml_forecast(df: pl.DataFrame, 
                      active_filters: dict,
                      forecast_horizon: int,
                      forecast_confidence: int) -> tuple:
    """
    Create ML forecasts with advanced features using Polars
    
    Args:
        df (pl.DataFrame): Input dataframe with columns Date, Value and filter columns
        active_filters (dict): Dictionary of active filters to apply
        forecast_horizon (int): Number of periods to forecast
        forecast_confidence (int): Confidence level for predictions (e.g. 95)
    
    Returns:
        tuple: (grouped_df, forecasts_df, feature_importance)
    """
    # Validate inputs
    if df.height == 0:
        raise ValueError("Input DataFrame is empty")
        
    if not active_filters:
        raise ValueError("No active filters provided")
    
    # Prepare LightGBM models with different objectives
    lgb_params = {
        'verbosity': -1,
        'num_leaves': 32,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'force_col_wise': True
    }

    # Create models for average and confidence intervals
    models = {
        'avg': lgb.LGBMRegressor(**lgb_params),
        f'q{(100-forecast_confidence)//2}': lgb.LGBMRegressor(
            **lgb_params, 
            objective='quantile',
            alpha=(100-forecast_confidence)/200
        ),
        f'q{(100+forecast_confidence)//2}': lgb.LGBMRegressor(
            **lgb_params, 
            objective='quantile',
            alpha=(100+forecast_confidence)/200
        ),
    }

    # Prepare data with Polars
    filter_expr = pl.lit(True)
    for k, v in active_filters.items():
        filter_expr &= (pl.col(k) == v)
    uids_column = list(active_filters.keys())[-1] if active_filters else None

    # Filter and group data using Polars
    filtered_df = df.filter(filter_expr)
    
    if filtered_df.height == 0:
        raise ValueError("No data matches the provided filters")

    # Group and prepare data
    grouped_df = (
        filtered_df.group_by(["Date", uids_column])
        .agg(pl.col("Value").sum().alias("Value"))
        .rename({uids_column: "unique_id"})
        .sort("Date")
        .with_columns([
            pl.col("Date").alias("ds"),
            pl.col("Value").alias("y")
        ])
        .drop("Date", "Value")
    )

    # Validate minimum data requirements
    if grouped_df.height < 24:  # At least 2 years of monthly data
        raise ValueError(f"Not enough data points: {grouped_df.height} (minimum 24 required)")

    # Create MLForecast instance with key features
    fcst = MLForecast(
        models=models,
        freq='1mo',
        lags=[1, 2, 3, 6, 12],  # Important lags for monthly data
        lag_transforms={
            1: [
                ExpandingMean(),
                ExponentiallyWeightedMean(alpha=0.7)
            ],
            12: [
                RollingMean(window_size=12),
                RollingMean(window_size=3)
            ]
        },
        date_features=['month', 'quarter'],
        target_transforms=[
            Differences([12]),  # Remove yearly seasonality
            LocalStandardScaler()  # Normalize data
        ],
    )

    try:
        # Fit and predict
        fcst.fit(grouped_df)

        forecasts_df = fcst.predict(forecast_horizon)

        n_windows = 1
        h=6

        required_training_points = h * n_windows + 2  # Minimum d'entraînement requis pour chaque fenêtre

        # Vérification des dimensions pour éviter les erreurs
        if len(grouped_df) < required_training_points:
            raise ValueError(
                f"Not enough data points for n_windows={n_windows} and h={h}. "
                f"Available={len(grouped_df)}, required={required_training_points}."
            )

        crossvalidation_df = fcst.cross_validation(
            df=grouped_df,
            h=h, #min(forecast_horizon, len(grouped_df) // (n_windows + 1)),
            step_size=forecast_horizon,
            n_windows=min(n_windows, len(grouped_df) // forecast_horizon)
        )

        return (grouped_df, forecasts_df, crossvalidation_df)
                
    except Exception as e:
        raise RuntimeError(f"Error during forecast generation: {str(e)}") from e
    
###################################################################################
##################           Hierarchical Forecast            #####################
###################################################################################

def create_hierarchical_forecast(df: pl.DataFrame, active_filters: dict, 
                               forecast_horizon: int, forecast_confidence: int):
    """
    Create hierarchical forecasts based on selected filter levels using Polars.
    """
    # Define the complete hierarchy order and spec
    HIERARCHY_ORDER = ['ATC2', 'ATC3', 'ATC5', 'Product', 'CIP13']
    
    # Determine the lowest selected level
    selected_levels = [level for level in HIERARCHY_ORDER if level in active_filters]
    if not selected_levels:
        raise ValueError("No hierarchy levels selected")
    
    # Create specification for aggregation
    spec = []
    for i in range(len(selected_levels)):
        spec.append(selected_levels[:(i+1)])
    
    # Filter data based on active filters
    filter_expr = pl.lit(True)
    for k, v in active_filters.items():
        filter_expr &= (pl.col(k) == v)
    filtered_df = df.filter(filter_expr)
    
    # Prepare base DataFrame with required columns
    base_df = (
        filtered_df
        .select([
            pl.col("Date").alias("ds"),
            pl.col("Value").alias("y"),
            *selected_levels
        ])
    )

    base_df_pandas = base_df.to_pandas()

    print("base_df_pandas", base_df_pandas)

    # Create hierarchical structure using aggregate function
    Y_df, S_df, tags = aggregate(base_df_pandas, spec)
    Y_df = Y_df.reset_index()

    # print("Y_df", Y_df.shape)

    # print("spec:", spec)
    # print("Y_df", Y_df)
    # print("S_df", S_df)
    # print("tags", tags)
    # # Save the Y_df dictionnary
    # df_Y = pl.DataFrame(Y_df)
    # df_Y.write_csv('Y_df.csv')
    # # Save the S_df and tags
    # df_S = pl.DataFrame(S_df)
    # df_S.write_csv('S_df.csv')
    # tags_df = pl.DataFrame(tags)
    # tags_df.write_csv('tags.csv')
    
    # Split into train/test sets
    n_test = 3
    Y_test_df = Y_df.groupby('unique_id').tail(n_test)
    Y_train_df = Y_df.drop(Y_test_df.index)
    
    Y_test_df = Y_test_df.set_index('unique_id')
    Y_train_df = Y_train_df.set_index('unique_id')

    # print("test shape", Y_test_df)
    # print("train shape", Y_train_df)
    
    # Initialize StatsForecast with ETS model
    fcst = StatsForecast(
        df=Y_train_df,
        models=[
            AutoARIMA(season_length=n_test)
        ],
        freq='MS',
        n_jobs=-1
    )

    print(forecast_horizon)
    
    # Generate base forecasts
    Y_hat_df = fcst.forecast(h=3, fitted=True)
    Y_fitted_df = fcst.forecast_fitted_values()

    # print("Y_hat_df", Y_hat_df)
    # print("Y_fitted_df", Y_fitted_df)
    
    # Define reconciliation methods
    reconcilers = [
        BottomUp(),
        TopDown(method='forecast_proportions'),
        # MiddleOut(middle_level=selected_levels[len(selected_levels)//2], top_down_method='forecast_proportions')
    ]
    
    # Create reconciliation
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_fitted_df,
        S=S_df,
        tags=tags
    )
    
    # print("Y_rec_df", Y_rec_df)

    # Create evaluation tags for different levels
    eval_tags = {}
    for i, level in enumerate(selected_levels):
        level_keys = selected_levels[:(i+1)]
        level_tag = '/'.join(level_keys)
        eval_tags[level] = tags[level_tag]
    eval_tags['All'] = np.concatenate(list(tags.values()))

    # print("eval_tags", eval_tags)
    
    # Define evaluation metrics
    def rmse(y, y_hat):
        return np.mean(np.sqrt(np.mean((y-y_hat)**2, axis=1)))
    
    def mase(y, y_hat, y_insample, seasonality=12):
        errors = np.mean(np.abs(y - y_hat), axis=1)
        scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
        return np.mean(errors / scale)
    
    # Perform evaluation
    evaluator = HierarchicalEvaluation(evaluators=[rmse, mase])
    evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df,
        Y_test_df=Y_test_df,
        tags=eval_tags,
        Y_df=Y_train_df,
        benchmark='AutoARIMA'
    )

    # print("evaluation", evaluation)
    
    # Format evaluation results
    evaluation = evaluation.drop('Overall') if 'Overall' in evaluation.index else evaluation
    evaluation = evaluation.round(2)
    
    # Prepare results summary
    results_summary = {
        'hierarchy_levels': selected_levels,
        'n_series': len(Y_df['unique_id'].unique()),
        'evaluation': evaluation
    }
    
    # Convert reconciled forecasts to Polars
    Y_rec_df = pl.from_pandas(Y_rec_df.reset_index())
    Y_hier_df = pl.from_pandas(Y_df)
    
    return Y_hier_df, Y_rec_df, results_summary