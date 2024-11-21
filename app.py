"""
Drug sales Forecasting is a project dedicated to apply time series forecasting 
to predict the sales of pharmaceutical products in France.
This project is based on the Nixtla library, which provides a unified interface.

Main Streamlit application for sales forecasting dashboard.
This module provides a web interface for visualizing and forecasting sales data
with various filtering options and model selections.

Author: Arnaud Duigou
Date : Nov 2024
"""

# Import libraries and dependances
import streamlit as st
st.set_page_config(page_title="Sales Dashboard", layout="wide")

import polars as pl
import plotly.express as px
from pathlib import Path
from datetime import datetime
from src.visualization import display_simple_viewer, display_seasonal_decomposition, create_combined_forecast_plot, create_combined_forecast_plot_ML
from src.forecasting import evaluate_cross_validation, get_best_model_forecast, create_ml_forecast
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    # DynamicOptimizedTheta as DOT,
    SeasonalNaive
)
from utilsforecast.losses import mse
# from utilsforecast.evaluation import evaluate

from datasetsforecast.hierarchical import HierarchicalData

# compute base forecast no coherent
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse

#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
from hierarchicalforecast.utils import aggregate

# Get the current directory
DIRECTORY = Path(__file__).resolve().parents[0]
DATA_DIRECTORY = DIRECTORY/'data'

###################################################################################
#######################           Data Loading           ##########################
###################################################################################

@st.cache_data()
def load_data() -> pl.DataFrame:
    """
    Load and prepare data with caching.

    Returns:
        pl.DataFrame: Prepared DataFrame containing sales data
    """
    print("Loading data...")  # Debug cache
    
    # Load datasets
    cip = pl.read_csv(
        str(DATA_DIRECTORY) + "/CIP_list.csv", 
        separator=";", 
        truncate_ragged_lines=True, 
        schema_overrides={"CIP13": pl.Utf8}
    )

    sales = pl.read_csv(
        str(DATA_DIRECTORY) + "/French_pharmaceutical_sales.csv", 
        separator=";", 
        truncate_ragged_lines=True, 
        schema_overrides={"CIP13": pl.Utf8}
    )
    
    # Merge and process data
    df = (
        sales.join(cip, on="CIP13", how="left")
        .with_columns(pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d'))
        .filter(
            (pl.col('ATC2').is_not_null()) & 
            (pl.col('actif') == True)
            )
    )
    
    return df

###################################################################################
###################           Side Bar & Filters            #######################
###################################################################################

def get_unique_column_values(df: pl.DataFrame, column: str) -> list:
    """
    Get unique values for a specific column from DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame
        column (str): Column name

    Returns:
        list: Sorted list of unique values
    """
    return sorted(df.select(column).unique().to_series().to_list())

def create_hierarchy_filters(df: pl.DataFrame) -> dict:
    """
    Create hierarchical filters for the dashboard.

    Args:
        df (pl.DataFrame): Input DataFrame

    Returns:
        dict: Dictionary containing selected filter values
    """
    HIERARCHY_LEVELS = [
        ('ATC2', 'Select ATC2'),
        ('ATC3', 'Select ATC3'),
        ('ATC5', 'Select ATC5'),
        ('Product', 'Select Product'),
        ('CIP13', 'Select CIP13')
    ]
    
    filters = {}
    filtered_df = df

    # Load the icon of the webapp
    st.sidebar.image(str(DIRECTORY/'assets'/'drug_sales_forecasting_icone.png'), width=250)
    
    st.sidebar.title("Filters")
    
    # Model selection
    model_type = st.sidebar.selectbox("Forecast Model",
        options=["Simple viewer", "Seasonal decomposition", "statsforecast", "MLForecast", "HierarchicalForecast"], index=0)
    filters['model_type'] = model_type
    st.sidebar.markdown("---")

    # Market type selection
    market_type = st.sidebar.radio("Market Type",options=["Both", "Hospital", "Community"], index=0)
    if market_type != "Both":
        filtered_df = filtered_df.filter(pl.col("Market_type") == market_type)
    filters['Market_type'] = market_type
    st.sidebar.markdown("---")
    
    # Create hierarchical filters
    for level, label in HIERARCHY_LEVELS:
        if level == 'ATC2' or all(filters[prev] for prev, _ in HIERARCHY_LEVELS[:HIERARCHY_LEVELS.index((level, label))]):
            options = get_unique_column_values(filtered_df, level)
            
            # Auto-select if only one option is available
            if len(options) == 1:
                selected = options[0]
                st.sidebar.selectbox(label, options=[selected], index=0, disabled=True)
            else:
                selected = st.sidebar.selectbox(
                    label,
                    options=[''] + options,
                    index=0
                )
            
            filters[level] = selected
            
            if selected:
                filtered_df = filtered_df.filter(pl.col(level) == selected)
            else:
                break
    st.sidebar.markdown("---")

    # Forecast horizon selection
    forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", min_value=3, max_value=12, value=6)
    filters['horizon'] = forecast_horizon
    forecast_confidence = st.sidebar.selectbox("Forecast Confidence Level", options=[80, 90, 95], index=1)
    filters['confidence'] = forecast_confidence
    st.sidebar.markdown("---")
                
    return filters

###################################################################################
###################                  Main                   #######################
###################################################################################

def create_hierarchical_forecast(df: pl.DataFrame, active_filters: dict, 
                               forecast_horizon: int, forecast_confidence: int):
    """
    Create hierarchical forecasts based on selected filter levels using Polars.
    """
    # Define the complete hierarchy order
    HIERARCHY_ORDER = ['ATC2', 'ATC3', 'ATC5', 'Product', 'CIP13']
    
    # Determine the lowest selected level
    selected_levels = [level for level in HIERARCHY_ORDER if level in active_filters]
    if not selected_levels:
        raise ValueError("No hierarchy levels selected")
    
    lowest_selected_level = selected_levels[-1]
    hierarchy_index = HIERARCHY_ORDER.index(lowest_selected_level)
    
    # Create the active hierarchy levels based on selection
    active_hierarchy = HIERARCHY_ORDER[:hierarchy_index + 1]
    
    # Filter data based on active filters
    filter_expr = pl.lit(True)
    for k, v in active_filters.items():
        filter_expr &= (pl.col(k) == v)
    filtered_df = df.filter(filter_expr)
    
    # Create hierarchy levels list for aggregation
    hierarchy_levels = []
    for i in range(len(active_hierarchy)):
        hierarchy_levels.append(active_hierarchy[:(i+1)])
    
    # Create hierarchical structure for each level
    hierarchical_dfs = []
    
    for level_cols in hierarchy_levels:
        level_df = (
            filtered_df
            .group_by(['Date'] + level_cols)
            .agg(pl.col("Value").sum().alias("Value"))
            .sort("Date")
        )
        
        # Create unique_id and select only necessary columns
        level_df = level_df.with_columns([
            pl.concat_str(level_cols, separator="/").alias("unique_id")
        ]).select(['Date', 'Value', 'unique_id'])
        
        hierarchical_dfs.append(level_df)
    
    # Combine all hierarchical levels
    Y_hier_df = pl.concat(hierarchical_dfs)
    
    # Create bottom level identifiers for S matrix
    bottom_level_df = (
        filtered_df
        .group_by(['Date'] + active_hierarchy)
        .agg(pl.col("Value").sum().alias("Value"))
        .sort("Date")
        .with_columns([
            pl.concat_str(active_hierarchy, separator="/").alias("unique_id")
        ])
    )
    
    bottom_ids = bottom_level_df.select("unique_id").unique().sort("unique_id")
    
    # Create S matrix
    S_rows = []
    aggregation_ids = []
    
    # Get all aggregation level IDs for each level
    for level_cols in hierarchy_levels:
        level_df = (
            filtered_df
            .group_by(level_cols)
            .agg(pl.count('Value'))
            .with_columns([
                pl.concat_str(level_cols, separator="/").alias("unique_id")
            ])
        )
        
        aggregation_ids.extend(level_df.select('unique_id').to_series().to_list())
    
    # Create S matrix rows
    for agg_id in aggregation_ids:
        agg_parts = agg_id.split("/")
        row = []
        
        for bottom_id in bottom_ids["unique_id"]:
            bottom_parts = bottom_id.split("/")
            is_part = all(ap == bp for ap, bp in zip(agg_parts, bottom_parts[:len(agg_parts)]))
            row.append(1.0 if is_part else 0.0)
        
        S_rows.append(row)
    
    S_df = pl.DataFrame(
        S_rows,
        schema={f"col_{i}": pl.Float64 for i in range(len(bottom_ids))}
    )
    
    # Prepare data for StatsForecast
    Y_hier_df = (
        Y_hier_df
        .rename({"Date": "ds", "Value": "y"})
        .with_columns([
            pl.col("ds").cast(pl.Date)
        ])
    )
    
    # Split into train/test sets using window functions
    n_test = 3  # Number of test periods
    
    # Create a window size for each unique_id
    Y_test_df = (
        Y_hier_df
        .group_by('unique_id')
        .agg([
            pl.col('ds').sort_by('ds').tail(n_test).alias('ds'),
            pl.col('y').sort_by('ds').tail(n_test).alias('y')
        ])
        .explode(['ds', 'y'])
    )
    
    # Create train set by excluding test data
    Y_train_df = Y_hier_df.join(
        Y_test_df,
        on=['unique_id', 'ds'],
        how='anti'
    )
    
    # Create forecasts with monthly frequency
    fcst = StatsForecast(
        df=Y_train_df,
        models=[
            AutoARIMA(season_length=12),
            Naive()
        ],
        freq='1M',  # Changed from '1mo' to '1M'
        n_jobs=-1
    )
    
    # Generate predictions
    Y_hat_df = fcst.forecast(h=forecast_horizon)
    Y_fitted_df = fcst.forecast_fitted_values()
    
    # Select appropriate reconciliation methods
    reconcilers = [BottomUp()]
    
    if len(active_hierarchy) > 2:
        reconcilers.extend([
            TopDown(method='forecast_proportions'),
            MiddleOut(
                middle_level='/'.join(active_hierarchy[:(len(active_hierarchy)//2)]),
                top_down_method='forecast_proportions'
            )
        ])
    
    # Create tags dictionary for each level of hierarchy
    tags = {}
    for i, level in enumerate(active_hierarchy):
        level_cols = active_hierarchy[:i+1]
        level_df = (
            filtered_df
            .group_by(level_cols)
            .agg(pl.count('Value'))
            .with_columns([
                pl.concat_str(level_cols, separator="/").alias("unique_id")
            ])
        )
        tags[level] = level_df.select('unique_id').to_series().to_list()
    
    # Perform reconciliation
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_fitted_df,
        S=S_df,
        tags=tags
    )
    
    # Convert reconciled forecasts to Polars
    Y_rec_df = pl.from_pandas(Y_rec_df.reset_index())
    
    # Create evaluation metrics
    evaluator = HierarchicalEvaluation(
        evaluators=[
            lambda y, y_hat: np.mean((y - y_hat) ** 2),  # MSE
            lambda y, y_hat: np.sqrt(np.mean((y - y_hat) ** 2)),  # RMSE
            lambda y, y_hat: np.mean(np.abs((y - y_hat) / y)) * 100  # MAPE
        ]
    )
    
    evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df,
        Y_test_df=Y_test_df.set_index("unique_id"),
        tags=tags,
        benchmark='Naive'
    )
    
    results_summary = {
        'hierarchy_levels': active_hierarchy,
        'n_series': Y_hier_df.select('unique_id').n_unique(),
        'evaluation': evaluation
    }
    
    return Y_hier_df, Y_rec_df, results_summary

def create_combined_forecast_plot_hierarchical(Y_hier_df, Y_rec_df, results_summary):
    """
    Create a plotly figure showing actual vs forecasted values for each level
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    hierarchy_levels = results_summary['hierarchy_levels']
    n_levels = len(hierarchy_levels)
    
    fig = make_subplots(
        rows=n_levels,
        cols=1,
        subplot_titles=[f"Level: {level}" for level in hierarchy_levels],
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, level in enumerate(hierarchy_levels, 1):
        level_series = Y_hier_df.filter(pl.col('unique_id').str.contains(level))
        level_forecasts = Y_rec_df.filter(pl.col('unique_id').str.contains(level))
        
        # Actual values
        fig.add_trace(
            go.Scatter(
                x=level_series['ds'],
                y=level_series['y'],
                name=f'Actual - {level}',
                line=dict(color=colors[0]),
                showlegend=(i==1)
            ),
            row=i, col=1
        )
        
        # Forecasted values
        fig.add_trace(
            go.Scatter(
                x=level_forecasts['ds'],
                y=level_forecasts['AutoARIMA/BottomUp'],
                name=f'Forecast - {level}',
                line=dict(color=colors[1], dash='dash'),
                showlegend=(i==1)
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=300 * n_levels,
        title_text="Hierarchical Forecasts by Level",
        showlegend=True
    )
    
    return fig

def main():
    """
    Main function to run the Streamlit application.
    """
    # Load data
    df = load_data()

    # Create filters
    selected_filters = create_hierarchy_filters(df)
    forecast_horizon = selected_filters['horizon']
    forecast_confidence = selected_filters['confidence']

    # Check if at least one filter is selected (excluding special filters)
    has_active_filters = any(
        val for key, val in selected_filters.items() 
        if key not in ['Market_type', 'model_type', 'horizon', 'confidence']
    )
    active_filters = {
        k: v for k, v in selected_filters.items()
        if k in df.columns and v != '' and not (k == 'Market_type' and v == 'Both')
    }

    if has_active_filters:
        # Handle different visualization modes
        if selected_filters['model_type'] == 'Simple viewer':
            fig = display_simple_viewer(df, selected_filters)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_filters['model_type'] == 'Seasonal decomposition':
            fig = display_seasonal_decomposition(df, selected_filters)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_filters['model_type'] == 'statsforecast':
            models = [
                HoltWinters(),
                Croston(),
                SeasonalNaive(season_length=12),
                HistoricAverage(),
                # DOT(season_length=12)
            ]

            # Instantiate StatsForecast class as sf
            sf = StatsForecast( 
                models=models,
                freq='1mo', 
                n_jobs=-1,
                fallback_model=SeasonalNaive(season_length=12),
                verbose=True
            )

            # Applied selected_filters on df
            filter_expr = pl.lit(True)  # Initialiser avec une expression True pour chaîner
            for k, v in active_filters.items():
                filter_expr &= (pl.col(k) == v)
            uids_column = list(active_filters.keys())[-1] if active_filters else None

            # Appliquer le filtre
            filtered_df = df.filter(filter_expr)
            grouped_df = (
                filtered_df.group_by(["Date", uids_column])
                .agg(pl.col("Value").sum().alias("Value"))  # Somme des valeurs
                .rename({uids_column: "uids"})  # Renommer la dernière clé en "uids"
                .sort("Date")
            )
            grouped_df = grouped_df.rename({"Date": "ds", "uids": "unique_id", "Value": "y"})

            crossvalidation_df = sf.cross_validation(
                df=grouped_df,
                h=forecast_horizon,
                step_size=forecast_horizon,
                n_windows=3
            )

            forecasts_df = sf.forecast(df=grouped_df, h=forecast_horizon, level=[forecast_confidence])
            
            # Evaluate the forecasts
            evaluation_df = evaluate_cross_validation(crossvalidation_df, mse)

            prod_forecasts_df = get_best_model_forecast(forecasts_df, evaluation_df)

            fig = create_combined_forecast_plot(grouped_df, prod_forecasts_df)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw prediction data"):
                # Display the name of the best model and the associated metric value
                st.write("The best statistical model for this time-serie is: ", evaluation_df[0, "best_model"], " with a MSE of: ", evaluation_df[0, evaluation_df[0, "best_model"]])
                st.write(prod_forecasts_df)

        elif selected_filters['model_type'] == 'MLForecast':

            grouped_df, forecasts_df, crossvalidation_df = create_ml_forecast(
                df=df,
                active_filters=active_filters,
                forecast_horizon=forecast_horizon,
                forecast_confidence=forecast_confidence
            )

            print(crossvalidation_df)

            cv_rmse = evaluate(
                crossvalidation_df.drop('cutoff'),
                metrics=[rmse],
                agg_fn='mean',
            )
            print(cv_rmse)

            fig = create_combined_forecast_plot_ML(grouped_df, forecasts_df, forecast_confidence)

            # fig = create_combined_forecast_plot_ML(grouped_df, prod_forecasts_df)
            st.plotly_chart(fig, use_container_width=True)
            # with st.expander("Raw prediction data"):
            #     best_model = evaluation_df[0, "best_model"]
            #     mse_value = evaluation_df[0, best_model]
                
            #     st.write(f"The best ML model for this time-series is: {best_model}")
            #     st.write(f"MSE: {mse_value:.2f}")
                
            #     # Feature importance if fcst is provided and using LightGBM
            #     # if fcst is not None and hasattr(fcst.models.get(best_model), 'feature_importances_'):
            #     #     st.write("\nFeature Importance:")
            #     #     importances = pd.DataFrame({
            #     #         'feature': fcst.feature_names,
            #     #         'importance': fcst.models[best_model].feature_importances_
            #     #     }).sort_values('importance', ascending=False)
            #     #     st.dataframe(importances)
                
            #     st.write("\nPredictions:")
            #     st.write(prod_forecasts_df)
        elif selected_filters['model_type'] == 'HierarchicalForecast':
            try:
                Y_hier_df, Y_rec_df, results_summary = create_hierarchical_forecast(
                    df=df,
                    active_filters=active_filters,
                    forecast_horizon=forecast_horizon,
                    forecast_confidence=forecast_confidence
                )
                
                # Create and display the plot
                fig = create_combined_forecast_plot_hierarchical(Y_hier_df, Y_rec_df, results_summary)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results summary in an expander
                with st.expander("Forecast Details"):
                    st.write("Hierarchy Levels:", ", ".join(results_summary['hierarchy_levels']))
                    st.write("Number of Series:", results_summary['n_series'])
                    
                    st.write("\nEvaluation Metrics by Level:")
                    eval_df = results_summary['evaluation']
                    st.dataframe(eval_df)
                    
                    # Show raw forecasts
                    st.write("\nRaw Forecasts:")
                    st.dataframe(Y_rec_df)
                    
            except Exception as e:
                st.error(f"Error in hierarchical forecasting: {str(e)}")
            
        else:
            st.warning("Please select at least one hierarchy level to display data.")

if __name__ == "__main__":
    main()