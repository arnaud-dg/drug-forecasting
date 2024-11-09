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
        .filter(pl.col('ATC2').is_not_null())
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

            print(evaluation_df)

            prod_forecasts_df = get_best_model_forecast(forecasts_df, evaluation_df)

            fig = create_combined_forecast_plot(grouped_df, prod_forecasts_df)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw prediction data"):
                # Display the name of the best model and the associated metric value
                st.write("The best statistical model for this time-serie is: ", evaluation_df[0, "best_model"], " with a MSE of: ", evaluation_df[0, evaluation_df[0, "best_model"]])
                st.write(prod_forecasts_df)

        elif selected_filters['model_type'] == 'MLForecast':

            grouped_df, crossvalidation_df, forecasts_df, prod_forecasts_df, evaluation_df, feature_importance = create_ml_forecast(
                df=df,
                active_filters=active_filters,
                forecast_horizon=forecast_horizon,
                forecast_confidence=forecast_confidence
            )

            print(grouped_df)

            print(prod_forecasts_df)

            fig = create_combined_forecast_plot_ML(grouped_df, prod_forecasts_df)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Raw prediction data"):
                best_model = evaluation_df[0, "best_model"]
                mse_value = evaluation_df[0, best_model]
                
                st.write(f"The best ML model for this time-series is: {best_model}")
                st.write(f"MSE: {mse_value:.2f}")
                
                # Feature importance if fcst is provided and using LightGBM
                if fcst is not None and hasattr(fcst.models.get(best_model), 'feature_importances_'):
                    st.write("\nFeature Importance:")
                    importances = pd.DataFrame({
                        'feature': fcst.feature_names,
                        'importance': fcst.models[best_model].feature_importances_
                    }).sort_values('importance', ascending=False)
                    st.dataframe(importances)
                
                st.write("\nPredictions:")
                st.write(prod_forecasts_df)

        else:
            st.warning("Please select at least one hierarchy level to display data.")

if __name__ == "__main__":
    main()