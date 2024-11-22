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
from src.visualization import display_simple_viewer, display_seasonal_decomposition, create_combined_forecast_plot, create_combined_forecast_plot_ML,create_combined_forecast_plot_hierarchical
from src.forecasting import evaluate_cross_validation, get_best_model_forecast, create_ml_forecast, create_hierarchical_forecast
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



# Get the current directory
DIRECTORY = Path(__file__).resolve().parents[0]
DATA_DIRECTORY = DIRECTORY/'data'

# Inject custom CSS
st.markdown("""
    <style>
    /* Divise par 2 la hauteur de la div située en haut de la sidebar */
    div[data-testid="stSidebarHeader"] {
        height: 3%; /* Divise la hauteur par 2 */
    }
    /* Supprime le texte des selectbox <hr> */
    label[data-testid="stWidgetLabel"] {
        display: none; /* Masquer complètement l'élément */
        height: 0;     /* S'assurer qu'il ne prend pas de place */
    }
    /* Réduire la hauteur du conteneur de <hr> */
    div[data-testid="stElementContainer"] hr {
        margin: 2px 0; /* Réduire les marges */
        border-width: 1px; /* Réduire l'épaisseur de la ligne */
    }
    /* Réduction du padding des titres */
    h3[level="3"] {
        padding: 0.5rem 0px 0.5rem !important; /* Réduction du padding */
    }
    .st-emotion-cache-1jicfl2.ea3mdgi5 {
        padding: 3rem 1rem 10rem !important; /* Applique le padding */
    }
    </style>
    """, unsafe_allow_html=True)

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

    df.write_csv(str(DATA_DIRECTORY) + "/df.csv")
    
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
    Create hierarchical filters for the dashboard with improved layout.

    Args:
        df (pl.DataFrame): Input DataFrame

    Returns:
        dict: Dictionary containing selected filter values
    """
    filters = {}
    filtered_df = df

    # Load the icon of the webapp
    st.sidebar.image(str(DIRECTORY/'assets'/'drug_sales_forecasting_icone.png'), width=250)
    
    # Introduction text
    st.sidebar.markdown("""
        *Drug sales forecasting is a web application that provides sales forecasts for drugs or 
        drug families using various time series forecasting algorithms.*
        <br> 
        *The data is sourced from the Medic'AM database provided by the French Health Insurance.*""", 
        unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Product hierarchy selection
    st.sidebar.markdown("""
        <h3 style='color: #082F9C;'>Product Selection</h3>
    """, unsafe_allow_html=True)
    HIERARCHY_LEVELS = [
        ('ATC2', 'Select an ATC2 category'),
        ('ATC3', 'Select ATC3'),
        ('ATC5', 'Select ATC5'),
        ('Product', 'Select Product'),
        ('CIP13', 'Select CIP13')
    ]
    
    # Create container for selectboxes to maintain consistent spacing
    hierarchy_container = st.sidebar.container()
    
    for level, label in HIERARCHY_LEVELS:
        if level == 'ATC2' or all(filters[prev] for prev, _ in HIERARCHY_LEVELS[:HIERARCHY_LEVELS.index((level, label))]):
            options = get_unique_column_values(filtered_df, level)
            
            # Auto-select if only one option is available
            if len(options) == 1:
                selected = options[0]
                hierarchy_container.selectbox(
                    label, 
                    options=[selected], 
                    index=0, 
                    disabled=True,
                    key=f"hierarchy_{level}"
                )
            else:
                selected = hierarchy_container.selectbox(
                    label,
                    options=[''] + options,
                    index=0,
                    key=f"hierarchy_{level}"
                )
            
            filters[level] = selected
            
            if selected:
                filtered_df = filtered_df.filter(pl.col(level) == selected)
            else:
                # Réserver l'espace pour les selectbox restants même s'ils sont désactivés
                for remaining_level, remaining_label in HIERARCHY_LEVELS[HIERARCHY_LEVELS.index((level, label))+1:]:
                    hierarchy_container.selectbox(
                        remaining_label,
                        options=[''],
                        disabled=True,
                        index=0,
                        key=f"hierarchy_{remaining_level}_disabled",
                        placeholder=f"Select {remaining_level} if needed",
                        label_visibility='hidden'
                    )
                break

    st.sidebar.markdown("---")

    # Market type selection with reduced spacing
    st.sidebar.markdown("""
        <h3 style='color: #082F9C;'>Market type</h3>
    """, unsafe_allow_html=True)
    market_type = st.sidebar.radio(
        "",  # Empty label since we used write() above
        options=["Both", "Hospital", "Pharmacy"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    if market_type == "Pharmacy":
        market_type='Community'
    if market_type != "Both":
        filtered_df = filtered_df.filter(pl.col("Market_type") == market_type)
    filters['Market_type'] = market_type
    
    st.sidebar.markdown("---")

    # View type selection
    st.sidebar.markdown("""
        <h3 style='color: #082F9C;'>Forecasting tools</h3>
    """, unsafe_allow_html=True)
    model_type = st.sidebar.selectbox(
        "",  # Empty label since we used write() above
        options=[
            "Simple viewer",
            "Seasonal trends Decomposition",
            "Statistically based Forecasts",
            "Machine Learning based Forecasts",
            "Hierarchical Forecasts"
        ],
        index=0,
        label_visibility="collapsed"
    )
    filters['model_type'] = model_type
    
    # Show forecast parameters only for forecasting models
    if model_type in ["Statistically based Forecasts", "Machine Learning based Forecasts", "Hierarchical Forecasts"]:
        with st.sidebar.expander("Parameters", expanded=True):
            st.write("Forecasting Horizon (months):")
            forecast_horizon = st.slider(
                "Forecast Horizon (months)",
                min_value=3,
                max_value=12,
                value=6,
                help="Number of months to forecast"
            )
            st.write("Forecasting Confidence Level:")
            forecast_confidence = st.selectbox(
                "Confidence Level (%)",
                options=[80, 90, 95],
                index=1,
                help="Confidence interval for the forecast"
            )
            
            filters['horizon'] = forecast_horizon
            filters['confidence'] = forecast_confidence
    else:
        # Set default values for non-forecasting views
        filters['horizon'] = 6
        filters['confidence'] = 90
    
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

        elif selected_filters['model_type'] == 'Seasonal trends Decomposition':
            fig = display_seasonal_decomposition(df, selected_filters)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_filters['model_type'] == 'Statistically based Forecasts':
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

        elif selected_filters['model_type'] == 'Machine Learning based Forecasts':

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
            st.plotly_chart(fig, use_container_width=True)

        elif selected_filters['model_type'] == 'Hierarchical Forecasts':

            Y_hier_df, Y_rec_df, results_summary = create_hierarchical_forecast(
                df=df,
                active_filters=active_filters,
                forecast_horizon=forecast_horizon,
                forecast_confidence=forecast_confidence
            )

            print("Y_hier_df", Y_hier_df)
            print("Y_rec_df", Y_rec_df)
            print("results_summary", results_summary)
            
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
            
        else:
            st.warning("Please select at least one hierarchy level to display data.")

if __name__ == "__main__":
    main()