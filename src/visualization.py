"""
Drug sales Forecasting is a project dedicated to apply time series forecasting 
to predict the sales of pharmaceutical products in France.
This project is based on the Nixtla library, which provides a unified interface.

this module contains visualization-related functions for the sales dashboard.
This module handles data preparation and visualization settings for the sales
dashboard, including filtering logic and plot title generation.

Author: Arnaud Duigou
Date : Nov 2024
"""

# Import libraries and dependances
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose

###################################################################################
#######################           Basic chart            ##########################
###################################################################################

class PlotSettings:
    """
    Class containing plot configuration settings.
    """
    DEFAULT_LAYOUT = {
        'yaxis_title': "Sales",
        'hovermode': 'x unified',
        'showlegend': True,
        'height': 600,  # Increased height
        'xaxis': {
            'showgrid': False,
            'dtick': "M1",  # Show ticks for each month
            'tickangle': 90,  # Rotate x labels
            'tickformat': '%Y-%m',
            'title': None  # Remove x-axis label
        }
    }
    
    @staticmethod
    def create_time_series_plot(data: pd.DataFrame, title: str) -> go.Figure:
        """
        Create a time series plot using plotly express.

        Args:
            data (pd.DataFrame): Data to plot
            title (str): Plot title

        Returns:
            go.Figure: Plotly figure object
        """
        # Create base figure
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Value'],
                mode='lines+markers',
                line=dict(
                    color='#3366CC',  # Couleur de la ligne
                    width=2
                ),
                marker=dict(
                    size=6,
                    color='white',
                    line=dict(
                        width=2,
                        color='#3366CC'
                    )
                ),
                showlegend=False  # Désactive l'affichage de la légende
            )
        )
        
        # Add vertical lines for January
        min_year = data['Date'].dt.year.min()
        max_year = data['Date'].dt.year.max()
        for year in range(min_year, max_year + 1):
            fig.add_vline(
                x=datetime(year, 1, 1),
                line_width=1,
                line_color='rgb(230, 230, 230)'
            )
        
        # Update layout with default settings and title
        layout = PlotSettings.DEFAULT_LAYOUT.copy()
        layout['title'] = title
        fig.update_layout(**layout)
        
        return fig

def generate_plot_title(filters: dict) -> str:
    """
    Generate plot title based on the most specific selected filter.

    Args:
        filters (dict): Dictionary containing filter selections

    Returns:
        str: Generated plot title
    """
    # Build market type suffix
    market_suffix = f" ({filters['Market_type']})" if filters['Market_type'] != "Both" else ""
    model_suffix = f" - {filters['model_type']}"
    
    # Hierarchy levels from most specific to least specific
    HIERARCHY_LEVELS = ['CIP13', 'Product', 'ATC5', 'ATC3', 'ATC2']
    
    # Find the most specific selected level
    for level in HIERARCHY_LEVELS:
        if filters.get(level):
            return f"Sales for {level}: {filters[level]}{market_suffix}{model_suffix}"
    
    return "Please select at least one hierarchy level"

def prepare_plot_data(dataframe: pl.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Prepare data for plotting based on selected filters.

    Args:
        dataframe (pl.DataFrame): Input DataFrame containing sales data
        filters (dict): Dictionary containing filter selections

    Returns:
        pd.DataFrame: Processed DataFrame ready for plotting, or None if no valid filters
    """
    filtered_df = dataframe
    
    # List of columns to exclude from filtering
    EXCLUDED_COLUMNS = ['Market_type', 'model_type', 'horizon']
    
    # Apply market type filter if specified
    if filters['Market_type'] != "Both":
        filtered_df = filtered_df.filter(pl.col("Market_type") == filters['Market_type'])
    
    # Create filter conditions for active filters
    active_filters = [
        pl.col(col) == val 
        for col, val in filters.items() 
        if val and col not in EXCLUDED_COLUMNS and col in dataframe.columns
    ]
    
    if not active_filters:
        return None
        
    # Group and aggregate data
    plot_data = (
        filtered_df.filter(pl.all_horizontal(active_filters))
        .group_by('Date')
        .agg(pl.col('Value').sum())
        .sort('Date')
    )
    
    # Convert to pandas with proper date handling
    pandas_df = plot_data.to_pandas()
    pandas_df['Date'] = pandas_df['Date'].dt.to_pydatetime()
    pandas_df['Date'] = np.array(pandas_df['Date'])
    
    return pandas_df

def display_simple_viewer(df: pl.DataFrame, filters: dict) -> go.Figure:
    """
    Display simple time series visualization.

    Args:
        df (pl.DataFrame): Input DataFrame
        filters (dict): Selected filters
    """
    plot_data = prepare_plot_data(df, filters)
    
    if plot_data is not None:
        fig = PlotSettings.create_time_series_plot(
            data=plot_data,
            title=generate_plot_title(filters)
        )
        return fig

###################################################################################
#################           Seasonal decomposition          #######################
###################################################################################

def create_seasonal_decomposition_plot(data: pd.DataFrame, title: str) -> go.Figure:
    """
    Create a seasonal decomposition plot using plotly with three subplots.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Date' and 'Value' columns
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure with seasonal decomposition
    """
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(
        data['Value'],
        period=12,  # Monthly data
        extrapolate_trend='freq'
    )
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, 
        cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.05,
        shared_xaxes=True,
        x_title='Date'
    )
    
    # Add original data
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Value'],
            mode='lines',
            name='Original',
            line=dict(color='#3366CC', width=2)
        ),
        row=1, col=1
    )
    
    # Add trend
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=decomposition.trend,
            mode='lines',
            name='Trend',
            line=dict(color='#DC3912', width=2)
        ),
        row=2, col=1
    )
    
    # Add seasonal
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=decomposition.seasonal,
            mode='lines',
            name='Seasonal',
            line=dict(color='#FF9900', width=2)
        ),
        row=3, col=1
    )
    
    # Add residual
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=decomposition.resid,
            name='Residual',
            marker_color=np.where(decomposition.resid >= 0, '#109618', '#DC3912'),
            marker_line_width=0,
            showlegend=False
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,  # Increased height for better visibility
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        hovermode='x unified',
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)
    
    # Add vertical lines for January of each year
    min_year = data['Date'].dt.year.min()
    max_year = data['Date'].dt.year.max()
    for year in range(min_year, max_year + 1):
        for row in range(1, 5):
            fig.add_vline(
                x=pd.Timestamp(f"{year}-01-01"),
                line_width=1,
                line_color='rgb(230, 230, 230)',
                row=row,
                col=1
            )
    
    return fig

def display_seasonal_decomposition(df: pl.DataFrame, filters: dict) -> go.Figure:
    """
    Display seasonal decomposition visualization in Streamlit.
    
    Args:
        df (pl.DataFrame): Input DataFrame
        filters (dict): Selected filters
    """
    plot_data = prepare_plot_data(df, filters)
    
    if plot_data is not None:
        # Check if we have enough data for seasonal decomposition
        if len(plot_data) < 24:  # Need at least 2 years of data
            print("Seasonal decomposition requires at least 2 years of data.")
            return None
            
        fig = create_seasonal_decomposition_plot(
            data=plot_data,
            title=generate_plot_title(filters)
        )
        
        return fig
        

###################################################################################
#################                Statsforecast              #######################
###################################################################################

def create_forecast_plot(forecast_data: pd.DataFrame, title: str) -> go.Figure:
    """
    Create a forecast plot with confidence intervals.

    Args:
        forecast_data (pd.DataFrame): DataFrame containing forecasts and intervals
        title (str): Plot title

    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.line(
        forecast_data, 
        x='Date',
        y=['Value', 'Forecast', 'Lower_CI', 'Upper_CI'],
        title=title
    )
    
    # Update layout with default settings
    fig.update_layout(**PlotSettings.DEFAULT_LAYOUT)
    
    return fig



def display_forecast_view(df: pl.DataFrame, filters: dict, forecast_horizon: int) -> go.Figure:
    """
    Display forecast visualization with statistics.

    Args:
        df (pl.DataFrame): Input DataFrame
        filters (dict): Selected filters
        forecast_horizon (int): Number of periods to forecast
    """
    forecast_data = ForecastModelManager.generate_forecasts(
        df, 
        filters, 
        forecast_horizon=forecast_horizon
    )
    
    if forecast_data is not None:
        # Create forecast plot using the new function
        fig = create_forecast_plot(
            forecast_data,
            generate_plot_title(filters)
        )
        
        return fig



