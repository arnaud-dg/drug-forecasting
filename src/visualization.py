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
        min_date = data['Date'].min()
        max_date = data['Date'].max()
        min_year = min_date.year
        max_year = max_date.year
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

def prepare_plot_data(dataframe: pl.DataFrame, filters: dict) -> pl.DataFrame:
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
    
    return plot_data

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
    min_date = data['Date'].min()
    max_date = data['Date'].max()
    min_year = min_date.year
    max_year = max_date.year
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

def create_combined_forecast_plot(grouped_df, prod_forecasts_df) -> go.Figure:
    """
    Create a plotly figure combining historical and forecasted values with dynamic confidence interval detection
    
    Parameters:
    -----------
    grouped_df : polars.DataFrame
        Historical data with columns ['ds', 'unique_id', 'y']
    prod_forecasts_df : polars.DataFrame
        Forecast data with dynamic confidence interval columns
    """
    # Detect confidence interval level
    interval_cols = [col for col in prod_forecasts_df.columns if "-lo-" in col or "-hi-" in col]
    if interval_cols:
        confidence_level = interval_cols[0].split("-")[2]
        hi_col = f"best_model-hi-{confidence_level}"
        lo_col = f"best_model-lo-{confidence_level}"
    else:
        raise ValueError("No confidence interval columns found")
    
    # Create figure
    fig = go.Figure()
    
    # Add historical values with styled markers
    fig.add_trace(
        go.Scatter(
            x=grouped_df['ds'],
            y=grouped_df['y'],
            name="Historical values",
            mode='lines+markers',
            line=dict(
                color='#3366CC',
                width=2
            ),
            marker=dict(
                size=6,
                color='white',
                line=dict(
                    width=2,
                    color='#3366CC'
                )
            )
        )
    )
    
    # Add forecasted values with styled markers
    fig.add_trace(
        go.Scatter(
            x=prod_forecasts_df['ds'],
            y=prod_forecasts_df['best_model'],
            name="Forecast",
            mode='lines+markers',
            line=dict(
                color='#9933CC',
                width=2
            ),
            marker=dict(
                size=6,
                color='white',
                line=dict(
                    width=2,
                    color='#9933CC'
                )
            )
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=prod_forecasts_df['ds'].to_list() + prod_forecasts_df['ds'].to_list()[::-1],
            y=prod_forecasts_df[hi_col].to_list() + 
              prod_forecasts_df[lo_col].to_list()[::-1],
            fill='toself',
            fillcolor='rgba(153, 51, 204, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level}% Confidence Interval',
            showlegend=True
        )
    )
    
    # Add vertical lines for January of each year
    min_date = grouped_df['ds'].min()
    max_date = prod_forecasts_df['ds'].max()
    for year in range(min_date.year, max_date.year + 1):
        fig.add_vline(
            x=f"{year}-01-01",
            line_width=1,
            line_color='rgb(230, 230, 230)'
        )
    
    # Update layout
    fig.update_layout(
        title=f"Sales Forecast for {grouped_df['unique_id'][0]}",
        yaxis_title="Sales",
        hovermode='x unified',
        showlegend=True,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            showgrid=False,
            dtick="M1",
            tickangle=90,
            tickformat='%Y-%m',
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgb(240, 240, 240)',
            tickformat=".2e"
        ),
        plot_bgcolor='white',
        margin=dict(t=50, l=50, r=20, b=100)
    )
    
    return fig

###################################################################################
#################                 ML Forecast               #######################
###################################################################################

def create_combined_forecast_plot_ML(grouped_df, forecast_df, forecast_confidence):
    """
    Create a plotly figure combining historical and forecasted values
    
    Parameters:
    -----------
    grouped_df : polars.DataFrame
        Historical data with columns ['ds', 'unique_id', 'y']
    forecast_df : polars.DataFrame
        Forecast data with columns ['ds', 'unique_id', 'avg', 'q5', 'q95']
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    
    if forecast_confidence == 90:
        upper_col = 'q95'
        lower_col = 'q5'
    elif forecast_confidence == 80:
        upper_col = 'q90'
        lower_col = 'q10'
    elif forecast_confidence == 95:
        upper_col = 'q97'
        lower_col = 'q2'
    
    # Create figure
    fig = go.Figure()
    
    # Add historical values
    fig.add_trace(
        go.Scatter(
            x=grouped_df['ds'],
            y=grouped_df['y'],
            name="Historical values",
            mode='lines+markers',
            line=dict(
                color='#3366CC',
                width=2
            ),
            marker=dict(
                size=6,
                color='white',
                line=dict(
                    width=2,
                    color='#3366CC'
                )
            )
        )
    )
    
    # Add forecasted values
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['avg'],
            name="Forecast",
            mode='lines+markers',
            line=dict(
                color='#9933CC',
                width=2
            ),
            marker=dict(
                size=6,
                color='white',
                line=dict(
                    width=2,
                    color='#9933CC'
                )
            )
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'].to_list() + forecast_df['ds'].to_list()[::-1],
            y=forecast_df[upper_col].to_list() + forecast_df[lower_col].to_list()[::-1],
            fill='toself',
            fillcolor='rgba(153, 51, 204, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Interval',
            showlegend=True
        )
    )
    
    # Add vertical lines for January of each year
    min_date = grouped_df['ds'].min()
    max_date = forecast_df['ds'].max()
    for year in range(min_date.year, max_date.year + 1):
        fig.add_vline(
            x=f"{year}-01-01",
            line_width=1,
            line_color='rgb(230, 230, 230)'
        )
    
    # Format title to use product name
    product_name = grouped_df['unique_id'][0].replace('_', ' ').title()
    
    # Update layout
    fig.update_layout(
        title=f"Sales Forecast for {product_name}",
        yaxis_title="Sales",
        hovermode='x unified',
        showlegend=True,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            showgrid=False,
            dtick="M2",  # Show every 2 months
            tickangle=90,
            tickformat='%Y-%m',
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgb(240, 240, 240)',
            tickformat=".2s"  # Use SI prefix formatting for better readability
        ),
        plot_bgcolor='white',
        margin=dict(t=50, l=50, r=20, b=100)
    )
    
    return fig

###################################################################################
#################            Hierarchical Forecast          #######################
###################################################################################

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