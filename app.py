import streamlit as st
st.set_page_config(page_title="Sales Dashboard", layout="wide")

import polars as pl
import plotly.express as px
import numpy as np
from datetime import datetime
from pathlib import Path
from src.helper import prepare_plot_data, get_plot_title

# Get the current directory
directory = Path(__file__).resolve().parents[0]
data_directory = directory/'data'

@st.cache_data()
def load_data():
    """
    Load and prepare data with caching
    """
    print("Loading data...")  # Pour débugger le cache
    
    # Load the datasets
    cip = pl.read_csv(str(data_directory) + "/CIP_list.csv", 
                    separator=";", 
                    truncate_ragged_lines=True, 
                    dtypes={"CIP13": pl.Utf8})

    Value = pl.read_csv(str(data_directory) + "/French_pharmaceutical_sales.csv", 
                        separator=";", 
                        truncate_ragged_lines=True, 
                        dtypes={"CIP13": pl.Utf8})
    # Merge the datasets
    df = Value.join(cip, on="CIP13", how="left")
    df = df.with_columns(
        pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d').alias('Date')
    )
    df = df.filter(pl.col('ATC2').is_not_null())
    return df

def get_unique_values_from_column(_df: pl.DataFrame, column: str) -> list:
    """
    Get unique values for a specific column from DataFrame
    Using _df to indicate to Streamlit not to hash the DataFrame
    """
    return sorted(_df.select(column).unique().to_series().to_list())

def create_hierarchy_filters(df):
    """
    Create hierarchical filters for the dashboard with auto-selection of single options
    """
    hierarchy_levels = [
        ('ATC2', 'Sélectionner ATC2'),
        ('ATC3', 'Sélectionner ATC3'),
        ('ATC5', 'Sélectionner ATC5'),
        ('Product', 'Sélectionner Produit'),
        ('CIP13', 'Sélectionner CIP13')
    ]
    
    filters = {}
    filtered_df = df
    
    st.sidebar.title("Filtres")
    
    # Sélection du modèle de prévision
    model_type = st.sidebar.selectbox(
        "Modèle de prévision",
        options=["statsforecast", "MLForecast", "HierarchicalForecast", "Seasonal decomposition"],
        index=0,
    )
    filters['model_type'] = model_type
    
    st.sidebar.markdown("---")
    
    # Radio button pour Market_type
    market_type = st.sidebar.radio(
        "Type de Marché",
        options=["Both", "Hospital", "Community"],
        index=0,
    )
    
    # Appliquer le filtre de Market_type si nécessaire
    if market_type != "Both":
        filtered_df = filtered_df.filter(pl.col("Market_type") == market_type)
    filters['Market_type'] = market_type
    
    st.sidebar.markdown("---")
    
    # Parcourir chaque niveau de la hiérarchie
    for level, label in hierarchy_levels:
        if level == 'ATC2' or all(filters[prev] for prev, _ in hierarchy_levels[:hierarchy_levels.index((level, label))]):
            options = get_unique_values_from_column(filtered_df, level)
            
            # Si une seule option est disponible, la sélectionner automatiquement
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
                
    return filters

# Main application
df = load_data()

# Create filters
selected_filters = create_hierarchy_filters(df)

# Create visualization if at least one filter is selected
if any(val for key, val in selected_filters.items() if key not in ['Market_type', 'model_type']):
    plot_data = prepare_plot_data(df, selected_filters)
    
    fig = px.line(plot_data, 
                 x='Date', 
                 y='Value',
                 title=get_plot_title(selected_filters))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Ventes",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    with st.expander("Statistiques descriptives"):
        stats_data = plot_data['Value'].describe()
        st.write(stats_data)
else:
    st.write("Veuillez sélectionner au moins un niveau de la hiérarchie pour afficher les données.")