import polars as pl
import plotly.express as px
import numpy as np
from datetime import datetime
from pathlib import Path

#########################################################################################
#######################            Graphical functions             ######################
#########################################################################################

def prepare_plot_data(_df: pl.DataFrame, filters: dict):
    """
    Prepare data for plotting based on selected filters
    Using _df to indicate to Streamlit not to hash the DataFrame
    """
    filtered_df = _df
    
    # Liste des colonnes à exclure du filtrage
    excluded_columns = ['Market_type', 'model_type', 'horizon']
    
    # Appliquer le filtre de Market_type
    if filters['Market_type'] != "Both":
        filtered_df = filtered_df.filter(pl.col("Market_type") == filters['Market_type'])
    
    # Appliquer les autres filtres
    active_filters = [pl.col(col) == val for col, val in filters.items() 
                     if val and col not in excluded_columns and col in _df.columns]
    
    if not active_filters:
        return None
        
    plot_data = (filtered_df.filter(pl.all_horizontal(active_filters))
                 .group_by('Date')
                 .agg(pl.col('Value').sum())
                 .sort('Date'))
    
    # Conversion en pandas avec traitement correct des dates
    pandas_df = plot_data.to_pandas()
    pandas_df['Date'] = pandas_df['Date'].dt.to_pydatetime()
    pandas_df['Date'] = np.array(pandas_df['Date'])
    
    return pandas_df

def get_plot_title(filters):
    """
    Generate plot title based on the most specific selected filter
    """
    market_type_text = f" ({filters['Market_type']})" if filters['Market_type'] != "Both" else ""
    model_text = f" - {filters['model_type']}"
    
    for level in ['CIP13', 'Product', 'ATC5', 'ATC3', 'ATC2']:
        if filters.get(level):
            return f"Ventes pour {level}: {filters[level]}{market_type_text}{model_text}"
    return "Veuillez sélectionner au moins un niveau de la hiérarchie"





































































