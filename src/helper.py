"""
Drug sales Forecasting is a project dedicated to apply time series forecasting 
to predict the sales of pharmaceutical products in France.
This project is based on the Nixtla library, which provides a unified interface.

This module hcontains generic functions used along the other different files and 
constants for the Drug Sales Forecasting project.

Author: Arnaud Duigou
Date : Nov 2024
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import polars as pl
from datetime import datetime

# Constants
HIERARCHY_LEVELS = [
    ('ATC2', 'Select ATC2'),
    ('ATC3', 'Select ATC3'),
    ('ATC5', 'Select ATC5'),
    ('Product', 'Select Product'),
    ('CIP13', 'Select CIP13')
]

HIERARCHY_ORDER = ['ATC2', 'ATC3', 'ATC5', 'Product', 'CIP13']
EXCLUDED_FILTER_COLUMNS = ['Market_type', 'model_type', 'horizon', 'confidence']

class DataLoader:
    """Class for handling data loading and initial processing."""
    
    @staticmethod
    def load_data(data_directory: Path) -> pl.DataFrame:
        """
        Load and prepare sales data with CIP information.
        
        Args:
            data_directory (Path): Directory containing data files
            
        Returns:
            pl.DataFrame: Prepared DataFrame containing sales data
            
        Raises:
            FileNotFoundError: If required data files are not found
        """
        print("Loading data...")  # Debug cache
        
        try:
            # Load datasets
            cip = pl.read_csv(
                str(data_directory / "CIP_list.csv"), 
                separator=";", 
                truncate_ragged_lines=True, 
                schema_overrides={"CIP13": pl.Utf8}
            )

            sales = pl.read_csv(
                str(data_directory / "French_pharmaceutical_sales.csv"), 
                separator=";", 
                truncate_ragged_lines=True, 
                schema_overrides={"CIP13": pl.Utf8}
            )

            # Merge and process data
            df = (
                sales.join(cip, on="CIP13", how="left")
                .with_columns([
                    pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d'),
                    pl.col('Value').cast(pl.Float64)
                ])
                .filter(
                    (pl.col('ATC2').is_not_null()) & 
                    (pl.col('actif') == True)
                )
            )
            
            # Ensure column names are consistent
            required_cols = ['Date', 'Value', 'CIP13', 'ATC2', 'ATC3', 'ATC5', 'Product']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

class DataValidator:
    """Utility class for data validation operations."""
    
    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """
        Validate DataFrame structure and content.
        
        Args:
            df (pl.DataFrame): DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Vérifier si le DataFrame est vide
        if df is None or df.height == 0:
            raise ValueError("DataFrame is empty")
            
        # Vérifier la présence des colonnes requises
        required_cols = {'Date', 'Value', 'CIP13', 'ATC2'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Vérifier les valeurs nulles dans les colonnes critiques
        null_counts = {}
        for col in required_cols:
            if col in df.columns:
                null_count = df.select(pl.col(col).null_count()).item()
                if null_count > 0:
                    null_counts[col] = null_count
        
        if null_counts:
            raise ValueError(
                f"Found null values in critical columns: {null_counts}"
            )
            
    @staticmethod
    def validate_filters(filters: Dict[str, Any]) -> None:
        """
        Validate filter dictionary structure and content.
        
        Args:
            filters (Dict[str, Any]): Filter dictionary to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")
            
        required_keys = {'model_type', 'Market_type', 'horizon', 'confidence'}
        missing_keys = required_keys - set(filters.keys())
        if missing_keys:
            raise ValueError(f"Missing required filter keys: {missing_keys}")
            
        # Validate specific filter values
        if filters['horizon'] not in range(3, 13):
            raise ValueError("Forecast horizon must be between 3 and 12 months")
            
        if filters['confidence'] not in {80, 90, 95}:
            raise ValueError("Confidence level must be 80, 90, or 95")

class DataProcessor:
    """Class for common data processing operations."""
    
    @staticmethod
    def get_active_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract active filters from filter dictionary.
        
        Args:
            filters (Dict[str, Any]): Complete filter dictionary
            
        Returns:
            Dict[str, Any]: Dictionary containing only active filters
        """
        return {
            k: v for k, v in filters.items()
            if v and k not in EXCLUDED_FILTER_COLUMNS 
            and k != 'Market_type'
        }
    
    @staticmethod
    def prepare_filtered_data(
        df: pl.DataFrame,
        filters: Dict[str, Any]
    ) -> Optional[pl.DataFrame]:
        """
        Prepare and filter data based on selected filters.
        
        Args:
            df (pl.DataFrame): Input DataFrame
            filters (Dict[str, Any]): Filter dictionary
            
        Returns:
            Optional[pl.DataFrame]: Processed DataFrame or None if invalid
        """
        try:
            # Vérifier que les colonnes requises existent
            required_cols = ['Date', 'Value']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

            # Apply market type filter
            filtered_df = df
            if filters.get('Market_type') != "Both":
                filtered_df = filtered_df.filter(
                    pl.col("Market_type") == filters['Market_type']
                )
                
            # Apply active filters
            active_filters = DataProcessor.get_active_filters(filters)
            if not active_filters:
                return None
                
            # Construire l'expression de filtrage
            filter_expr = pl.lit(True)
            for k, v in active_filters.items():
                if k in df.columns:  # Vérifier que la colonne existe
                    filter_expr &= (pl.col(k) == v)
                
            filtered_df = filtered_df.filter(filter_expr)
            
            if filtered_df.height == 0:
                return None
                
            # Prepare for forecasting
            processed_df = (
                filtered_df
                .group_by('Date')
                .agg(pl.col('Value').sum().alias('Value'))
                .sort('Date')
                .with_columns([
                    pl.lit('series1').alias('unique_id'),
                    pl.col('Date').alias('ds'),
                    pl.col('Value').alias('y')
                ])
            )
            
            return processed_df
            
        except Exception as e:
            print(f"Error in prepare_filtered_data: {str(e)}")
            print(f"Available columns: {df.columns}")
            return None
    
    @staticmethod
    def get_unique_column_values(
        df: pl.DataFrame,
        column: str
    ) -> List[str]:
        """
        Get unique values for a specific column.
        
        Args:
            df (pl.DataFrame): Input DataFrame
            column (str): Column name
            
        Returns:
            List[str]: Sorted list of unique values
        """
        return sorted(df.select(column).unique().to_series().to_list())

class TimeSeriesHelper:
    """Helper class for time series operations."""
    
    @staticmethod
    def get_date_range(dates: pl.Series) -> Tuple[datetime, datetime]:
        """
        Get min and max dates from a series.
        
        Args:
            dates (pl.Series): Series of dates
            
        Returns:
            Tuple[datetime, datetime]: (min_date, max_date)
        """
        return dates.min(), dates.max()
    
    @staticmethod
    def generate_year_breaks(
        start_date: datetime,
        end_date: datetime
    ) -> List[str]:
        """
        Generate list of January dates between start and end dates.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            List[str]: List of dates in YYYY-01-01 format
        """
        years = range(start_date.year, end_date.year + 1)
        return [f"{year}-01-01" for year in years]