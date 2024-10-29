import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    HoltWinters, 
    SeasonalNaive,
    DynamicOptimizedTheta
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict

def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prépare les données pour le forecasting
    """
    # Création d'un identifiant unique pour chaque série
    df = df.with_columns([
        pl.concat_str([
            pl.col('Product'),
            pl.col('Market_type')
        ], separator='/').alias('unique_id'),
        
        pl.col('Date').str.strptime(pl.Datetime, fmt='%Y-%m-%d').alias('ds'),
        pl.col('Value').alias('y')
    ])
    
    # Sélection des colonnes nécessaires au format StatsForecast
    forecast_df = df.select(['unique_id', 'ds', 'y'])
    
    return forecast_df

def forecast_series(df: pl.DataFrame, horizon: int = 12, season_length: int = 12):
    """
    Effectue les prédictions avec plusieurs modèles
    """
    # Conversion en format StatsForecast
    forecast_df = prepare_data(df)
    
    # Définition des modèles
    models = [
        AutoARIMA(season_length=season_length),
        HoltWinters(season_length=season_length),
        SeasonalNaive(season_length=season_length),
        DynamicOptimizedTheta(season_length=season_length)
    ]
    
    # Création de l'objet StatsForecast
    fcst = StatsForecast(
        df=forecast_df,
        models=models,
        freq='M',  # Fréquence mensuelle
        n_jobs=-1  # Utilise tous les cores disponibles
    )
    
    # Split train/test (optionnel, pour validation)
    cutoff_date = forecast_df.select(pl.col('ds').max()).item() 
    train_df = forecast_df.filter(pl.col('ds') <= cutoff_date)
    
    # Prédictions
    forecasts = fcst.forecast(h=horizon)
    fitted_values = fcst.forecast_fitted_values()
    
    return forecasts, fitted_values, train_df

def add_prediction_intervals(forecasts: pl.DataFrame, level: list = [90]):
    """
    Ajoute des intervalles de prédiction pour les modèles qui le supportent
    """
    fcst = StatsForecast(
        df=train_df,
        models=models,
        freq='M',
        n_jobs=-1
    )
    
    forecasts_with_intervals = fcst.forecast(h=horizon, level=level)
    return forecasts_with_intervals

def visualize_forecasts(
    train_df: pl.DataFrame,
    forecasts: pl.DataFrame,
    unique_id: str,
    models: List[str] = None
):
    """
    Visualise les prédictions pour une série spécifique
    """
    # Filtrer les données pour la série spécifique
    train_series = train_df.filter(pl.col('unique_id') == unique_id)
    forecast_series = forecasts.filter(pl.col('unique_id') == unique_id)
    
    # Créer le graphique
    fig = go.Figure()
    
    # Ajouter les données historiques
    fig.add_trace(
        go.Scatter(
            x=train_series['ds'],
            y=train_series['y'],
            name='Historique',
            mode='lines',
            line=dict(color='black')
        )
    )
    
    # Couleurs pour les différents modèles
    colors = ['blue', 'red', 'green', 'purple']
    
    # Ajouter les prédictions de chaque modèle
    if models is None:
        models = [col for col in forecast_series.columns if col not in ['unique_id', 'ds']]
    
    for model, color in zip(models, colors):
        # Prédictions
        fig.add_trace(
            go.Scatter(
                x=forecast_series['ds'],
                y=forecast_series[model],
                name=f'Prédiction {model}',
                mode='lines',
                line=dict(color=color, dash='dash')
            )
        )
        
        # Intervalles de confiance si disponibles
        lo_col = f'{model}-lo-90'
        hi_col = f'{model}-hi-90'
        if lo_col in forecast_series.columns and hi_col in forecast_series.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_series['ds'].to_list() + forecast_series['ds'].to_list()[::-1],
                    y=forecast_series[hi_col].to_list() + forecast_series[lo_col].to_list()[::-1],
                    fill='toself',
                    fillcolor=f'rgba({",".join(map(str, [int(x*255) for x in plt.colors.to_rgb(color)]))},0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Intervalle de confiance 90% - {model}'
                )
            )
    
    fig.update_layout(
        title=f'Prédictions pour {unique_id}',
        xaxis_title='Date',
        yaxis_title='Valeur',
        height=600,
        showlegend=True
    )
    
    return fig

def evaluate_models(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    fitted_values: pl.DataFrame,
    models: List[str]
) -> Dict:
    """
    Évalue la performance des modèles
    """
    results = {}
    
    for model in models:
        # Calcul des métriques sur les valeurs ajustées
        mae = mean_absolute_error(
            test_df['y'],
            fitted_values[model]
        )
        rmse = np.sqrt(mean_squared_error(
            test_df['y'],
            fitted_values[model]
        ))
        mape = np.mean(np.abs((test_df['y'] - fitted_values[model]) / test_df['y'])) * 100
        
        results[model] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    return results

def optimize_parameters(
    df: pl.DataFrame,
    unique_id: str,
    season_lengths: List[int] = [4, 12, 52],
    test_size: int = 12
):
    """
    Optimise les paramètres des modèles pour une série spécifique
    """
    best_params = {}
    best_scores = {}
    
    # Filtrer les données pour la série spécifique
    series_df = df.filter(pl.col('unique_id') == unique_id)
    
    # Split train/test
    test_df = series_df.tail(test_size)
    train_df = series_df.head(len(series_df) - test_size)
    
    for season_length in season_lengths:
        models = [
            AutoARIMA(season_length=season_length),
            HoltWinters(season_length=season_length),
            SeasonalNaive(season_length=season_length),
            DynamicOptimizedTheta(season_length=season_length)
        ]
        
        # Créer et entraîner StatsForecast
        fcst = StatsForecast(
            df=train_df,
            models=models,
            freq='M',
            n_jobs=-1
        )
        
        # Faire des prédictions
        forecasts = fcst.forecast(h=test_size)
        
        # Évaluer les performances
        for model in models:
            model_name = model.__class__.__name__
            mae = mean_absolute_error(test_df['y'], forecasts[model_name])
            
            if model_name not in best_scores or mae < best_scores[model_name]['MAE']:
                best_scores[model_name] = {'MAE': mae}
                best_params[model_name] = {'season_length': season_length}
    
    return best_params, best_scores