import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    HoltWinters, 
    SeasonalNaive,
    DynamicOptimizedTheta
)
# from helper import prepare_data, forecast_series, add_prediction_intervals, visualize_forecasts, evaluate_models, optimize_parameters

# Load the datasets
cip = pl.read_csv("CIP_list.csv", 
                  separator=";", 
                  truncate_ragged_lines=True, 
                  dtypes={"CIP13": pl.Utf8},
                  null_values=["Homéopathie"])
sales = pl.read_csv("French_pharmaceutical_sales.csv", 
                    separator=";", 
                    truncate_ragged_lines=True, 
                    dtypes={"CIP13": pl.Utf8},
                    null_values=["Homéopathie"])

# Merge the datasets
df = cip.join(sales, on="CIP13", how="inner")

print(df.head())

print("Column names and data types:")
print(df.schema)

# 1. Préparer et faire les prédictions
forecasts, fitted_values, train_df = forecast_series(df, horizon=12)

# # 2. Visualiser les résultats pour une série spécifique
# unique_id = df['unique_id'].unique()[0]  # Premier ID comme exemple
# fig = visualize_forecasts(train_df, forecasts, unique_id)
# fig.show()

# # 3. Évaluer les modèles
# models = ['AutoARIMA', 'HoltWinters', 'SeasonalNaive', 'DynamicOptimizedTheta']
# evaluation = evaluate_models(train_df, test_df, fitted_values, models)
# print("\nÉvaluation des modèles:")
# for model, metrics in evaluation.items():
#     print(f"\n{model}:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.2f}")

# # 4. Optimiser les paramètres pour une série spécifique
# best_params, best_scores = optimize_parameters(df, unique_id)
# print("\nMeilleurs paramètres:")
# for model, params in best_params.items():
#     print(f"\n{model}:")
#     print(f"Paramètres: {params}")
#     print(f"Score MAE: {best_scores[model]['MAE']:.2f}")