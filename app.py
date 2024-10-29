import streamlit as st
# Configuration de la page
st.set_page_config(page_title="Sales Dashboard", layout="wide")

import polars as pl
import plotly.express as px
from datetime import datetime
# from src.helper import prepare_data, forecast_series, add_prediction_intervals, visualize_forecasts, evaluate_models, optimize_parameters

# Fonction pour charger et préparer les données
@st.cache_data
def load_data():
    # Load the datasets
    cip = pl.read_csv("data/CIP_list.csv", 
                    separator=";", 
                    truncate_ragged_lines=True, 
                    dtypes={"CIP13": pl.Utf8})
    
    # Supprimer la ligne cotenant la valeur 'Total' dans la colonne 'CIP13'
    cip = cip.filter(pl.col('CIP13') != 'Total')
    # Remplacer la valeur 'HomÃ©opathie' par 9999999999999 dans la colonne 'CIP13'
    cip = cip.select(
        pl.col('CIP13').str.replace("HomÃ©opathie ", "9999999999999").alias('CIP13')
    )

    sales = pl.read_csv("data/French_pharmaceutical_sales.csv", 
                        separator=";", 
                        truncate_ragged_lines=True, 
                        dtypes={"CIP13": pl.Utf8})
    # Merge the datasets
    df = sales.join(cip, on="CIP13", how="left")
    # Convertir la colonne Date en datetime
    df = df.with_columns(
        pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d').alias('Date')
    )

    df = df.filter(pl.col('Variable') == 'Reimbursement_Base')

    # Affichage des colonnes et de leur type
    print("Column names and data types:")
    print(df.schema)

    # Remplacer les valeurs nulles par "Non catégorisé"
    df = df.with_columns([
        pl.col('Level1').fill_null('Non catégorisé'),
        pl.col('Level2').fill_null('Non catégorisé'),
        pl.col('Level3').fill_null('Non catégorisé'),
        pl.col('Product').fill_null('Non catégorisé')
    ])

    return df

# Chargement des données
df = load_data()

# Enregistre le csv 
df.write_csv('data/df.csv')

# Affiche les colonnes et les types de données du dataframe
# print("Column names and data types:")
# print(df.schema)

# Sidebar pour les filtres
st.sidebar.title("Filtres")

# Initialisation des variables de sélection
selected_level1 = None
selected_level2 = None
selected_level3 = None
selected_product = None
selected_cip13 = None

# Sélection Level 1
all_level1 = sorted(df.select('Level1').unique().to_series().to_list())
selected_level1 = st.sidebar.selectbox('Sélectionner Level 1', 
                                     options=[''] + all_level1,
                                     index=0)

# Filtrage et affichage conditionnel des niveaux suivants
if selected_level1:
    # Filtrer les données pour Level 2
    df_filtered = df.filter(pl.col('Level1') == selected_level1)
    all_level2 = sorted(df_filtered.select('Level2').unique().to_series().to_list())
    selected_level2 = st.sidebar.selectbox('Sélectionner Level 2',
                                         options=[''] + all_level2,
                                         index=0)
    
    if selected_level2:
        # Filtrer les données pour Level 3
        df_filtered = df_filtered.filter(pl.col('Level2') == selected_level2)
        all_level3 = sorted(df_filtered.select('Level3').unique().to_series().to_list())
        selected_level3 = st.sidebar.selectbox('Sélectionner Level 3',
                                             options=[''] + all_level3,
                                             index=0)
        
        if selected_level3:
            # Filtrer les données pour Product
            df_filtered = df_filtered.filter(pl.col('Level3') == selected_level3)
            all_products = sorted(df_filtered.select('Product').unique().to_series().to_list())
            selected_product = st.sidebar.selectbox('Sélectionner Produit',
                                                  options=[''] + all_products,
                                                  index=0)
            
            if selected_product:
                # Filtrer les données pour CIP13
                df_filtered = df_filtered.filter(pl.col('Product') == selected_product)
                all_cip13 = sorted(df_filtered.select('CIP13').unique().to_series().to_list())
                selected_cip13 = st.sidebar.selectbox('Sélectionner CIP13',
                                                    options=[''] + all_cip13,
                                                    index=0)

# Préparation des données pour le graphique
def prepare_plot_data(df, level1, level2=None, level3=None, product=None, cip13=None):
    # Commencer avec le filtre Level1
    filters = [pl.col('Level1') == level1]
    
    # Ajouter les autres filtres si nécessaire
    if level2:
        filters.append(pl.col('Level2') == level2)
    if level3:
        filters.append(pl.col('Level3') == level3)
    if product:
        filters.append(pl.col('Product') == product)
    if cip13:
        filters.append(pl.col('CIP13') == cip13)
    
    # Appliquer tous les filtres et agréger les données
    plot_data = (df.filter(pl.all_horizontal(filters))
                 .groupby('Date')
                 .agg(pl.col('Sales').sum())
                 .sort('Date'))
    
    # Convertir en DataFrame pandas pour Plotly
    return plot_data.to_pandas()

# Création du graphique
if selected_level1:
    plot_data = prepare_plot_data(df, selected_level1, selected_level2, 
                                selected_level3, selected_product, selected_cip13)
    
    # Déterminer le titre du graphique
    if selected_cip13:
        title = f"Ventes pour CIP13: {selected_cip13}"
    elif selected_product:
        title = f"Ventes pour Produit: {selected_product}"
    elif selected_level3:
        title = f"Ventes pour Level 3: {selected_level3}"
    elif selected_level2:
        title = f"Ventes pour Level 2: {selected_level2}"
    else:
        title = f"Ventes pour Level 1: {selected_level1}"
    
    # Création du graphique avec Plotly
    fig = px.line(plot_data, 
                 x='Date', 
                 y='Sales',
                 title=title)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Ventes",
        hovermode='x unified'
    )
    
    # Affichage du graphique
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Veuillez sélectionner au moins un niveau de la hiérarchie pour afficher les données.")

# Ajout d'informations supplémentaires
if selected_level1:
    with st.expander("Statistiques descriptives"):
        stats_data = plot_data['Sales'].describe()
        st.write(stats_data)