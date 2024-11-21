import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define the directory containing xls files from Medic'AM
directory = Path(__file__).resolve().parents[1]
data_directory = directory/'data'
data_csv_directory = directory/'data'/'csv files'

# Define an empty list to collect dataframes
liste_dataframes = []

# Loop over specific sheet IDs : Community pharmacy + Hospital
for file in tqdm(os.listdir(str(data_directory))):
    # Loop over files in the specified directory
    for sheet_id in [2, 3]:
        # Check if the file has a .xls extension
        if file.endswith('.xls') and "CIP" not in file:
            
            # Construct the full path of the file
            full_path = os.path.join(str(data_directory), file)
            
            # Read the Excel file using pandas
            df = pd.read_excel(full_path, sheet_name=sheet_id)
            
            # Rename the columns of the dataframe
            df.rename(columns={df.columns[0]: "CIP13", 
                               df.columns[1]: "Designation", 
                               df.columns[2]: "Product",
                               df.columns[3]: "EphMRA",
                               df.columns[4]: "Class", 
                               df.columns[5]: "ATC_Code", 
                               df.columns[6]: "ATC_Class", 
                               df.columns[7]: "ATC_Code2", 
                               df.columns[8]: "ATC_Class2"}, inplace=True)
            
            # Split the dataframe into fixed and to-melt parts to pivot the dataframe
            df_fixed = df.iloc[:, :9]
            df_to_melt = df.iloc[:, 9:]
            
            # Melt the dataframe
            df_melted = df.melt(id_vars=df_fixed.columns, value_vars=df_to_melt.columns, var_name='Variable', value_name='Value')
            
            # Assign a market based on the sheet ID
            if sheet_id == 2:
                df_melted['Market_type'] = "Community"
            elif sheet_id == 3:
                df_melted['Market_type'] = "Hospital"
            
            # Add the melted dataframe to the list
            liste_dataframes.append(df_melted)

# Concatenate all the dataframes in the list
df_final = pd.concat(liste_dataframes, ignore_index=True)

# Extract date values
df_final['Month'] = pd.to_numeric(df_final['Variable'].str[-2:])
df_final['Year'] = pd.to_numeric(df_final['Variable'].str[-7:-3])
df_final['Date'] = pd.to_datetime(dict(year=df_final['Year'], month=df_final['Month'], day=1))
df_final['Variable'] = df_final['Variable'].str[:-7]

# Strip any leading or trailing whitespaces from the 'Variable' column
df_final['Variable'] = df_final['Variable'].replace("/n", "")
df_final['Variable'] = df_final['Variable'].str.strip()

# Replace French terms in the 'Variable' column with their English equivalents
df_final['Variable'] = df_final['Variable'].replace('Base de remboursement', 'Reimbursement_Base')
df_final['Variable'] = df_final['Variable'].replace('Nombre de boites remboursées', 'Number_Units')
df_final['Variable'] = df_final['Variable'].replace('Montant remboursé', 'Amount')

df_final = df_final[df_final['Variable'] == 'Amount']

# Focus on years 2021, 2022, 2023, 2024
df_final = df_final[df_final['Year'] >= 2021]

# Get product properties from df_final
df_product = df_final[['CIP13', 'Designation', 'Product', 'EphMRA', 'Class', 'ATC_Code', 'ATC_Class', 'ATC_Code2', 'ATC_Class2']]
df_product = df_product.drop_duplicates(subset=['CIP13'])
df_final = df_final[['CIP13', 'Variable', 'Value', 'Market_type', 'Month', 'Year', 'Date']]

# Clean the table - replace "Homéopathie" in CIP13 by 9999999999999 and remove total lines
df_final['CIP13'] = df_final['CIP13'].replace('Homéopathie ', 9999999999999)
df_final = df_final[df_final['CIP13'] != 'Total']
df_product['CIP13'] = df_product['CIP13'].replace('Homéopathie ', 9999999999999)
df_product = df_product[df_product['CIP13'] != 'Total']

df_product.columns = ['CIP13', 'Designation', 'Product', 'ATC3_code', 'ATC3', 'ATC5_code', 'ATC5', 'ATC2_code', 'ATC2']
df_product = df_product[['CIP13', 'Designation', 'Product', 'ATC2_code', 'ATC2', 'ATC3_code', 'ATC3', 'ATC5_code', 'ATC5']]
# Capitalize the first letter of each word in the 'Product' column
for i in ['Product', 'Designation', 'ATC2', 'ATC3', 'ATC5']:
    df_product[i] = df_product[i].str.capitalize()


# Add the equivalence table between CIP13 and CIS
df_equiv = pd.read_csv(str(data_csv_directory) + '/Correspondance-UCD-CIP-CIS.csv', sep=",", encoding='utf-8')
# Merge this file with CIP_list
df_product = pd.merge(df_product, df_equiv, how='left', on='CIP13')
# Add the CIS dataset with vendor names
df_cis = pd.read_csv(str(data_csv_directory) + '/CIS.csv', sep=";", encoding='utf-8')
# Merge this file with CIP_list
df_product = pd.merge(df_product, df_cis, how='left', on='CIS')

# Save the 2 dataframes
df_final.to_csv(str(data_directory) + '/French_pharmaceutical_sales.csv', sep=";", index=False, encoding='utf-8')
df_product.to_csv(str(data_directory) + '/CIP_list.csv', sep=";", index=False, encoding='utf-8')
print('Job done')
