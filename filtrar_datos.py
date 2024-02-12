# Fires Forest Prediction - Data Fitering
# Axel Campero Vega
# THIS PROGRAM FILTERS RECORDS THAT HAVE CONFIDENCE > 90 AND ARE TYPE 0 (FIRES)
# Import pandas library
import pandas as pd

# Read the files generated with the "unir_files.py" program 
# and save them in a variable of the data frame type
datos_global = pd.read_csv('data/datos_2001-2020.csv')     #Read complete data 2001-2020
datos_2021 = pd.read_csv('data/datos_2021.csv')            #Read full data 2021

# Filter records from the file "data_2001-2020.csv" and save them in "filtered_2001-2020.csv"
filtro_alta_confianza = datos_global['confidence'] > 90                           #Filter confidence > 90%
filtro_incendios = datos_global['type'] == 0                                      #Filter Type == 0 (fires)
datos_historico_filtrado = datos_global[filtro_alta_confianza & filtro_incendios] #Filter data
datos_historico_filtrado.to_csv('data/filtrado_2001-2020.csv', index=False)       #Save filtered data 2001-2020

# Filter records from the file "datos_2021.csv" and save them in "filfilado_2021.csv"
filtro_alta_confianza = datos_2021['confidence'] > 90                             #Filter confidence > 90%
filtro_incendios = datos_2021['type'] == 0                                        #Filter Type == 0 (fires)
datos_incendios_actual = datos_2021[filtro_alta_confianza & filtro_incendios]     #Filter data
datos_incendios_actual.to_csv('data/filtrado_2021.csv', index=False)              #Save fitered data 2021