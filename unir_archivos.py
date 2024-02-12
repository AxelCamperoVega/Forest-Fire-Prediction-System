# Fires Forest Prediction - Data integration
# Axel Campero Vega
# THIS PROGRAM JOINS THE .CSV FILES FROM THE YEAR 2001-2020 INTO A SINGLE FILE AND COPIES
# IT TO THE DATA FOLDER. ALSO COPY THE .CSV FILE FOR THE YEAR 2021 TO THE DATA FOLDER.

# Import pandas library
import pandas as pd

# Read_data function: Read the data from the .csv files in the modis_bolivia folder
def leer_datos(year):
    ubicacion_archivo = f"modis_bolivia/modis_{year}_Bolivia.csv"
    return pd.read_csv(ubicacion_archivo)

# Reads the data from the years 2001 to 2020,
# concatenates them and saves them in a single file data_2001-2020.csv
datos_global = pd.concat(list(map(leer_datos, range(2001, 2021))), axis=0)
datos_global.to_csv('data/datos_2001-2020.csv', index=False)

# Read the data for the year 2021 and save it in the data_2021.csv file
datos_2021 = leer_datos(2021)
datos_2021.to_csv('data/datos_2021.csv', index=False)
