#Script funciones

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

# Función describe_df


def resumen_dataframe(df):
    # Crear un diccionario para almacenar la información
    resumen = {
        'Tipo de Dato': df.dtypes,
        '% Valores Nulos': df.isnull().mean() * 100,
        'Valores Únicos': df.nunique(),
        '% Cardinalidad': (df.nunique() / len(df)) * 100
    }
    
    # Crear un DataFrame a partir del diccionario
    resumen_df = pd.DataFrame(resumen)
    
    # Ajustar el formato de la salida (por ejemplo, redondear los porcentajes)
    resumen_df['% Valores Nulos'] = resumen_df['% Valores Nulos'].round(2)
    resumen_df['% Cardinalidad'] = resumen_df['% Cardinalidad'].round(2)
    
    return resumen_df.T

# Funcion: tipifica_variables


def sugerir_tipo_variable(df, umbral_categoria, umbral_continua):
    # Inicializar una lista para almacenar el resultado
    sugerencias = []

    # Recorrer cada columna del DataFrame
    for col in df.columns:
        # Calcular la cardinalidad (número de valores únicos)
        cardinalidad = df[col].nunique()
        # Calcular el porcentaje de cardinalidad
        porcentaje_cardinalidad = (cardinalidad / len(df)) * 100
        
        # Determinar el tipo sugerido
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo_sugerido = "Numerica Continua"
        else:
            tipo_sugerido = "Numerica Discreta"
        
        # Añadir la sugerencia a la lista
        sugerencias.append({
            'nombre_variable': col,
            'tipo_sugerido': tipo_sugerido
        })
    
    # Convertir la lista de sugerencias en un DataFrame
    resultado_df = pd.DataFrame(sugerencias)
    
    return resultado_df


# Funcion: get_features_num_regression


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    La función identifica columnas numéricas en un dataframe que esten correlacionadas
    con la columna objetivo (target_col), utilizando umbrales de correlación y el p-value.

    Argumentos de la función:
    df (pd.DataFrame): dataframe que contiene los datos.
    target_col (str): nombre de la columna objetivo para el análisis de regresión.
    umbral_corr (float): valor absoluto mínimo de correlación para seleccionar las características.
    pvalue (float y opcional): filtra según la significancia de la correlación. Si es None, no se realiza este test.

    Retorna:
    list: Una lista con los nombres de las columnas que están correlacionadas con la columna objetivo por encima del umbral.
    """
    # Verifica que la columna objetivo existe y es numérica. Si no lo es, mostramos un mensaje de error y terminamos la función.
    if target_col not in df.columns:
        print("Error: target_col debe ser una columna existente en el DataFrame.")
        return None
    
    # Obtener el resumen del DataFrame para verificar el tipo de dato de la columna objetivo
    resumen = resumen_dataframe(df)

    # Verificar que la columna objetivo es numérica
    if resumen[target_col]['Tipo de Dato'] not in ['int64', 'float64']:
        print("Error: target_col debe ser numérica.")
        return None
    
     # Usar la función sugerir_tipo_variable para obtener las columnas numéricas
    tipificacion = sugerir_tipo_variable(df, umbral_categoria=10, umbral_continua=0.05) #Umbral_categoria se puede modificar si tenemos menos valores únicos
    
    # Filtrar las columnas que son numéricas (continuas o discretas)
    numeric_cols = tipificacion[tipificacion['tipo_sugerido'].isin(['Numerica Continua', 'Numerica Discreta'])]['nombre_variable']
    
    # Excluir la columna objetivo de la lista de columnas a analizar
    numeric_cols = numeric_cols[numeric_cols != target_col]

    # Lista para almacenar las columnas seleccionadas.
    selected_columns = []

    # Para cada columna numérica, calcular la correlación con la columna objetivo
    for col in numeric_cols:
        correlacion, p_val = pearsonr(df[col], df[target_col]) #Se usa pearsonr para saber como de fuerte es la relación entre el tarjet y las columnas numéricas.
        
        # Si la correlación supera el umbral
        if abs(correlacion) >= umbral_corr: #abs se utiliza para identificar la magnitud de la relación. Con esto vemos lo fuerte que es la realación entre el tarjet y otra columna. Mayor o igual al umbral que definimos
            #Si la columna no está bien correlacionada la descartamos por no tener datos relevantes.
            if pvalue is not None: # Si se especifica pvalue, también verifica la significancia estadística
                if p_val <= (1 - pvalue): #Buscamos que si pvalue es menor o igual 1, la confianza es mayor al 95%.
                    selected_columns.append(col)
            else:
                selected_columns.append(col)

    return selected_columns
    

# Funcion: plot_features_num_regression


# Funcion: get_features_cat_regression


# Funcion: plot_features_cat_regression