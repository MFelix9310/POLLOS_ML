#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo predictivo para estimar el consumo diario de alimento en pollos Cobb 500
Utiliza regresión Random Forest para generar predicciones precisas basadas en características clave.
Soporta predicciones para pollos desde el día 1 hasta el día 45.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

# Añadir la ruta del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos(ruta_archivo):
    """
    Carga y preprocesa los datos desde un archivo CSV.
    
    Args:
        ruta_archivo: Ruta al archivo CSV con los datos
        
    Returns:
        DataFrame procesado y listo para modelado
    """
    print(f'Cargando datos desde: {ruta_archivo}')
    df = pd.read_csv(ruta_archivo)
    print(f'Dimensiones iniciales: {df.shape}')
    
    # Verificar si la columna Consumo_g ya existe, si no, crearla
    if 'Consumo_g' not in df.columns:
        print('ADVERTENCIA: La columna Consumo_g no está presente en los datos.')
        print('El modelo necesita esta columna para entrenamiento.')
        return None
    
    # Eliminar filas con NaN en columnas críticas
    df_clean = df.dropna(subset=['Peso_g', 'Ganancia_Peso_g', 'Consumo_g'])
    print(f'Filas después de eliminar NaN: {len(df_clean)}')
    
    return df_clean

def crear_caracteristicas(df):
    """
    Crea características adicionales para mejorar el rendimiento del modelo.
    
    Args:
        df: DataFrame con los datos originales
        
    Returns:
        DataFrame con características adicionales
    """
    # Crear características adicionales útiles para predecir consumo
    df['Peso_Relativo'] = df['Peso_g'] / df['Peso_Final_Promedio_g']
    df['Dia_Cuadrado'] = df['Dia'] ** 2  # Capturar relaciones no lineales
    df['Peso_Cuadrado'] = df['Peso_g'] ** 2
    
    # Características para capturar patrones en el crecimiento
    df['Semana'] = np.ceil(df['Dia'] / 7).astype(int)
    df['Etapa_Crecimiento'] = pd.cut(
        df['Dia'], 
        bins=[0, 7, 14, 21, 28, 35, 45], 
        labels=['Semana1', 'Semana2', 'Semana3', 'Semana4', 'Semana5', 'Semana6+']
    )
    
    # Relación peso-ganancia
    df['Eficiencia_Ganancia'] = df['Ganancia_Peso_g'] / df['Peso_g']
    df['Eficiencia_Ganancia'].fillna(0, inplace=True)
    
    # Extraer el tratamiento base (T0, T1, T2)
    df['Tratamiento_Base'] = df['Tratamiento'].str.extract(r'(T\d+)')
    
    return df

def dividir_datos(df, features, target):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        df: DataFrame con los datos
        features: Lista de características a usar
        target: Nombre de la columna objetivo
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separar características y objetivo
    X = df[features]
    y = df[target]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['Semana']
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def crear_pipeline(numeric_features, categorical_features):
    """
    Crea un pipeline de preprocesamiento y modelado.
    
    Args:
        numeric_features: Lista de características numéricas
        categorical_features: Lista de características categóricas
        
    Returns:
        Pipeline de scikit-learn
    """
    # Crear preprocesador para manejar características numéricas y categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # Crear pipeline con preprocesamiento y modelo Random Forest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    return pipeline

def optimizar_hiperparametros(pipeline, X_train, y_train):
    """
    Realiza búsqueda de hiperparámetros para optimizar el modelo.
    
    Args:
        pipeline: Pipeline a optimizar
        X_train: Características de entrenamiento
        y_train: Valores objetivo de entrenamiento
        
    Returns:
        Pipeline optimizado
    """
    print("Optimizando hiperparámetros del modelo...")
    
    # Definir parámetros para búsqueda de cuadrícula
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 15, 30],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    
    # Realizar búsqueda de cuadrícula con validación cruzada
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, 
        scoring='r2', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluar_modelo(model, X_test, y_test):
    """
    Evalúa el rendimiento del modelo en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Valores objetivo de prueba
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    # Predecir en conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Evaluación del modelo:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def visualizar_resultados(X_test, y_test, y_pred, output_dir):
    """
    Crea visualizaciones para evaluar el rendimiento del modelo.
    
    Args:
        X_test: Características de prueba
        y_test: Valores reales
        y_pred: Valores predichos
        output_dir: Directorio donde guardar las visualizaciones
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de dispersión: Valores reales vs. predichos
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax1.set_xlabel('Consumo real (g)')
    ax1.set_ylabel('Consumo predicho (g)')
    ax1.set_title('Valores reales vs. predichos')
    
    # Histograma de residuos
    residuos = y_test - y_pred
    ax2.hist(residuos, bins=30, edgecolor='k')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residuos')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de residuos')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluacion_modelo_consumo_cobb500.png'))
    print(f"Visualización guardada en: {os.path.join(output_dir, 'evaluacion_modelo_consumo_cobb500.png')}")
    plt.close()
    
    # Gráfico de error por día
    test_df = X_test.copy()
    test_df['Consumo_Real'] = y_test
    test_df['Consumo_Predicho'] = y_pred
    test_df['Error_Absoluto'] = abs(y_test - y_pred)
    
    # Agrupar por día
    error_por_dia = test_df.groupby('Dia')['Error_Absoluto'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.bar(error_por_dia['Dia'], error_por_dia['Error_Absoluto'], color='coral')
    plt.xlabel('Día')
    plt.ylabel('Error Absoluto Medio (g)')
    plt.title('Error de predicción por día')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'error_por_dia_cobb500.png'))
    print(f"Visualización guardada en: {os.path.join(output_dir, 'error_por_dia_cobb500.png')}")
    plt.close()
    
    # Gráfico de importancia de características
    if hasattr(test_df, 'columns'):
        modelo = model.named_steps['model']
        
        # Obtener nombres de características después del preprocesamiento
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        
        for name, trans, cols in preprocessor.transformers_:
            if name == 'cat':
                # Para características categóricas, obtener los nombres de las categorías
                feature_names.extend([f"{col}_{category}" for col in cols 
                                     for category in trans.categories_[0][1:]])
            else:
                feature_names.extend(cols)
        
        # Crear gráfico de importancia
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Importancia de características')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                                        for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'importancia_caracteristicas_cobb500.png'))
        print(f"Visualización guardada en: {os.path.join(output_dir, 'importancia_caracteristicas_cobb500.png')}")
        plt.close()

def guardar_modelo(model, ruta_archivo):
    """
    Guarda el modelo entrenado para uso futuro.
    
    Args:
        model: Modelo entrenado
        ruta_archivo: Ruta donde guardar el modelo
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    
    joblib.dump(model, ruta_archivo)
    print(f"Modelo guardado en: {ruta_archivo}")

def main():
    """Función principal que ejecuta todo el proceso de modelado"""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo predictivo para consumo de pollos Cobb 500')
    parser.add_argument('--archivo', type=str, default='src/data/datos_entrenamiento_cobb500.csv',
                        help='Ruta al archivo CSV con datos de entrenamiento')
    parser.add_argument('--salida', type=str, default='src/models/modelo_consumo_cobb500.pkl',
                        help='Ruta donde guardar el modelo entrenado')
    parser.add_argument('--visualizaciones', type=str, default='src/models/visualizaciones',
                        help='Directorio donde guardar las visualizaciones')
    
    args = parser.parse_args()
    
    # Cargar y preprocesar datos
    df = cargar_datos(args.archivo)
    if df is None:
        return
    
    # Crear características adicionales
    df = crear_caracteristicas(df)
    
    # Definir características para el modelo
    features = [
        'Dia', 'Dia_Cuadrado', 'Peso_g', 'Peso_Cuadrado', 
        'Peso_Relativo', 'Ganancia_Peso_g', 'Eficiencia_Ganancia',
        'Conversion_Alimenticia_Global', 'Tratamiento_Base', 'Etapa_Crecimiento'
    ]
    target = 'Consumo_g'
    
    # Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(df, features, target)
    
    # Identificar columnas numéricas y categóricas
    numeric_features = [col for col in features if col not in ['Tratamiento_Base', 'Etapa_Crecimiento']]
    categorical_features = ['Tratamiento_Base', 'Etapa_Crecimiento']
    
    # Crear pipeline
    pipeline = crear_pipeline(numeric_features, categorical_features)
    
    # Optimizar hiperparámetros
    best_model = optimizar_hiperparametros(pipeline, X_train, y_train)
    
    # Evaluar modelo
    metrics = evaluar_modelo(best_model, X_test, y_test)
    
    # Visualizar resultados
    y_pred = best_model.predict(X_test)
    visualizar_resultados(X_test, y_test, y_pred, args.visualizaciones)
    
    # Guardar modelo
    guardar_modelo(best_model, args.salida)
    
    print("\nProceso completo. El modelo está listo para realizar predicciones hasta 45 días.")

if __name__ == "__main__":
    main() 