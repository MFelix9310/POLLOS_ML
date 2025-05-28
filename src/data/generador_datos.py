#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generador de datos para el sistema de predicción de consumo en pollos Cobb 500.
Genera datos realistas para entrenamiento y prueba con soporte hasta 45 días.
"""

import pandas as pd
import numpy as np
import os
import argparse

def generar_datos_ejemplo(num_filas=100, incluir_consumo=True, dias_max=45):
    """
    Genera un DataFrame con datos de ejemplo para pollos Cobb 500.
    
    Args:
        num_filas: Número de filas a generar
        incluir_consumo: Si se debe incluir la columna Consumo_g
        dias_max: Número máximo de días a considerar (hasta 45)
        
    Returns:
        DataFrame con datos de ejemplo
    """
    # Definir rangos realistas para cada característica
    dias = np.random.randint(1, dias_max + 1, size=num_filas)  # Días 1-45
    
    # Distribución de pesos según los días (valores más realistas basados en curva de crecimiento Cobb500)
    pesos = []
    for dia in dias:
        if dia <= 7:  # Primera semana
            # Crecimiento inicial moderado
            base = 40 + (dia * 15)  # ~40g inicial a ~145g al día 7
            variacion = np.random.uniform(0.8, 1.2)
            pesos.append(int(base * variacion))
        elif dia <= 14:  # Segunda semana
            # Crecimiento más rápido
            base = 145 + ((dia - 7) * 30)  # ~145g a ~355g
            variacion = np.random.uniform(0.85, 1.15)
            pesos.append(int(base * variacion))
        elif dia <= 21:  # Tercera semana
            # Crecimiento continuo acelerado
            base = 355 + ((dia - 14) * 50)  # ~355g a ~705g
            variacion = np.random.uniform(0.9, 1.1)
            pesos.append(int(base * variacion))
        elif dia <= 28:  # Cuarta semana
            # Crecimiento acelerado
            base = 705 + ((dia - 21) * 70)  # ~705g a ~1195g
            variacion = np.random.uniform(0.9, 1.1)
            pesos.append(int(base * variacion))
        elif dia <= 35:  # Quinta semana
            # Crecimiento acelerado
            base = 1195 + ((dia - 28) * 85)  # ~1195g a ~1790g
            variacion = np.random.uniform(0.9, 1.1)
            pesos.append(int(base * variacion))
        else:  # Sexta semana en adelante
            # Crecimiento algo más lento pero continuo
            base = 1790 + ((dia - 35) * 75)  # ~1790g a ~2315g al día 42
            variacion = np.random.uniform(0.9, 1.1)
            pesos.append(int(base * variacion))
    
    # Tratamientos disponibles
    tratamientos = []
    for _ in range(num_filas):
        tipo = np.random.choice(['T0', 'T1', 'T2'])
        replica = np.random.choice(['R1', 'R2', 'R3', 'R4', 'R5'])
        tratamientos.append(f"{tipo}{replica}")
    
    # Generar pesos anteriores (ligeramente menores que los actuales)
    pesos_anteriores = []
    for i in range(num_filas):
        # Para el primer día, algunos valores pueden ser NaN
        if dias[i] == 1 and np.random.random() < 0.3:
            pesos_anteriores.append(np.nan)
        else:
            # Peso anterior menor que el actual
            factor = np.random.uniform(0.85, 0.98)
            pesos_anteriores.append(round(pesos[i] * factor))
    
    # Calcular ganancias de peso
    ganancias_peso = []
    for i in range(num_filas):
        if pd.isna(pesos_anteriores[i]):
            ganancias_peso.append(np.nan)
        else:
            ganancias_peso.append(pesos[i] - pesos_anteriores[i])
    
    # Valores para los pesos iniciales y finales promedio
    # Simular diferentes grupos experimentales
    pesos_iniciales = []
    pesos_finales = []
    conversiones = []
    
    for tratamiento in tratamientos:
        if tratamiento.startswith('T0'):
            pesos_iniciales.append(39.4)
            pesos_finales.append(3363.45)
            conversiones.append(1.739702)
        elif tratamiento.startswith('T1'):
            pesos_iniciales.append(40.2)
            pesos_finales.append(3290.12)
            conversiones.append(1.812543)
        else:
            pesos_iniciales.append(41.5)
            pesos_finales.append(3425.78)
            conversiones.append(1.684291)
    
    # Consumos (si se solicitan)
    consumos = None
    if incluir_consumo:
        consumos = []
        for i in range(num_filas):
            # Consumo basado en una función más compleja que considera edad y peso
            if dias[i] <= 7:
                # Primera semana: consumo bajo
                base_consumo = pesos[i] * 0.08 + dias[i] * 2
            elif dias[i] <= 21:
                # 2-3 semanas: consumo moderado, incrementando
                base_consumo = pesos[i] * 0.1 + dias[i] * 3
            elif dias[i] <= 35:
                # 4-5 semanas: consumo alto
                base_consumo = pesos[i] * 0.12 + dias[i] * 4
            else:
                # 6+ semanas: consumo muy alto
                base_consumo = pesos[i] * 0.15 + dias[i] * 5
            
            # Añadir variación aleatoria
            variacion = np.random.uniform(0.85, 1.15)
            consumos.append(round(base_consumo * variacion))
    
    # Crear el DataFrame
    datos = {
        'Dia': dias,
        'Tratamiento': tratamientos,
        'Peso_g': pesos,
        'Peso_Anterior_g': pesos_anteriores,
        'Ganancia_Peso_g': ganancias_peso,
        'Peso_Inicial_Promedio_g': pesos_iniciales,
        'Peso_Final_Promedio_g': pesos_finales,
        'Conversion_Alimenticia_Global': conversiones
    }
    
    if incluir_consumo:
        datos['Consumo_g'] = consumos
    
    df = pd.DataFrame(datos)
    
    # Asegurar que no hay valores negativos en ganancia de peso
    df.loc[df['Ganancia_Peso_g'] < 0, 'Ganancia_Peso_g'] = np.random.randint(5, 30)
    
    # Ordenar por día para facilitar análisis
    df = df.sort_values('Dia').reset_index(drop=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generador de datos de ejemplo para modelo de predicción de consumo')
    parser.add_argument('--filas', type=int, default=500, 
                        help='Número de filas a generar (por defecto: 500)')
    parser.add_argument('--dias', type=int, default=45,
                        help='Número máximo de días a considerar (por defecto: 45)')
    parser.add_argument('--output', type=str, default='src/data/datos_entrenamiento_cobb500.csv',
                        help='Ruta del archivo de salida (por defecto: src/data/datos_entrenamiento_cobb500.csv)')
    
    args = parser.parse_args()
    
    # Generar datos
    print(f"Generando {args.filas} filas de datos de ejemplo para entrenamiento (hasta {args.dias} días)...")
    df = generar_datos_ejemplo(args.filas, incluir_consumo=True, dias_max=args.dias)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Guardar a CSV
    df.to_csv(args.output, index=False)
    print(f"Datos guardados en: {args.output}")
    
    # Mostrar resumen
    print("\nResumen de los datos generados:")
    print(df.head())
    
    # Estadísticas básicas
    print("\nEstadísticas por día:")
    stats = df.groupby('Dia')[['Peso_g', 'Consumo_g']].agg(['mean', 'std']).reset_index()
    print(stats.head(10))  # Mostrar primeros 10 días
    
    # Instrucciones de uso
    print("\nPara usar estos datos con el modelo de predicción, ejecute:")
    print(f"python src/models/modelo_prediccion_consumo_cobb500.py --archivo {args.output}")

if __name__ == "__main__":
    main() 