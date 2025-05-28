# Sistema de Predicción de Consumo para Pollos Cobb 500

Este proyecto implementa un modelo de aprendizaje automático para predecir el consumo diario de alimento de pollos Cobb 500 basado en datos históricos de crecimiento y conversión alimenticia.

## Contenido

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
  - [Entrenamiento del Modelo](#entrenamiento-del-modelo)
  - [Realizar Predicciones](#realizar-predicciones)
  - [Generar Datos de Ejemplo](#generar-datos-de-ejemplo)
- [Variables del Modelo](#variables-del-modelo)
- [Resultados](#resultados)

## Descripción

Este sistema utiliza técnicas de aprendizaje automático (Random Forest) para predecir el consumo diario de alimento de pollos Cobb 500. El modelo ha sido entrenado con datos históricos y puede realizar predicciones tanto a partir de archivos CSV como mediante entrada manual.

## Requisitos

- Python 3.6 o superior
- Bibliotecas requeridas:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

Puede instalar todas las dependencias con:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Estructura del Proyecto

- `modelo_prediccion_consumo_cobb500.py`: Script principal para entrenar el modelo
- `predecir_consumo_cobb500.py`: Script para realizar predicciones con el modelo entrenado
- `generar_datos_ejemplo.py`: Herramienta para generar datos de ejemplo
- `Tabla_Unica_ML_Cobb500_Limpia.csv`: Conjunto de datos para entrenar el modelo
- `modelo_consumo_cobb500.pkl`: Modelo entrenado (se crea después de ejecutar el entrenamiento)

## Uso

### Entrenamiento del Modelo

Para entrenar el modelo usando el conjunto de datos proporcionado:

```bash
python modelo_prediccion_consumo_cobb500.py
```

Esto entrenará un modelo de Random Forest y lo guardará como `modelo_consumo_cobb500.pkl`. También generará visualizaciones para evaluar el rendimiento del modelo.

### Realizar Predicciones

#### Desde un archivo CSV:

```bash
python predecir_consumo_cobb500.py --archivo datos_nuevos.csv
```

El archivo CSV debe contener las columnas necesarias: `Dia`, `Peso_g`, `Ganancia_Peso_g`, `Tratamiento`, `Peso_Inicial_Promedio_g`, `Peso_Final_Promedio_g`, `Conversion_Alimenticia_Global`.

#### Mediante entrada manual:

```bash
python predecir_consumo_cobb500.py
```

Este modo le permitirá ingresar los datos de un pollo específico y obtendrá la predicción de consumo correspondiente.

### Generar Datos de Ejemplo

Si no tiene datos propios disponibles, puede generar datos de ejemplo:

```bash
python generar_datos_ejemplo.py --filas 20
```

Para generar datos con valores de consumo (útil para entrenamiento):

```bash
python generar_datos_ejemplo.py --filas 100 --incluir-consumo
```

## Variables del Modelo

El modelo utiliza las siguientes variables para la predicción:

- **Día**: Día de vida del pollo (1-10)
- **Peso_g**: Peso del pollo en gramos
- **Ganancia_Peso_g**: Incremento de peso respecto a la medición anterior
- **Tratamiento**: Código del tratamiento aplicado (ej. T0R1, T1R2, etc.)
- **Peso_Inicial_Promedio_g**: Peso inicial promedio del lote
- **Peso_Final_Promedio_g**: Peso final promedio del lote
- **Conversion_Alimenticia_Global**: Índice de conversión alimenticia

Además, el modelo crea automáticamente algunas características derivadas:
- **Peso_Relativo**: Relación entre el peso actual y el peso final promedio
- **Dia_Cuadrado**: Día al cuadrado (captura relaciones no lineales)
- **Peso_Cuadrado**: Peso al cuadrado (captura relaciones no lineales)
- **Tratamiento_Base**: Extracción del tipo de tratamiento base (T0, T1, T2)

## Resultados

El modelo Random Forest logra un coeficiente de determinación (R²) superior a 0.99, lo que indica una excelente capacidad predictiva. Los errores de predicción son mínimos, lo que hace que este modelo sea adecuado para aplicaciones prácticas en la industria avícola.

Para más detalles sobre el rendimiento del modelo, consulte las visualizaciones generadas durante el entrenamiento. 