#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicación de interfaz gráfica para la predicción de consumo diario de alimento
en pollos Cobb 500 con proyección para alcanzar un peso objetivo.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                           QPushButton, QComboBox, QTabWidget, QFileDialog,
                           QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator, QFont, QColor, QPalette

class MplCanvas(FigureCanvas):
    """Clase para embeber gráficos de Matplotlib en PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

class AppPrediccionConsumoCobb500(QMainWindow):
    """Aplicación para la predicción de consumo en pollos Cobb 500 con objetivo de peso"""
    
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana principal
        self.setWindowTitle("Predicción de Consumo - Pollos Cobb 500")
        self.setMinimumSize(1000, 700)
        
        # Cargar el modelo
        self.cargar_modelo()
        
        # Configurar la interfaz de usuario
        self.setup_ui()
        
        # Centrar la ventana
        self.center_window()
    
    def cargar_modelo(self):
        """Carga el modelo entrenado"""
        try:
            # Intentar cargar el mejor modelo disponible
            modelo_path = 'mejor_modelo_cobb500.pkl'
            if os.path.exists(modelo_path):
                self.modelo = joblib.load(modelo_path)
                self.modelo_cargado = True
                print(f"Modelo cargado correctamente desde: {modelo_path}")
            else:
                # Intentar con el otro modelo si el primero no existe
                modelo_path_alt = 'modelo_consumo_cobb500.pkl'
                if os.path.exists(modelo_path_alt):
                    self.modelo = joblib.load(modelo_path_alt)
                    self.modelo_cargado = True
                    print(f"Modelo alternativo cargado correctamente desde: {modelo_path_alt}")
                else:
                    self.modelo = None
                    self.modelo_cargado = False
                    print(f"No se encontraron modelos entrenados en las rutas: {modelo_path} o {modelo_path_alt}")
        except Exception as e:
            self.modelo = None
            self.modelo_cargado = False
            print(f"Error al cargar el modelo: {str(e)}")
    
    def setup_ui(self):
        """Configura todos los elementos de la interfaz de usuario"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Verificar si el modelo está cargado
        if not self.modelo_cargado:
            warning_label = QLabel("⚠️ ADVERTENCIA: No se pudo cargar el modelo. Por favor, entrene el modelo primero.")
            warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            main_layout.addWidget(warning_label)
        
        # Crear pestañas
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Pestaña de predicción individual
        individual_tab = QWidget()
        tabs.addTab(individual_tab, "Predicción Individual")
        self.setup_individual_tab(individual_tab)
        
        # Pestaña de visualización
        visualization_tab = QWidget()
        tabs.addTab(visualization_tab, "Visualización")
        self.setup_visualization_tab(visualization_tab)
        
        # Pestaña de información
        info_tab = QWidget()
        tabs.addTab(info_tab, "Información")
        self.setup_info_tab(info_tab)
        
        # Barra de estado
        self.statusBar().showMessage("Listo para realizar predicciones")
        
        # Aplicar estilo
        self.aplicar_estilo()
    
    def center_window(self):
        """Centra la ventana en la pantalla"""
        screen_geometry = QApplication.desktop().availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def setup_individual_tab(self, tab):
        """Configura la pestaña de predicción individual"""
        layout = QVBoxLayout(tab)
        
        # Grupo para entrada de datos
        input_group = QGroupBox("Datos de Predicción")
        input_layout = QFormLayout(input_group)
        
        # Crear campos de entrada
        self.peso_actual_input = QLineEdit()
        self.peso_actual_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.peso_actual_input.setPlaceholderText("En gramos")
        input_layout.addRow("Peso actual (g):", self.peso_actual_input)
        
        self.peso_objetivo_input = QLineEdit()
        self.peso_objetivo_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.peso_objetivo_input.setPlaceholderText("En gramos (ej: 6000)")
        input_layout.addRow("Peso objetivo (g):", self.peso_objetivo_input)
        
        self.tratamiento_combo = QComboBox()
        # Tratamiento T0 es el más efectivo según la conversión alimenticia más baja
        for tipo in ["T0", "T1", "T2"]:
            for replica in ["R1", "R2", "R3", "R4", "R5"]:
                tratamiento = f"{tipo}{replica}"
                self.tratamiento_combo.addItem(tratamiento)
                # Seleccionar T0R1 por defecto (mejor tratamiento)
                if tratamiento == "T0R1":
                    self.tratamiento_combo.setCurrentText(tratamiento)
        input_layout.addRow("Tratamiento:", self.tratamiento_combo)
        
        self.conversion_input = QLineEdit()
        self.conversion_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        # Usar la conversión del tratamiento T0 (1.739702)
        self.conversion_input.setText("1.73")
        input_layout.addRow("Conversión alimenticia:", self.conversion_input)
        
        layout.addWidget(input_group)
        
        # Botones de acción
        button_layout = QHBoxLayout()
        
        # Botón de predicción
        predict_button = QPushButton("Generar Tabla de Consumo")
        predict_button.setStyleSheet("font-weight: bold; padding: 8px;")
        predict_button.clicked.connect(self.generar_tabla_consumo)
        button_layout.addWidget(predict_button)
        
        # Botón para exportar CSV
        export_button = QPushButton("Exportar a CSV")
        export_button.clicked.connect(self.exportar_resultados_csv)
        button_layout.addWidget(export_button)
        
        layout.addLayout(button_layout)
        
        # Tabla de resultados
        results_group = QGroupBox("Tabla de Consumo Diario")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Día", "Peso Proyectado (g)", "Ganancia Diaria (g)", "Consumo Diario (g)"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        # Resumen
        self.resumen_label = QLabel("Complete los datos y genere la tabla de consumo")
        self.resumen_label.setAlignment(Qt.AlignCenter)
        self.resumen_label.setStyleSheet("font-size: 14px; margin: 10px;")
        results_layout.addWidget(self.resumen_label)
        
        layout.addWidget(results_group)
    
    def exportar_resultados_csv(self):
        """Exporta los resultados de la tabla de consumo a un archivo CSV"""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "Error", "No hay datos para exportar. Genere la tabla de consumo primero.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Resultados", "", "Archivos CSV (*.csv);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                # Crear DataFrame con los datos de la tabla
                datos = []
                for i in range(self.results_table.rowCount()):
                    fila = {
                        'Dia': int(self.results_table.item(i, 0).text()),
                        'Peso_Proyectado_g': float(self.results_table.item(i, 1).text()),
                        'Ganancia_Diaria_g': float(self.results_table.item(i, 2).text()),
                        'Consumo_Diario_g': float(self.results_table.item(i, 3).text())
                    }
                    datos.append(fila)
                
                # Convertir a DataFrame y guardar
                df = pd.DataFrame(datos)
                df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Éxito", f"Datos exportados correctamente a {file_path}")
                self.statusBar().showMessage(f"Datos exportados a {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar los datos: {str(e)}")
                self.statusBar().showMessage("Error al exportar los datos")
    
    def generar_tabla_consumo(self):
        """Genera la tabla de consumo proyectado"""
        if not self.modelo_cargado:
            QMessageBox.warning(self, "Error", "No se ha cargado el modelo. Por favor, entrene el modelo primero.")
            return
        
        try:
            # Obtener datos de entrada
            peso_actual = float(self.peso_actual_input.text())
            peso_objetivo = float(self.peso_objetivo_input.text())
            tratamiento = self.tratamiento_combo.currentText()
            conversion = float(self.conversion_input.text())
            
            # Validar datos
            if peso_actual <= 0 or peso_objetivo <= 0 or conversion <= 0:
                QMessageBox.warning(self, "Error", "Los valores deben ser positivos.")
                return
            
            if peso_actual >= peso_objetivo:
                QMessageBox.warning(self, "Error", "El peso objetivo debe ser mayor al peso actual.")
                return
            
            # Proyectar crecimiento
            dias, pesos, ganancias, consumos = self.proyectar_crecimiento(
                peso_actual, peso_objetivo, tratamiento, conversion
            )
            
            # Preparar tabla de resultados
            self.results_table.setRowCount(len(dias))
            consumo_total = 0
            
            for i, (dia, peso, ganancia, consumo) in enumerate(zip(dias, pesos, ganancias, consumos)):
                # Asegurar que no haya valores negativos
                consumo = max(0, consumo)
                ganancia = max(0, ganancia)
                
                consumo_total += consumo
                
                self.results_table.setItem(i, 0, QTableWidgetItem(str(dia)))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{peso:.2f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{ganancia:.2f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{consumo:.2f}"))
                
                # Colorear filas alternadas
                if i % 2 == 0:
                    for j in range(4):
                        self.results_table.item(i, j).setBackground(QColor(240, 240, 240))
            
            # Actualizar resumen
            total_dias = len(dias)
            self.resumen_label.setText(
                f"Tiempo estimado: {total_dias} días | Consumo total: {consumo_total:.2f} g | "
                f"Consumo promedio diario: {consumo_total/total_dias:.2f} g"
            )
            
            # Actualizar barra de estado
            self.statusBar().showMessage(f"Tabla de consumo generada para alcanzar {peso_objetivo} g en {total_dias} días")
            
        except ValueError:
            QMessageBox.warning(self, "Error de Entrada", 
                              "Por favor, complete todos los campos con valores numéricos válidos.")
            self.statusBar().showMessage("Error: Verifique los datos de entrada")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error durante la predicción: {str(e)}")
            self.statusBar().showMessage("Error durante la predicción")
    
    def proyectar_crecimiento(self, peso_actual, peso_objetivo, tratamiento, conversion):
        """
        Proyecta el crecimiento diario y consumo hasta alcanzar el peso objetivo
        
        Args:
            peso_actual: Peso actual en gramos
            peso_objetivo: Peso objetivo en gramos
            tratamiento: Tratamiento aplicado
            conversion: Índice de conversión alimenticia
            
        Returns:
            dias, pesos, ganancias, consumos: Listas con la proyección
        """
        # Extraer el tratamiento base
        tratamiento_base = tratamiento[:2]  # T0, T1, T2
        
        # Inicializar variables
        dias = []
        pesos = []
        ganancias = []
        consumos = []
        
        # Valores mínimos para evitar predicciones negativas
        min_ganancia_peso = 5  # Ganancia mínima diaria en gramos
        min_dia = 1
        
        # Día inicial (asumimos que estamos en el día 1)
        dia_actual = 1
        peso_actual_dia = peso_actual
        ganancia_anterior = min_ganancia_peso  # Inicializar con un valor mínimo positivo
        
        # Proyectar hasta alcanzar el peso objetivo o un máximo de 100 días
        while peso_actual_dia < peso_objetivo and dia_actual <= 100:
            dias.append(dia_actual)
            pesos.append(peso_actual_dia)
            
            # Crear características para el modelo
            datos = pd.DataFrame({
                'Dia': [max(min_dia, dia_actual)],  # Asegurar día mínimo
                'Peso_g': [max(10, peso_actual_dia)],  # Asegurar peso mínimo
                'Ganancia_Peso_g': [ganancia_anterior],  # Usar la ganancia del día anterior
                'Tratamiento': [tratamiento],
                'Peso_Inicial_Promedio_g': [max(10, peso_actual)],  # Asegurar peso inicial mínimo
                'Peso_Final_Promedio_g': [max(100, peso_objetivo)],  # Asegurar peso final mínimo
                'Conversion_Alimenticia_Global': [max(1.0, conversion)]  # Asegurar conversión mínima
            })
            
            # Añadir características adicionales necesarias para el modelo
            datos['Peso_Relativo'] = datos['Peso_g'] / datos['Peso_Final_Promedio_g']
            datos['Dia_Cuadrado'] = datos['Dia'] ** 2
            datos['Peso_Cuadrado'] = datos['Peso_g'] ** 2
            datos['Tratamiento_Base'] = tratamiento_base
            
            # Características para predicción
            features = [
                'Dia', 'Dia_Cuadrado', 'Peso_g', 'Peso_Cuadrado', 
                'Peso_Relativo', 'Ganancia_Peso_g', 
                'Conversion_Alimenticia_Global', 'Tratamiento_Base'
            ]
            
            try:
                # Predecir consumo
                consumo_diario = self.modelo.predict(datos[features])[0]
                
                # Asegurar que el consumo no sea negativo
                consumo_diario = max(10, consumo_diario)  # Mínimo 10g de consumo diario
                
            except Exception as e:
                print(f"Error en predicción: {str(e)}")
                # En caso de error, asignar un valor de consumo positivo basado en el peso
                consumo_diario = max(10, peso_actual_dia * 0.05)  # Aproximadamente 5% del peso
            
            consumos.append(consumo_diario)
            
            # Calcular ganancia de peso para el siguiente día
            # La ganancia de peso es aproximadamente: consumo / conversión
            ganancia_diaria = consumo_diario / max(1.0, conversion)
            
            # Asegurar que la ganancia no sea negativa y tenga un mínimo razonable
            ganancia_diaria = max(min_ganancia_peso, ganancia_diaria)
            
            ganancias.append(ganancia_diaria)
            
            # Guardar la ganancia para el próximo ciclo
            ganancia_anterior = ganancia_diaria
            
            # Actualizar peso para el siguiente día
            peso_actual_dia += ganancia_diaria
            dia_actual += 1
        
        return dias, pesos, ganancias, consumos 

    def setup_visualization_tab(self, tab):
        """Configura la pestaña de visualización"""
        layout = QVBoxLayout(tab)
        
        # Sección de visualización
        visual_group = QGroupBox("Visualización de Proyecciones")
        visual_layout = QVBoxLayout(visual_group)
        
        # Canvas para gráficos
        self.plot_canvas = MplCanvas(self, width=9, height=5, dpi=100)
        visual_layout.addWidget(self.plot_canvas)
        
        # Controles para gráficos
        controls_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Consumo Diario vs. Peso", 
            "Peso Proyectado vs. Días", 
            "Consumo Acumulado vs. Días"
        ])
        controls_layout.addWidget(QLabel("Tipo de Gráfico:"))
        controls_layout.addWidget(self.plot_type_combo)
        
        plot_button = QPushButton("Generar Gráfico")
        plot_button.clicked.connect(self.generar_grafico_proyeccion)
        controls_layout.addWidget(plot_button)
        
        visual_layout.addLayout(controls_layout)
        layout.addWidget(visual_group)
    
    def generar_grafico_proyeccion(self):
        """Genera gráficos de proyección de crecimiento y consumo"""
        if not self.modelo_cargado:
            QMessageBox.warning(self, "Error", "No se ha cargado el modelo. Por favor, entrene el modelo primero.")
            return
        
        try:
            # Usar el peso objetivo de la pestaña de predicción individual
            if not self.peso_objetivo_input.text():
                QMessageBox.warning(self, "Error", "Por favor, ingrese un peso objetivo en la pestaña de Predicción Individual.")
                return
            
            peso_objetivo = float(self.peso_objetivo_input.text())
            peso_objetivo = max(100, peso_objetivo)  # Asegurar peso objetivo mínimo
            
            # Definir pesos iniciales para proyección
            pesos_iniciales = [100, 500, 1000, 1500, 2000, 2500]
            tratamiento = "T0R1"  # Tratamiento por defecto
            conversion = 1.73  # Conversión por defecto
            
            # Limpiar gráfico anterior
            self.plot_canvas.axes.clear()
            
            # Seleccionar tipo de gráfico
            plot_type = self.plot_type_combo.currentText()
            
            if plot_type == "Consumo Diario vs. Peso":
                # Generar datos para diferentes pesos
                pesos_x = np.linspace(50, peso_objetivo, 100)
                consumos_y = []
                
                for peso in pesos_x:
                    # Crear datos para predicción
                    datos = pd.DataFrame({
                        'Dia': [5],  # Día promedio
                        'Peso_g': [max(10, peso)],  # Asegurar peso mínimo
                        'Ganancia_Peso_g': [max(5, 50)],  # Ganancia mínima positiva
                        'Tratamiento': [tratamiento],
                        'Peso_Inicial_Promedio_g': [100],
                        'Peso_Final_Promedio_g': [max(100, peso_objetivo)],
                        'Conversion_Alimenticia_Global': [max(1.0, conversion)]
                    })
                    
                    # Añadir características adicionales
                    datos['Peso_Relativo'] = datos['Peso_g'] / datos['Peso_Final_Promedio_g']
                    datos['Dia_Cuadrado'] = datos['Dia'] ** 2
                    datos['Peso_Cuadrado'] = datos['Peso_g'] ** 2
                    datos['Tratamiento_Base'] = tratamiento[:2]
                    
                    # Características para predicción
                    features = [
                        'Dia', 'Dia_Cuadrado', 'Peso_g', 'Peso_Cuadrado', 
                        'Peso_Relativo', 'Ganancia_Peso_g', 
                        'Conversion_Alimenticia_Global', 'Tratamiento_Base'
                    ]
                    
                    try:
                        # Predecir consumo
                        consumo = self.modelo.predict(datos[features])[0]
                        
                        # Asegurar que el consumo no sea negativo
                        consumo = max(10, consumo)  # Mínimo 10g de consumo
                        
                    except Exception as e:
                        print(f"Error en predicción gráfica: {str(e)}")
                        # En caso de error, asignar un valor de consumo positivo basado en el peso
                        consumo = max(10, peso * 0.05)  # Aproximadamente 5% del peso
                    
                    consumos_y.append(consumo)
                
                # Graficar
                self.plot_canvas.axes.plot(pesos_x, consumos_y, 'b-')
                self.plot_canvas.axes.set_xlabel('Peso (g)')
                self.plot_canvas.axes.set_ylabel('Consumo Diario (g)')
                self.plot_canvas.axes.set_title('Consumo Diario Estimado vs. Peso')
                self.plot_canvas.axes.grid(True)
                
                # Establecer límite inferior del eje y a 0 para evitar mostrar valores negativos
                self.plot_canvas.axes.set_ylim(bottom=0)
                
            elif plot_type == "Peso Proyectado vs. Días":
                # Obtener peso actual de la pestaña de predicción individual
                if not self.peso_actual_input.text():
                    peso_actual = 100  # valor por defecto
                else:
                    peso_actual = float(self.peso_actual_input.text())
                    peso_actual = max(10, peso_actual)  # Asegurar peso mínimo
                    
                # Calcular proyección
                dias, pesos, _, _ = self.proyectar_crecimiento(
                    peso_actual, peso_objetivo, tratamiento, conversion
                )
                
                # Graficar
                self.plot_canvas.axes.plot(dias, pesos, label=f'Inicial: {peso_actual}g')
                
                self.plot_canvas.axes.set_xlabel('Días')
                self.plot_canvas.axes.set_ylabel('Peso Proyectado (g)')
                self.plot_canvas.axes.set_title(f'Proyección de Peso hasta {peso_objetivo}g')
                self.plot_canvas.axes.grid(True)
                self.plot_canvas.axes.legend()
                
                # Establecer límite inferior del eje y a 0
                self.plot_canvas.axes.set_ylim(bottom=0)
                
            elif plot_type == "Consumo Acumulado vs. Días":
                # Obtener peso actual de la pestaña de predicción individual
                if not self.peso_actual_input.text():
                    peso_actual = 100  # valor por defecto
                else:
                    peso_actual = float(self.peso_actual_input.text())
                    peso_actual = max(10, peso_actual)  # Asegurar peso mínimo
                    
                # Calcular proyección
                dias, _, _, consumos = self.proyectar_crecimiento(
                    peso_actual, peso_objetivo, tratamiento, conversion
                )
                
                # Calcular consumo acumulado
                # Asegurar que todos los consumos son positivos antes de acumular
                consumos = [max(0, c) for c in consumos]
                consumo_acumulado = np.cumsum(consumos)
                
                # Graficar
                self.plot_canvas.axes.plot(dias, consumo_acumulado, label=f'Inicial: {peso_actual}g')
                
                self.plot_canvas.axes.set_xlabel('Días')
                self.plot_canvas.axes.set_ylabel('Consumo Acumulado (g)')
                self.plot_canvas.axes.set_title(f'Consumo Acumulado hasta {peso_objetivo}g')
                self.plot_canvas.axes.grid(True)
                self.plot_canvas.axes.legend()
                
                # Establecer límite inferior del eje y a 0
                self.plot_canvas.axes.set_ylim(bottom=0)
            
            # Actualizar gráfico
            self.plot_canvas.fig.tight_layout()
            self.plot_canvas.draw()
            
            # Actualizar barra de estado
            self.statusBar().showMessage(f"Gráfico generado: {plot_type}")
            
        except ValueError:
            QMessageBox.warning(self, "Error de Entrada", "Por favor, ingrese un valor numérico válido para el peso objetivo.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar el gráfico: {str(e)}")
    
    def setup_info_tab(self, tab):
        """Configura la pestaña de información"""
        layout = QVBoxLayout(tab)
        
        info_text = """
        <h2>Sistema de Predicción de Consumo para Pollos Cobb 500</h2>
        
        <p>Este sistema utiliza técnicas de aprendizaje automático (Random Forest) para predecir el consumo diario de alimento de pollos Cobb 500 y proyectar el crecimiento hasta alcanzar un peso objetivo.</p>
        
        <h3>Características del Sistema</h3>
        <ul>
            <li><b>Predicción Individual</b>: Genere una tabla de consumo diario proyectado desde un peso actual hasta un peso objetivo.</li>
            <li><b>Visualización</b>: Explore gráficamente las proyecciones de consumo y crecimiento.</li>
        </ul>
        
        <h3>Uso de la Predicción Individual</h3>
        <ol>
            <li>Ingrese el peso actual del pollo</li>
            <li>Especifique el peso objetivo a alcanzar</li>
            <li>Seleccione el tratamiento aplicado</li>
            <li>Indique el índice de conversión alimenticia</li>
            <li>Haga clic en "Generar Tabla de Consumo"</li>
        </ol>
        
        <h3>Notas Importantes</h3>
        <ul>
            <li>El modelo está entrenado con datos de pollos Cobb 500 hasta aproximadamente 6 kg de peso.</li>
            <li>Las proyecciones son estimaciones y pueden variar según las condiciones reales de cría.</li>
            <li>Se recomienda utilizar índices de conversión alimenticia entre 1.5 y 2.0 para mayor precisión.</li>
            <li>El tratamiento T0 muestra los mejores resultados de conversión alimenticia.</li>
        </ul>
        """
        
        info_label = QLabel(info_text)
        info_label.setTextFormat(Qt.RichText)
        info_label.setWordWrap(True)
        info_label.setOpenExternalLinks(True)
        info_label.setStyleSheet("padding: 20px;")
        
        layout.addWidget(info_label)
    
    def aplicar_estilo(self):
        """Aplica estilos a la aplicación"""
        # Estilo general
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QTableWidget {
                alternate-background-color: #f9f9f9;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-bottom-color: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #f0f0f0;
                border-bottom-color: #f0f0f0;
            }
        """)

def main():
    app = QApplication(sys.argv)
    window = AppPrediccionConsumoCobb500()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 