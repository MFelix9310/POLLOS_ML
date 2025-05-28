#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Componentes reutilizables para la interfaz gráfica del sistema de predicción de consumo.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                            QLabel, QLineEdit, QPushButton, QComboBox, 
                            QGroupBox, QTableWidget, QTableWidgetItem, 
                            QHeaderView, QMessageBox, QFileDialog, QSplitter)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator, QColor, QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import os
import pandas as pd
import numpy as np

class MplCanvas(FigureCanvas):
    """Clase para embeber gráficos de Matplotlib en PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

class FormularioPrediccionIndividual(QWidget):
    """Widget que contiene el formulario para predicción individual"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Grupo para entrada de datos
        input_group = QGroupBox("Datos del Pollo")
        input_layout = QFormLayout(input_group)
        
        # Crear campos de entrada
        self.dia_input = QLineEdit()
        self.dia_input.setValidator(QRegExpValidator(QRegExp("[1-9][0-9]*")))
        self.dia_input.setPlaceholderText("Entre 1 y 45")
        input_layout.addRow("Día:", self.dia_input)
        
        self.peso_input = QLineEdit()
        self.peso_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.peso_input.setPlaceholderText("En gramos")
        input_layout.addRow("Peso actual:", self.peso_input)
        
        self.peso_anterior_input = QLineEdit()
        self.peso_anterior_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.peso_anterior_input.setPlaceholderText("En gramos")
        input_layout.addRow("Peso anterior:", self.peso_anterior_input)
        
        self.tratamiento_combo = QComboBox()
        for tipo in ["T0", "T1", "T2"]:
            for replica in ["R1", "R2", "R3", "R4", "R5"]:
                self.tratamiento_combo.addItem(f"{tipo}{replica}")
        input_layout.addRow("Tratamiento:", self.tratamiento_combo)
        
        self.peso_inicial_input = QLineEdit()
        self.peso_inicial_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.peso_inicial_input.setText("39.4")
        input_layout.addRow("Peso inicial promedio (g):", self.peso_inicial_input)
        
        self.peso_final_input = QLineEdit()
        self.peso_final_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.peso_final_input.setText("3363.45")
        input_layout.addRow("Peso final promedio (g):", self.peso_final_input)
        
        self.conversion_input = QLineEdit()
        self.conversion_input.setValidator(QRegExpValidator(QRegExp("[0-9]*\\.?[0-9]+")))
        self.conversion_input.setText("1.73")
        input_layout.addRow("Conversión alimenticia global:", self.conversion_input)
        
        layout.addWidget(input_group)
        
        # Botón de predicción
        self.predict_button = QPushButton("Realizar Predicción")
        self.predict_button.setStyleSheet("font-weight: bold; padding: 8px;")
        layout.addWidget(self.predict_button)
        
        # Grupo de resultados
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout(results_group)
        
        self.resultado_label = QLabel("Realice una predicción para ver el resultado")
        self.resultado_label.setAlignment(Qt.AlignCenter)
        self.resultado_label.setStyleSheet("font-size: 16px; margin: 20px;")
        results_layout.addWidget(self.resultado_label)
        
        layout.addWidget(results_group)
        layout.addStretch()

class FormularioProcesarLotes(QWidget):
    """Widget que contiene el formulario para procesamiento por lotes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Sección de carga de archivo
        file_group = QGroupBox("Cargar Archivo CSV")
        file_layout = QHBoxLayout(file_group)
        
        self.file_path_label = QLabel("No se ha seleccionado ningún archivo")
        file_layout.addWidget(self.file_path_label)
        
        self.browse_button = QPushButton("Examinar...")
        file_layout.addWidget(self.browse_button)
        
        self.process_button = QPushButton("Procesar Archivo")
        file_layout.addWidget(self.process_button)
        
        layout.addWidget(file_group)
        
        # Tabla de resultados
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)  # Ajustar según las columnas que quieras mostrar
        self.results_table.setHorizontalHeaderLabels(["Día", "Peso (g)", "Tratamiento", "Consumo Predicho (g)", ""])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        self.save_button = QPushButton("Guardar Resultados")
        results_layout.addWidget(self.save_button)
        
        layout.addWidget(results_group)

class VisualizacionDatos(QWidget):
    """Widget para visualización de datos y resultados"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Sección de visualización
        visual_group = QGroupBox("Visualización de Datos")
        visual_layout = QVBoxLayout(visual_group)
        
        # Canvas para gráficos
        self.plot_canvas = MplCanvas(self, width=9, height=5, dpi=100)
        visual_layout.addWidget(self.plot_canvas)
        
        # Controles para gráficos
        controls_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Consumo por Día", 
            "Consumo por Peso", 
            "Consumo por Tratamiento",
            "Curva de Crecimiento",
            "Relación Consumo-Ganancia"
        ])
        controls_layout.addWidget(QLabel("Tipo de Gráfico:"))
        controls_layout.addWidget(self.plot_type_combo)
        
        self.plot_button = QPushButton("Generar Gráfico")
        controls_layout.addWidget(self.plot_button)
        
        visual_layout.addLayout(controls_layout)
        layout.addWidget(visual_group)

def aplicar_estilo_general(widget):
    """
    Aplica un estilo general consistente a cualquier widget de la aplicación
    
    Args:
        widget: El widget al que aplicar el estilo
    """
    widget.setStyleSheet("""
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