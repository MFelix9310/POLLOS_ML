#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para ejecutar el Sistema de Predicción de Consumo de Pollos Cobb 500.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Sistema de Predicción de Consumo para Pollos Cobb 500')
    parser.add_argument('--generar-datos', action='store_true',
                        help='Generar datos de ejemplo para entrenamiento')
    parser.add_argument('--entrenar-modelo', action='store_true',
                        help='Entrenar modelo con los datos generados')
    parser.add_argument('--interfaz', action='store_true',
                        help='Iniciar la interfaz gráfica (predeterminado)')
    
    args = parser.parse_args()
    
    # Si no se especifica ninguna acción, inicia la interfaz
    if not (args.generar_datos or args.entrenar_modelo or args.interfaz):
        args.interfaz = True
    
    # Ejecutar las acciones solicitadas
    if args.generar_datos:
        print("Generando datos de ejemplo...")
        from src.data.generador_datos import main as generar_datos
        generar_datos()
    
    if args.entrenar_modelo:
        print("Entrenando modelo...")
        from src.models.modelo_prediccion_consumo_cobb500 import main as entrenar_modelo
        entrenar_modelo()
    
    if args.interfaz:
        print("Iniciando interfaz gráfica...")
        from src.ui.app_prediccion_consumo_cobb500 import main as iniciar_app
        iniciar_app()

if __name__ == "__main__":
    main() 