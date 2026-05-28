# =============================================================================
# main.py - Consola Interactiva del Sistema Experto Cardíaco
# =============================================================================
# Punto de entrada principal del proyecto.
#
# FLUJO DEL PROGRAMA
# ──────────────────
# 1. Al iniciarse, verifica si existe un modelo entrenado.
#    - Si no existe → entrena automáticamente.
#    - Si existe    → lo carga directamente.
# 2. Muestra un menú principal con opciones.
# 3. El usuario puede:
#    a. Ingresar datos de un paciente y obtener un diagnóstico.
#    b. Ver todas las reglas del sistema.
#    c. Consultar las descripciones de variables médicas.
#    d. Re-entrenar el modelo desde cero.
#    e. Salir del programa.
#
# DETERMINISMO
# ────────────
# El sistema es completamente determinista: para los mismos datos
# de entrada siempre producirá exactamente el mismo diagnóstico.
# Esto es posible porque:
#   - El árbol de decisión no tiene componentes estocásticos.
#   - La inferencia recorre un único camino posible por el árbol.
#   - No se usa ningún algoritmo probabilístico ni de muestreo.
# =============================================================================

import os
import sys

# Agregar src/ al path para importaciones relativas
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from utils import (
    FEATURE_NAMES,
    VARIABLE_INFO,
    validar_valor,
    formatear_separador,
    imprimir_encabezado,
)
from src.entrenamiento import ejecutar_entrenamiento_completo, RUTA_MODELO
from src.sistema_experto import SistemaExperto


# =============================================================================
# CLASE: AplicacionConsola
# =============================================================================

class AplicacionConsola:
    """
    Controlador principal de la interfaz de consola.

    Gestiona el menú, la recolección de datos del usuario,
    la llamada al sistema experto y la presentación de resultados.
    """

    def __init__(self):
        self.sistema = SistemaExperto()
        self._inicializar_sistema()

    # ------------------------------------------------------------------
    # INICIALIZACIÓN
    # ------------------------------------------------------------------

    def _inicializar_sistema(self) -> None:
        """Carga o entrena el modelo según disponibilidad."""
        self._limpiar_pantalla()
        imprimir_encabezado(
            "SISTEMA EXPERTO DE DIAGNÓSTICO CARDÍACO", ancho=65
        )
        print("\n  Tecnología: Árbol de Decisión + Reglas IF-THEN")
        print("  Paradigma : IA Simbólica — Inferencia Determinista")
        print("  Dataset   : Heart Statlog / Cleveland / Hungary")
        print(formatear_separador(65))

        if os.path.exists(RUTA_MODELO):
            print("\n  Cargando modelo entrenado...")
            self.sistema.cargar_modelo()
            print(f"  Modelo cargado. Base de conocimiento: {self.sistema.num_reglas} reglas.")
        else:
            print("\n  No se encontró modelo previo.")
            print("  Iniciando entrenamiento automático...")
            print()
            modelo_entrenado = ejecutar_entrenamiento_completo(verbose=True)
            self.sistema.cargar_modelo()
            print(f"\n  Sistema listo. Base de conocimiento: {self.sistema.num_reglas} reglas.")

        input("\n  Presione Enter para continuar...")

    # ------------------------------------------------------------------
    # MENÚ PRINCIPAL
    # ------------------------------------------------------------------

    def ejecutar(self) -> None:
        """Bucle principal del menú de la aplicación."""
        while True:
            self._mostrar_menu_principal()
            opcion = input("\n  Ingrese su opción: ").strip()

            if opcion == "1":
                self._flujo_diagnostico()
            elif opcion == "2":
                self._limpiar_pantalla()
                self.sistema.mostrar_todas_las_reglas()
                input("\n  Presione Enter para volver al menú...")
            elif opcion == "3":
                self._limpiar_pantalla()
                self.sistema.mostrar_variables()
                input("\n  Presione Enter para volver al menú...")
            elif opcion == "4":
                self._limpiar_pantalla()
                self.sistema.mostrar_estadisticas_reglas()
                input("\n  Presione Enter para volver al menú...")
            elif opcion == "5":
                self._flujo_reentrenamiento()
            elif opcion == "6":
                self._mostrar_teoria()
                input("\n  Presione Enter para volver al menú...")
            elif opcion == "7":
                self._limpiar_pantalla()
                print("\n  Cerrando el Sistema Experto de Diagnóstico Cardíaco.")
                print("  ¡Hasta luego!\n")
                break
            else:
                print("\n  [ERROR] Opción no válida. Intente nuevamente.")
                input("  Presione Enter para continuar...")

    def _mostrar_menu_principal(self) -> None:
        """Muestra el menú principal formateado."""
        self._limpiar_pantalla()
        sep = formatear_separador(65)
        print(f"\n{sep}")
        print("  SISTEMA EXPERTO — DIAGNÓSTICO CARDÍACO")
        print(f"  Reglas activas: {self.sistema.num_reglas} | Determinista | IA Simbólica")
        print(sep)
        print()
        print("  [1]  Diagnosticar un paciente")
        print("  [2]  Ver base de conocimiento (todas las reglas)")
        print("  [3]  Ver descripción de variables médicas")
        print("  [4]  Ver estadísticas del sistema")
        print("  [5]  Re-entrenar el modelo")
        print("  [6]  Ver teoría del sistema (entropía, Gini, etc.)")
        print("  [7]  Salir")
        print(f"\n{sep}")

    # ------------------------------------------------------------------
    # FLUJO: DIAGNÓSTICO
    # ------------------------------------------------------------------

    def _flujo_diagnostico(self) -> None:
        """Guía al usuario para ingresar datos del paciente y muestra el diagnóstico."""
        self._limpiar_pantalla()
        imprimir_encabezado("NUEVO DIAGNÓSTICO — INGRESO DE DATOS", ancho=65)

        print("\n  Ingrese los datos del paciente.")
        print("  Escriba '?' antes de cualquier variable para ver su descripción.")
        print("  Escriba 'menu' para cancelar y volver al menú principal.")
        print(formatear_separador(65))

        datos = {}

        for col in FEATURE_NAMES:
            info = VARIABLE_INFO[col]
            nombre = info["nombre_mostrado"]
            tipo = info["tipo"]
            rmin, rmax = info["rango_valido"]
            unidad = info.get("unidad", "")
            opciones = info.get("opciones")

            print(f"\n  ── {nombre} ──")
            if unidad:
                print(f"     Unidad: {unidad}  |  Rango: [{rmin} – {rmax}]")
            else:
                print(f"     Rango: [{rmin} – {rmax}]")

            if opciones:
                for k, v in opciones.items():
                    print(f"       {k} = {v}")

            while True:
                entrada = input(f"  Valor para '{nombre}': ").strip()

                if entrada.lower() == "menu":
                    print("\n  Diagnóstico cancelado. Volviendo al menú...")
                    input("  Presione Enter...")
                    return

                if entrada.startswith("?"):
                    from utils import describir_variable
                    print(describir_variable(col))
                    continue

                valido, mensaje_error = validar_valor(col, entrada)
                if not valido:
                    print(f"  [ERROR] {mensaje_error}")
                    continue

                # Convertir al tipo correcto
                if tipo == "float":
                    datos[col] = float(entrada)
                else:
                    datos[col] = int(float(entrada))
                break

        # Ejecutar inferencia determinista
        self._limpiar_pantalla()
        print("\n  Ejecutando inferencia determinista...")
        print(f"  Recorriendo el árbol de decisión...\n")

        resultado = self.sistema.diagnosticar(datos)
        resultado.imprimir()

        input(f"\n{formatear_separador(60)}\n  Presione Enter para volver al menú...")

    # ------------------------------------------------------------------
    # FLUJO: RE-ENTRENAMIENTO
    # ------------------------------------------------------------------

    def _flujo_reentrenamiento(self) -> None:
        """Permite re-entrenar el modelo desde cero."""
        self._limpiar_pantalla()
        imprimir_encabezado("RE-ENTRENAMIENTO DEL MODELO", ancho=65)

        print("\n  ADVERTENCIA: Esto re-entrenará el árbol de decisión desde cero.")
        print("  El modelo actual será reemplazado.")
        confirmacion = input("\n  ¿Desea continuar? (s/n): ").strip().lower()

        if confirmacion != "s":
            print("\n  Re-entrenamiento cancelado.")
            input("  Presione Enter para volver al menú...")
            return

        print()
        ejecutar_entrenamiento_completo(verbose=True)
        self.sistema.cargar_modelo()
        print(f"\n  Sistema actualizado. Base de conocimiento: {self.sistema.num_reglas} reglas.")
        input("\n  Presione Enter para volver al menú...")

    # ------------------------------------------------------------------
    # SECCIÓN: TEORÍA
    # ------------------------------------------------------------------

    def _mostrar_teoria(self) -> None:
        """Muestra la teoría matemática del sistema."""
        self._limpiar_pantalla()
        sep = formatear_separador(65)

        print(f"\n{sep}")
        print("  FUNDAMENTOS TEÓRICOS DEL SISTEMA EXPERTO")
        print(sep)

        print("""
  ┌─────────────────────────────────────────────────────────┐
  │  1. ÍNDICE DE GINI                                      │
  └─────────────────────────────────────────────────────────┘

  Mide la impureza de un nodo del árbol.
  Un nodo es "puro" cuando todos sus ejemplos son de la misma clase.

  Fórmula:
    Gini(t) = 1 - Σ p(i|t)²

  Donde:
    • p(i|t) = proporción de ejemplos de clase i en el nodo t
    • Σ = suma sobre todas las clases

  Ejemplo:
    Nodo con 60% clase 0 y 40% clase 1:
    Gini = 1 - (0.60² + 0.40²)
         = 1 - (0.36 + 0.16)
         = 1 - 0.52 = 0.48  ← Impuro

    Nodo con 100% clase 1:
    Gini = 1 - (1.0²) = 0   ← Puro (decisión perfecta)

  ┌─────────────────────────────────────────────────────────┐
  │  2. ENTROPÍA                                            │
  └─────────────────────────────────────────────────────────┘

  Medida alternativa de impureza, basada en teoría de información.

  Fórmula:
    H(t) = - Σ p(i|t) · log₂(p(i|t))

  Ejemplo:
    Nodo con 50% clase 0 y 50% clase 1:
    H = -(0.5 · log₂(0.5) + 0.5 · log₂(0.5))
      = -(0.5 · (-1) + 0.5 · (-1))
      = 1.0 bit  ← Máxima incertidumbre

    Nodo puro (100% clase 1):
    H = -(1.0 · log₂(1.0)) = 0 bits  ← Sin incertidumbre

  ┌─────────────────────────────────────────────────────────┐
  │  3. GANANCIA DE INFORMACIÓN                             │
  └─────────────────────────────────────────────────────────┘

  Cuantifica cuánto reduce la impureza dividir por una variable.

  Fórmula:
    IG = H(padre) - Σ (|hijo_j| / |padre|) · H(hijo_j)

  El árbol elige en cada nodo la variable que maximiza IG.
  Mayor IG = esa variable separa mejor las dos clases.

  ┌─────────────────────────────────────────────────────────┐
  │  4. DETERMINISMO EN EL ÁRBOL                           │
  └─────────────────────────────────────────────────────────┘

  Dado un árbol entrenado y un vector de entrada X:

    1. Se evalúa la condición del nodo raíz: X[j] ≤ umbral?
    2. Si SÍ  → moverse al hijo izquierdo.
    3. Si NO  → moverse al hijo derecho.
    4. Repetir hasta llegar a una hoja.
    5. La hoja asigna la clase mayoritaria de sus ejemplos.

  No hay probabilidades, no hay aleatoriedad.
  Misma entrada → misma hoja → mismo diagnóstico SIEMPRE.

  ┌─────────────────────────────────────────────────────────┐
  │  5. REGLAS IF-THEN                                      │
  └─────────────────────────────────────────────────────────┘

  Cada camino raíz→hoja del árbol se traduce en una regla:

    SI   condición_1 (nodo 1)
    AND  condición_2 (nodo 2)
    AND  condición_3 (nodo 3)
    ENTONCES  clase_de_la_hoja

  Esto hace el razonamiento completamente explicable y auditable.
""")

    # ------------------------------------------------------------------
    # UTILIDAD
    # ------------------------------------------------------------------

    @staticmethod
    def _limpiar_pantalla() -> None:
        """Limpia la terminal según el sistema operativo."""
        os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    app = AplicacionConsola()
    app.ejecutar()
