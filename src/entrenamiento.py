# =============================================================================
# train_model.py - Entrenamiento del Árbol de Decisión
# =============================================================================
# Este módulo se encarga de:
#   1. Entrenar el árbol de decisión sobre el dataset cardíaco.
#   2. Extraer y generar reglas IF-THEN automáticamente.
#   3. Visualizar el árbol gráficamente y guardarlo como PNG.
#   4. Calcular y reportar métricas de evaluación básicas.
#
# TEORÍA MATEMÁTICA UTILIZADA
# ─────────────────────────────
# El árbol de decisión utiliza el criterio GINI para seleccionar
# el mejor atributo en cada nodo de división.
#
# Índice Gini de un nodo t:
#
#   Gini(t) = 1 - Σ p(i|t)²
#
# donde p(i|t) es la proporción de ejemplos de clase i en el nodo t.
# Un Gini = 0 indica nodo puro (todos de la misma clase).
# Un Gini = 0.5 indica máxima impureza (50% / 50%).
#
# Ganancia de información (alternativa con entropía):
#
#   Entropía(t) = - Σ p(i|t) · log₂(p(i|t))
#
#   GananciaInfo = Entropía(padre) - Σ (|hijo_j| / |padre|) · Entropía(hijo_j)
#
# El árbol elige la división que maximiza la ganancia de información
# (o equivalentemente minimiza el Gini ponderado de los hijos).
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla para servidores/scripts
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils import (
    cargar_dataset,
    obtener_X_y,
    FEATURE_NAMES,
    TARGET_NAME,
    VARIABLE_INFO,
    imprimir_encabezado,
    formatear_separador,
)

# ---------------------------------------------------------------------------
# RUTAS DE ARCHIVOS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATASET = os.path.join(BASE_DIR, "data", "heart_statlog_cleveland_hungary_final.csv")
RUTA_MODELO = os.path.join(BASE_DIR, "data", "modelo_arbol.pkl")
RUTA_REGLAS = os.path.join(BASE_DIR, "data", "reglas.pkl")
RUTA_IMAGEN_ARBOL = os.path.join(BASE_DIR, "images", "arbol_decision.png")
RUTA_IMAGEN_ARBOL_ROOT = os.path.join(BASE_DIR, "tree.png")  # copia raíz

# Hiperparámetros del árbol – valores intencionalmente conservadores
# para mantener el árbol interpretable y no sobreajustado.
MAX_DEPTH = 5        # Profundidad máxima: evita sobreajuste y mantiene legibilidad
MIN_SAMPLES_LEAF = 10  # Mínimo de muestras en hoja: evita reglas sobre casos aislados
CRITERIO = "gini"    # Criterio de división: índice de Gini


# =============================================================================
# CLASE: ModeloArbol
# =============================================================================

class ModeloArbol:
    """
    Encapsula el entrenamiento, evaluación, extracción de reglas
    y visualización del árbol de decisión.

    Atributos
    ---------
    arbol : DecisionTreeClassifier
        El clasificador de árbol de decisión de sklearn.
    reglas : list[dict]
        Lista de reglas IF-THEN extraídas del árbol.
    X_train, X_test, y_train, y_test : arrays
        Particiones del dataset para entrenamiento y evaluación.
    nombres_caracteristicas : list[str]
        Nombres de las variables de entrada.
    """

    def __init__(self):
        self.arbol: DecisionTreeClassifier = None
        self.reglas: list = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.nombres_caracteristicas = FEATURE_NAMES

    # ------------------------------------------------------------------
    # MÉTODO: entrenar
    # ------------------------------------------------------------------

    def entrenar(self, df: pd.DataFrame, verbose: bool = True) -> None:
        """
        Entrena el árbol de decisión a partir del DataFrame.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataset completo cargado desde CSV.
        verbose : bool
            Si es True, imprime información del proceso.
        """
        X, y = obtener_X_y(df)

        # División estratificada 80% entrenamiento / 20% prueba
        # Estratificada = mantiene proporción de clases en ambas particiones
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,       # Semilla fija → resultado determinista siempre
            stratify=y,
        )

        if verbose:
            print(f"  Muestras de entrenamiento : {len(self.X_train)}")
            print(f"  Muestras de prueba        : {len(self.X_test)}")
            print(f"  Distribución entrenamiento: {dict(self.y_train.value_counts())}")

        # Crear y entrenar el árbol
        self.arbol = DecisionTreeClassifier(
            criterion=CRITERIO,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=42,   # Determinismo garantizado
        )
        self.arbol.fit(self.X_train, self.y_train)

        if verbose:
            print(f"  Profundidad real del árbol: {self.arbol.get_depth()}")
            print(f"  Número de nodos           : {self.arbol.tree_.node_count}")
            print(f"  Número de hojas           : {self.arbol.get_n_leaves()}")

    # ------------------------------------------------------------------
    # MÉTODO: evaluar
    # ------------------------------------------------------------------

    def evaluar(self) -> dict:
        """
        Evalúa el modelo sobre el conjunto de prueba y retorna métricas.

        Retorna
        -------
        dict con accuracy, reporte de clasificación y matriz de confusión.
        """
        if self.arbol is None:
            raise RuntimeError("El árbol no ha sido entrenado. Llame a entrenar() primero.")

        y_pred = self.arbol.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        reporte = classification_report(
            self.y_test, y_pred,
            target_names=["Sin enfermedad (0)", "Con enfermedad (1)"],
        )
        matriz = confusion_matrix(self.y_test, y_pred)

        return {
            "accuracy": acc,
            "reporte": reporte,
            "matriz_confusion": matriz,
        }

    # ------------------------------------------------------------------
    # MÉTODO: extraer_reglas
    # ------------------------------------------------------------------

    def extraer_reglas(self) -> list:
        """
        Recorre el árbol de decisión y extrae todas las reglas IF-THEN
        que llevan desde la raíz hasta cada hoja.

        Cada regla es un diccionario con:
            - condiciones : list[str]  → lista de condiciones (AND)
            - clase       : int        → clase predicha (0 o 1)
            - muestras    : int        → número de muestras de entrenamiento en la hoja
            - gini        : float      → impureza Gini de la hoja (0 = puro)
            - confianza   : float      → proporción de la clase mayoritaria en la hoja

        Retorna
        -------
        list[dict]
        """
        if self.arbol is None:
            raise RuntimeError("El árbol no ha sido entrenado.")

        tree = self.arbol.tree_
        feature_names = self.nombres_caracteristicas
        reglas = []

        def _recorrer(nodo_id: int, condiciones_actuales: list) -> None:
            """Recorre recursivamente el árbol construyendo las condiciones."""

            # NODO HOJA: no tiene hijos → genera una regla
            if tree.children_left[nodo_id] == -1:
                conteos = tree.value[nodo_id][0]
                clase_predicha = int(np.argmax(conteos))
                total_muestras = int(np.sum(conteos))
                confianza = float(conteos[clase_predicha] / total_muestras)

                # Calcular Gini de la hoja
                proporciones = conteos / total_muestras
                gini_hoja = 1.0 - float(np.sum(proporciones ** 2))

                regla = {
                    "condiciones": list(condiciones_actuales),
                    "clase": clase_predicha,
                    "muestras": total_muestras,
                    "gini": round(gini_hoja, 4),
                    "confianza": round(confianza, 4),
                }
                reglas.append(regla)
                return

            # NODO INTERNO: tiene hijos → agregar condición y seguir
            caracteristica_idx = tree.feature[nodo_id]
            umbral = tree.threshold[nodo_id]
            nombre_col = feature_names[caracteristica_idx]

            # Obtener nombre descriptivo de la variable
            info = VARIABLE_INFO.get(nombre_col, {})
            nombre_desc = info.get("nombre_mostrado", nombre_col)

            # Rama izquierda: condición ≤ umbral (VERDADERA)
            cond_izq = f"{nombre_desc} ({nombre_col}) ≤ {umbral:.2f}"
            _recorrer(
                tree.children_left[nodo_id],
                condiciones_actuales + [cond_izq],
            )

            # Rama derecha: condición > umbral (FALSA)
            cond_der = f"{nombre_desc} ({nombre_col}) > {umbral:.2f}"
            _recorrer(
                tree.children_right[nodo_id],
                condiciones_actuales + [cond_der],
            )

        _recorrer(0, [])  # Iniciar desde la raíz (nodo 0)
        self.reglas = reglas
        return reglas

    # ------------------------------------------------------------------
    # MÉTODO: visualizar_arbol
    # ------------------------------------------------------------------

    def visualizar_arbol(self, guardar: bool = True) -> None:
        """
        Genera una visualización gráfica del árbol de decisión y la
        guarda como imagen PNG.

        Parámetros
        ----------
        guardar : bool
            Si True, guarda el archivo en images/ y en la raíz del proyecto.
        """
        if self.arbol is None:
            raise RuntimeError("El árbol no ha sido entrenado.")

        # Nombres descriptivos para el plot
        nombres_display = [
            VARIABLE_INFO.get(col, {}).get("nombre_mostrado", col)
            for col in self.nombres_caracteristicas
        ]

        fig, ax = plt.subplots(figsize=(60, 22))
        plot_tree(
            self.arbol,
            feature_names=nombres_display,
            class_names=["Sin Enfermedad", "Con Enfermedad"],
            filled=True,           # Colorea nodos según clase
            rounded=True,          # Bordes redondeados
            fontsize=8,
            ax=ax,
            impurity=True,         # Muestra el Gini en cada nodo
            proportion=False,      # Muestra conteos absolutos
        )
        ax.set_title(
            "Árbol de Decisión — Sistema Experto de Diagnóstico Cardíaco\n"
            f"(Criterio: Gini | Profundidad máx: {MAX_DEPTH} | "
            f"Mín. muestras/hoja: {MIN_SAMPLES_LEAF})",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        if guardar:
            os.makedirs(os.path.dirname(RUTA_IMAGEN_ARBOL), exist_ok=True)
            plt.savefig(RUTA_IMAGEN_ARBOL, dpi=150, bbox_inches="tight")
            plt.savefig(RUTA_IMAGEN_ARBOL_ROOT, dpi=150, bbox_inches="tight")
            print(f"  Árbol guardado en: {RUTA_IMAGEN_ARBOL}")
            print(f"  Árbol guardado en: {RUTA_IMAGEN_ARBOL_ROOT}")

        plt.close(fig)

    # ------------------------------------------------------------------
    # MÉTODO: imprimir_reglas_texto
    # ------------------------------------------------------------------

    def imprimir_reglas_texto(self) -> None:
        """
        Imprime en consola la representación textual del árbol (formato sklearn)
        y todas las reglas IF-THEN generadas.
        """
        if self.arbol is None:
            raise RuntimeError("El árbol no ha sido entrenado.")

        nombres_display = [
            VARIABLE_INFO.get(col, {}).get("nombre_mostrado", col)
            for col in self.nombres_caracteristicas
        ]

        print("\n" + formatear_separador(70))
        print("  REPRESENTACIÓN TEXTUAL DEL ÁRBOL")
        print(formatear_separador(70))
        texto_arbol = export_text(
            self.arbol,
            feature_names=nombres_display,
        )
        print(texto_arbol)

        print("\n" + formatear_separador(70))
        print(f"  REGLAS IF-THEN GENERADAS AUTOMÁTICAMENTE ({len(self.reglas)} reglas)")
        print(formatear_separador(70))

        for i, regla in enumerate(self.reglas, start=1):
            etiqueta = "CON ENFERMEDAD" if regla["clase"] == 1 else "SIN ENFERMEDAD"
            print(f"\n  REGLA #{i:03d}  →  Diagnóstico: {etiqueta}")
            print(f"  Muestras: {regla['muestras']}  |  Gini: {regla['gini']}  |  Confianza: {regla['confianza']*100:.1f}%")
            print("  SI:")
            for cond in regla["condiciones"]:
                print(f"      AND  {cond}")
            print(f"  ENTONCES: {etiqueta}")

    # ------------------------------------------------------------------
    # MÉTODOS: guardar / cargar modelo
    # ------------------------------------------------------------------

    def guardar(self) -> None:
        """Serializa el modelo y las reglas en disco usando pickle."""
        os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
        with open(RUTA_MODELO, "wb") as f:
            pickle.dump(self.arbol, f)
        with open(RUTA_REGLAS, "wb") as f:
            pickle.dump(self.reglas, f)
        print(f"  Modelo guardado en: {RUTA_MODELO}")
        print(f"  Reglas guardadas en: {RUTA_REGLAS}")

    def cargar(self) -> bool:
        """
        Carga el modelo y las reglas desde disco.

        Retorna True si se cargó correctamente, False si no existen archivos.
        """
        if not os.path.exists(RUTA_MODELO) or not os.path.exists(RUTA_REGLAS):
            return False
        with open(RUTA_MODELO, "rb") as f:
            self.arbol = pickle.load(f)
        with open(RUTA_REGLAS, "rb") as f:
            self.reglas = pickle.load(f)
        return True


# =============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO COMPLETO
# =============================================================================

def ejecutar_entrenamiento_completo(verbose: bool = True) -> ModeloArbol:
    """
    Orquesta el flujo completo de entrenamiento:
    1. Cargar dataset
    2. Entrenar árbol
    3. Evaluar métricas
    4. Extraer reglas
    5. Visualizar árbol
    6. Guardar modelo

    Retorna el objeto ModeloArbol entrenado.
    """
    imprimir_encabezado("ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")

    print(f"\n  Cargando dataset desde: {RUTA_DATASET}")
    df = cargar_dataset(RUTA_DATASET)
    print(f"  Total de registros cargados: {len(df)}")
    print(f"  Distribución de clases: {dict(df[TARGET_NAME].value_counts())}")

    print("\n  Entrenando árbol de decisión...")
    modelo = ModeloArbol()
    modelo.entrenar(df, verbose=verbose)

    print("\n  Evaluando sobre conjunto de prueba...")
    metricas = modelo.evaluar()
    print(f"\n  Exactitud (Accuracy): {metricas['accuracy']*100:.2f}%")
    print(f"\n  Reporte de clasificación:\n{metricas['reporte']}")
    print(f"\n  Matriz de confusión:\n{metricas['matriz_confusion']}")

    print("\n  Extrayendo reglas IF-THEN del árbol...")
    reglas = modelo.extraer_reglas()
    print(f"  Total de reglas generadas: {len(reglas)}")

    print("\n  Generando imagen del árbol de decisión...")
    modelo.visualizar_arbol(guardar=True)

    print("\n  Guardando modelo en disco...")
    modelo.guardar()

    imprimir_encabezado("ENTRENAMIENTO COMPLETADO")
    return modelo


# =============================================================================
# EJECUCIÓN DIRECTA (modo script)
# =============================================================================

if __name__ == "__main__":
    modelo = ejecutar_entrenamiento_completo(verbose=True)
    modelo.imprimir_reglas_texto()
