# =============================================================================
# expert_system.py - Sistema Experto Determinista de Diagnóstico Cardíaco
# =============================================================================
# Este módulo implementa el motor de inferencia del sistema experto.
#
# PRINCIPIOS DEL SISTEMA EXPERTO DETERMINISTA
# ─────────────────────────────────────────────
# 1. BASE DE CONOCIMIENTO: Conjunto de reglas IF-THEN derivadas del árbol.
# 2. MOTOR DE INFERENCIA: Recorre el árbol nodo a nodo aplicando las
#    condiciones sobre los datos del paciente. El resultado es siempre
#    el mismo para las mismas entradas (DETERMINISMO TOTAL).
# 3. EXPLICACIÓN: El sistema registra cada nodo recorrido y la condición
#    evaluada, construyendo una traza completa del razonamiento.
#
# DETERMINISMO GARANTIZADO PORQUE:
#   - No hay aleatoriedad en la inferencia.
#   - El árbol tiene una única ruta desde la raíz hasta la hoja
#     para cada combinación de valores de entrada.
#   - Misma entrada → misma ruta → mismo diagnóstico siempre.
# =============================================================================

import numpy as np
import os
import pickle

from utils import (
    FEATURE_NAMES,
    VARIABLE_INFO,
    TARGET_INFO,
    formatear_separador,
    imprimir_encabezado,
)

# Rutas (relativas al directorio del módulo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "data", "modelo_arbol.pkl")
RUTA_REGLAS = os.path.join(BASE_DIR, "data", "reglas.pkl")


# =============================================================================
# CLASE: PasoRazonamiento
# =============================================================================

class PasoRazonamiento:
    """
    Representa un paso individual en la traza del razonamiento del árbol.

    Atributos
    ---------
    numero_nodo   : int    → identificador del nodo en el árbol
    profundidad   : int    → nivel de profundidad (0 = raíz)
    variable      : str    → nombre de la variable evaluada
    umbral        : float  → valor de corte utilizado en la división
    valor_paciente: float  → valor real del paciente para esa variable
    condicion     : str    → descripción textual de la condición evaluada
    resultado     : bool   → True si la condición se cumple (rama izquierda)
    rama_tomada   : str    → "izquierda (≤)" o "derecha (>)"
    """

    def __init__(
        self,
        numero_nodo: int,
        profundidad: int,
        variable: str,
        umbral: float,
        valor_paciente: float,
        condicion: str,
        resultado: bool,
    ):
        self.numero_nodo = numero_nodo
        self.profundidad = profundidad
        self.variable = variable
        self.umbral = umbral
        self.valor_paciente = valor_paciente
        self.condicion = condicion
        self.resultado = resultado
        self.rama_tomada = "izquierda (≤)" if resultado else "derecha (>)"

    def __str__(self) -> str:
        indent = "  " * self.profundidad
        check = "✓" if self.resultado else "✗"
        return (
            f"{indent}[Nodo {self.numero_nodo:03d}] Profundidad {self.profundidad}\n"
            f"{indent}  Variable : {self.condicion}\n"
            f"{indent}  Valor paciente : {self.valor_paciente}\n"
            f"{indent}  Umbral árbol   : {self.umbral:.4f}\n"
            f"{indent}  Condición cumplida: {check} {'SÍ' if self.resultado else 'NO'}\n"
            f"{indent}  Rama tomada    : {self.rama_tomada}"
        )


# =============================================================================
# CLASE: ResultadoDiagnostico
# =============================================================================

class ResultadoDiagnostico:
    """
    Contiene el resultado completo de un diagnóstico del sistema experto.

    Atributos
    ---------
    clase_predicha : int             → 0 = sin enfermedad, 1 = con enfermedad
    pasos          : list[PasoRazonamiento] → traza completa del razonamiento
    nodo_hoja      : int             → ID del nodo hoja donde se tomó la decisión
    muestras_hoja  : int             → ejemplos de entrenamiento en la hoja
    gini_hoja      : float           → impureza Gini de la hoja
    confianza      : float           → proporción de la clase en la hoja
    regla_activada : dict            → la regla IF-THEN que se activó
    datos_paciente : dict            → valores de entrada del paciente
    """

    def __init__(self):
        self.clase_predicha: int = -1
        self.pasos: list = []
        self.nodo_hoja: int = -1
        self.muestras_hoja: int = 0
        self.gini_hoja: float = 0.0
        self.confianza: float = 0.0
        self.regla_activada: dict = {}
        self.datos_paciente: dict = {}

    def imprimir(self) -> None:
        """Imprime el resultado completo del diagnóstico en consola."""

        # ── Encabezado ──────────────────────────────────────────────────
        imprimir_encabezado("RESULTADO DEL DIAGNÓSTICO")
        info_clase = TARGET_INFO[self.clase_predicha]
        sep = formatear_separador(60)

        print(f"\n  {'*' * 50}")
        print(f"  DIAGNÓSTICO: {info_clase['etiqueta']}")
        print(f"  {info_clase['descripcion']}")
        print(f"  {'*' * 50}")
        print(f"\n  Nodo hoja alcanzado : #{self.nodo_hoja}")
        print(f"  Muestras en la hoja : {self.muestras_hoja} pacientes de entrenamiento")
        print(f"  Impureza Gini       : {self.gini_hoja:.4f}")
        print(f"  Consistencia interna: {self.confianza*100:.1f}%")

        # ── Regla activada ───────────────────────────────────────────────
        print(f"\n{sep}")
        print("  REGLA IF-THEN ACTIVADA")
        print(sep)
        if self.regla_activada:
            print("  SI:")
            for cond in self.regla_activada.get("condiciones", []):
                print(f"      AND  {cond}")
            etiqueta_regla = (
                "CON ENFERMEDAD" if self.regla_activada["clase"] == 1 else "SIN ENFERMEDAD"
            )
            print(f"  ENTONCES: {etiqueta_regla}")
        else:
            print("  [No se encontró regla coincidente]")

        # ── Traza del razonamiento ────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  TRAZA DEL RAZONAMIENTO ({len(self.pasos)} pasos)")
        print(sep)
        for i, paso in enumerate(self.pasos, start=1):
            print(f"\n  ── Paso {i} ──")
            print(paso)

        # ── Datos del paciente ────────────────────────────────────────────
        print(f"\n{sep}")
        print("  DATOS DEL PACIENTE EVALUADO")
        print(sep)
        for col, valor in self.datos_paciente.items():
            info = VARIABLE_INFO.get(col, {})
            nombre = info.get("nombre_mostrado", col)
            opciones = info.get("opciones")
            if opciones and int(valor) in opciones:
                desc_val = f"{valor}  ({opciones[int(valor)]})"
            else:
                unidad = info.get("unidad", "")
                desc_val = f"{valor} {unidad}".strip()
            print(f"  {nombre:40s}: {desc_val}")


# =============================================================================
# CLASE: SistemaExperto
# =============================================================================

class SistemaExperto:
    """
    Motor de inferencia determinista basado en árbol de decisión.

    El sistema:
    1. Carga el árbol entrenado y las reglas generadas.
    2. Para un conjunto de datos de entrada (paciente), recorre el árbol
       nodo a nodo de forma determinista.
    3. Registra cada paso del recorrido.
    4. Devuelve el diagnóstico con explicación completa.

    IMPORTANTE: No existe ningún elemento estocástico (aleatorio) en
    el proceso de inferencia. Misma entrada → misma salida, siempre.
    """

    def __init__(self):
        self._arbol = None        # Modelo DecisionTreeClassifier
        self._reglas: list = []   # Reglas IF-THEN extraídas
        self._cargado: bool = False

    # ------------------------------------------------------------------
    # MÉTODO: cargar_modelo
    # ------------------------------------------------------------------

    def cargar_modelo(self) -> None:
        """Carga el árbol de decisión y las reglas desde disco."""
        if not os.path.exists(RUTA_MODELO):
            raise FileNotFoundError(
                f"No se encontró el modelo en {RUTA_MODELO}.\n"
                "Ejecute primero: python src/train_model.py"
            )
        if not os.path.exists(RUTA_REGLAS):
            raise FileNotFoundError(
                f"No se encontró el archivo de reglas en {RUTA_REGLAS}."
            )

        with open(RUTA_MODELO, "rb") as f:
            self._arbol = pickle.load(f)
        with open(RUTA_REGLAS, "rb") as f:
            self._reglas = pickle.load(f)

        self._cargado = True

    # ------------------------------------------------------------------
    # MÉTODO: diagnosticar
    # ------------------------------------------------------------------

    def diagnosticar(self, datos_paciente: dict) -> ResultadoDiagnostico:
        """
        Ejecuta la inferencia determinista del sistema experto.

        El método recorre el árbol de decisión desde la raíz hasta una hoja,
        evaluando en cada nodo interno si el valor del paciente ≤ umbral
        (rama izquierda) o > umbral (rama derecha).

        Parámetros
        ----------
        datos_paciente : dict
            Diccionario {nombre_columna: valor} con las 11 variables médicas.

        Retorna
        -------
        ResultadoDiagnostico
            Objeto con diagnóstico y traza completa del razonamiento.
        """
        if not self._cargado:
            raise RuntimeError("Cargue el modelo primero con cargar_modelo().")

        # Construir vector de entrada en el orden correcto
        vector = np.array(
            [[datos_paciente[col] for col in FEATURE_NAMES]],
            dtype=float,
        )

        tree = self._arbol.tree_
        resultado = ResultadoDiagnostico()
        resultado.datos_paciente = datos_paciente
        pasos = []

        # ── Recorrido determinista del árbol ──────────────────────────────
        nodo_actual = 0  # Siempre empezamos desde la raíz (nodo 0)

        while tree.children_left[nodo_actual] != -1:
            # Es un nodo interno (tiene hijos)
            feat_idx = tree.feature[nodo_actual]
            umbral = tree.threshold[nodo_actual]
            nombre_col = FEATURE_NAMES[feat_idx]
            valor_paciente = float(vector[0, feat_idx])

            info = VARIABLE_INFO.get(nombre_col, {})
            nombre_desc = info.get("nombre_mostrado", nombre_col)
            unidad = info.get("unidad", "")

            condicion_texto = (
                f"{nombre_desc} ({nombre_col}) ≤ {umbral:.4f} {unidad}".strip()
            )

            # Evaluación determinista: ¿Se cumple la condición?
            condicion_cumplida = valor_paciente <= umbral

            paso = PasoRazonamiento(
                numero_nodo=nodo_actual,
                profundidad=int(np.sum(tree.feature[:nodo_actual] >= 0)),
                variable=nombre_col,
                umbral=umbral,
                valor_paciente=valor_paciente,
                condicion=condicion_texto,
                resultado=condicion_cumplida,
            )
            pasos.append(paso)

            # Moverse al hijo correspondiente
            if condicion_cumplida:
                nodo_actual = tree.children_left[nodo_actual]
            else:
                nodo_actual = tree.children_right[nodo_actual]

        # ── Nodo hoja alcanzado ───────────────────────────────────────────
        conteos = tree.value[nodo_actual][0]
        clase_predicha = int(np.argmax(conteos))
        total_muestras = int(np.sum(conteos))
        confianza = float(conteos[clase_predicha] / total_muestras)
        proporciones = conteos / total_muestras
        gini_hoja = float(1.0 - np.sum(proporciones ** 2))

        resultado.clase_predicha = clase_predicha
        resultado.pasos = pasos
        resultado.nodo_hoja = nodo_actual
        resultado.muestras_hoja = total_muestras
        resultado.gini_hoja = round(gini_hoja, 4)
        resultado.confianza = round(confianza, 4)

        # ── Buscar la regla IF-THEN coincidente ───────────────────────────
        resultado.regla_activada = self._buscar_regla_coincidente(pasos, clase_predicha)

        return resultado

    # ------------------------------------------------------------------
    # MÉTODO PRIVADO: _buscar_regla_coincidente
    # ------------------------------------------------------------------

    def _buscar_regla_coincidente(
        self, pasos: list, clase_predicha: int
    ) -> dict:
        """
        Busca entre las reglas generadas aquella que mejor corresponde
        al recorrido realizado por el árbol.

        La búsqueda compara las condiciones del recorrido con las de cada
        regla y selecciona la de mayor solapamiento.

        Retorna el dict de la regla o un dict vacío si no se encuentra.
        """
        if not self._reglas:
            return {}

        # Reconstruir las condiciones del recorrido actual
        condiciones_recorrido = set()
        for paso in pasos:
            info = VARIABLE_INFO.get(paso.variable, {})
            nombre_desc = info.get("nombre_mostrado", paso.variable)
            if paso.resultado:
                cond = f"{nombre_desc} ({paso.variable}) ≤ {paso.umbral:.2f}"
            else:
                cond = f"{nombre_desc} ({paso.variable}) > {paso.umbral:.2f}"
            condiciones_recorrido.add(cond)

        mejor_regla = None
        mejor_score = -1

        for regla in self._reglas:
            if regla["clase"] != clase_predicha:
                continue
            condiciones_regla = set(regla["condiciones"])
            # Score = número de condiciones en común
            score = len(condiciones_recorrido & condiciones_regla)
            if score > mejor_score:
                mejor_score = score
                mejor_regla = regla

        return mejor_regla if mejor_regla else {}

    # ------------------------------------------------------------------
    # MÉTODO: mostrar_todas_las_reglas
    # ------------------------------------------------------------------

    def mostrar_todas_las_reglas(self) -> None:
        """Muestra todas las reglas IF-THEN de la base de conocimiento."""
        if not self._reglas:
            print("  No hay reglas cargadas.")
            return

        imprimir_encabezado(f"BASE DE CONOCIMIENTO — {len(self._reglas)} REGLAS IF-THEN")
        for i, regla in enumerate(self._reglas, start=1):
            etiqueta = "CON ENFERMEDAD" if regla["clase"] == 1 else "SIN ENFERMEDAD"
            print(f"\n  REGLA #{i:03d}  →  {etiqueta}")
            print(f"  Muestras: {regla['muestras']}  |  Gini: {regla['gini']}  |  Confianza: {regla['confianza']*100:.1f}%")
            print("  CONDICIONES (todas deben cumplirse):")
            for cond in regla["condiciones"]:
                print(f"    [AND]  {cond}")
            print(f"  DIAGNÓSTICO: {etiqueta}")

    # ------------------------------------------------------------------
    # MÉTODO: mostrar_estadisticas_reglas
    # ------------------------------------------------------------------

    def mostrar_estadisticas_reglas(self) -> None:
        """Muestra un resumen estadístico de las reglas generadas."""
        if not self._reglas:
            print("  No hay reglas cargadas.")
            return

        total = len(self._reglas)
        reglas_0 = [r for r in self._reglas if r["clase"] == 0]
        reglas_1 = [r for r in self._reglas if r["clase"] == 1]

        imprimir_encabezado("ESTADÍSTICAS DE LA BASE DE CONOCIMIENTO")
        print(f"\n  Total de reglas              : {total}")
        print(f"  Reglas → Sin enfermedad (0)  : {len(reglas_0)}")
        print(f"  Reglas → Con enfermedad (1)  : {len(reglas_1)}")

        confianzas = [r["confianza"] for r in self._reglas]
        print(f"\n  Confianza promedio de reglas : {np.mean(confianzas)*100:.1f}%")
        print(f"  Confianza mínima             : {min(confianzas)*100:.1f}%")
        print(f"  Confianza máxima             : {max(confianzas)*100:.1f}%")

        ginis = [r["gini"] for r in self._reglas]
        print(f"\n  Gini promedio en hojas       : {np.mean(ginis):.4f}")
        print(f"  Hojas perfectamente puras    : {sum(1 for g in ginis if g == 0.0)}")

        longitudes = [len(r["condiciones"]) for r in self._reglas]
        print(f"\n  Longitud media de regla      : {np.mean(longitudes):.1f} condiciones")
        print(f"  Regla más corta              : {min(longitudes)} condiciones")
        print(f"  Regla más larga              : {max(longitudes)} condiciones")

    # ------------------------------------------------------------------
    # MÉTODO: mostrar_variables
    # ------------------------------------------------------------------

    def mostrar_variables(self) -> None:
        """Imprime la descripción completa de todas las variables médicas."""
        from utils import describir_variable
        imprimir_encabezado("VARIABLES MÉDICAS DEL SISTEMA")
        for col in FEATURE_NAMES:
            print(f"\n{'─' * 55}")
            print(describir_variable(col))
        print(f"{'─' * 55}")

    # ------------------------------------------------------------------
    # PROPIEDAD: cargado
    # ------------------------------------------------------------------

    @property
    def cargado(self) -> bool:
        """Indica si el modelo ha sido cargado exitosamente."""
        return self._cargado

    @property
    def num_reglas(self) -> int:
        """Retorna el número de reglas en la base de conocimiento."""
        return len(self._reglas)
