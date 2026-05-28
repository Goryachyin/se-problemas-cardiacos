# =============================================================================
# utils.py - Utilidades del Sistema Experto de Diagnóstico Cardíaco
# =============================================================================
# Contiene funciones de apoyo: carga de datos, descripción de variables,
# validación de entradas y formateo de resultados.
# =============================================================================

import pandas as pd
import os

# ---------------------------------------------------------------------------
# CONSTANTES: Nombres descriptivos de columnas y sus metadatos médicos
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "age",
    "sex",
    "chest pain type",
    "resting bp s",
    "cholesterol",
    "fasting blood sugar",
    "resting ecg",
    "max heart rate",
    "exercise angina",
    "oldpeak",
    "ST slope",
]

TARGET_NAME = "target"

# ---------------------------------------------------------------------------
# DESCRIPCIONES MÉDICAS DE CADA VARIABLE
# Cada entrada contiene:
#   - nombre_mostrado : etiqueta para el usuario
#   - descripcion     : qué mide la variable
#   - influencia      : cómo afecta el riesgo cardíaco
#   - rango_valido    : tupla (min, max) de valores aceptables
#   - tipo            : 'int' o 'float'
#   - opciones        : dict de valores posibles (para variables categóricas)
# ---------------------------------------------------------------------------

VARIABLE_INFO = {
    "age": {
        "nombre_mostrado": "Edad",
        "unidad": "años",
        "descripcion": (
            "Edad del paciente en años. A mayor edad, los vasos sanguíneos tienden "
            "a perder elasticidad y acumular placa arterial (aterosclerosis)."
        ),
        "influencia": (
            "El riesgo cardiovascular aumenta progresivamente con la edad. "
            "Pacientes >50 años tienen significativamente mayor probabilidad de "
            "enfermedad coronaria."
        ),
        "rango_valido": (1, 120),
        "tipo": "int",
        "opciones": None,
    },
    "sex": {
        "nombre_mostrado": "Sexo",
        "unidad": "",
        "descripcion": (
            "Sexo biológico del paciente. Los hombres tienen mayor riesgo cardíaco "
            "antes de los 55 años; después de la menopausia el riesgo en mujeres se iguala."
        ),
        "influencia": (
            "El sexo masculino (1) se asocia con mayor incidencia de enfermedades "
            "coronarias en edades medias."
        ),
        "rango_valido": (0, 1),
        "tipo": "int",
        "opciones": {0: "Femenino", 1: "Masculino"},
    },
    "chest pain type": {
        "nombre_mostrado": "Tipo de dolor de pecho",
        "unidad": "",
        "descripcion": (
            "Clasificación del dolor torácico según sus características clínicas. "
            "Es uno de los síntomas principales evaluados en cardiología."
        ),
        "influencia": (
            "La angina típica (tipo 1) es el síntoma más específico de isquemia "
            "miocárdica. El dolor asintomático (tipo 4) paradójicamente puede "
            "indicar enfermedad silente avanzada."
        ),
        "rango_valido": (1, 4),
        "tipo": "int",
        "opciones": {
            1: "Angina típica",
            2: "Angina atípica",
            3: "Dolor no anginoso",
            4: "Asintomático",
        },
    },
    "resting bp s": {
        "nombre_mostrado": "Presión arterial en reposo",
        "unidad": "mmHg",
        "descripcion": (
            "Presión arterial sistólica medida en reposo. Refleja la fuerza con "
            "que el corazón bombea sangre hacia las arterias."
        ),
        "influencia": (
            "Valores >140 mmHg (hipertensión) dañan las paredes arteriales con "
            "el tiempo, incrementando el riesgo de infarto y accidente cerebrovascular."
        ),
        "rango_valido": (50, 250),
        "tipo": "int",
        "opciones": None,
    },
    "cholesterol": {
        "nombre_mostrado": "Colesterol sérico",
        "unidad": "mg/dl",
        "descripcion": (
            "Nivel total de colesterol en sangre. El colesterol LDL (malo) se "
            "deposita en las arterias formando placas que las obstruyen."
        ),
        "influencia": (
            "Colesterol >200 mg/dl se considera borderline alto; >240 mg/dl es "
            "alto riesgo. Contribuye directamente a la aterosclerosis coronaria."
        ),
        "rango_valido": (0, 600),
        "tipo": "int",
        "opciones": None,
    },
    "fasting blood sugar": {
        "nombre_mostrado": "Azúcar en sangre en ayunas",
        "unidad": "",
        "descripcion": (
            "Indica si el nivel de glucosa en ayunas supera 120 mg/dl. "
            "La diabetes mellitus es un factor de riesgo cardiovascular mayor."
        ),
        "influencia": (
            "La hiperglucemia crónica daña el endotelio vascular y acelera la "
            "formación de placas ateroscleróticas en arterias coronarias."
        ),
        "rango_valido": (0, 1),
        "tipo": "int",
        "opciones": {0: "≤ 120 mg/dl (Normal)", 1: "> 120 mg/dl (Elevado)"},
    },
    "resting ecg": {
        "nombre_mostrado": "Electrocardiograma en reposo",
        "unidad": "",
        "descripcion": (
            "Resultado del electrocardiograma (ECG) en reposo. Mide la actividad "
            "eléctrica del corazón y detecta anomalías en el ritmo o estructura."
        ),
        "influencia": (
            "Anomalías en el ECG (hipertrofia ventricular, alteraciones del segmento ST) "
            "son indicadores directos de daño miocárdico o sobrecarga cardíaca."
        ),
        "rango_valido": (0, 2),
        "tipo": "int",
        "opciones": {
            0: "Normal",
            1: "Anomalía en onda ST-T",
            2: "Hipertrofia ventricular izquierda",
        },
    },
    "max heart rate": {
        "nombre_mostrado": "Frecuencia cardíaca máxima",
        "unidad": "lpm",
        "descripcion": (
            "Frecuencia cardíaca máxima alcanzada durante la prueba de esfuerzo. "
            "Refleja la reserva funcional del corazón."
        ),
        "influencia": (
            "Una frecuencia cardíaca máxima baja sugiere capacidad aeróbica reducida "
            "y posible disfunción cardíaca. Valores altos generalmente indican mejor "
            "condición cardiovascular."
        ),
        "rango_valido": (60, 220),
        "tipo": "int",
        "opciones": None,
    },
    "exercise angina": {
        "nombre_mostrado": "Angina inducida por ejercicio",
        "unidad": "",
        "descripcion": (
            "Indica si el paciente experimentó angina (dolor de pecho) durante "
            "la prueba de esfuerzo físico."
        ),
        "influencia": (
            "La angina de esfuerzo es un síntoma clásico de isquemia miocárdica: "
            "el corazón no recibe suficiente oxígeno bajo demanda, lo que sugiere "
            "obstrucción arterial significativa."
        ),
        "rango_valido": (0, 1),
        "tipo": "int",
        "opciones": {0: "No", 1: "Sí"},
    },
    "oldpeak": {
        "nombre_mostrado": "Depresión del segmento ST (oldpeak)",
        "unidad": "mm",
        "descripcion": (
            "Depresión del segmento ST inducida por el ejercicio relativa al reposo, "
            "medida en milímetros en el ECG."
        ),
        "influencia": (
            "Una depresión del ST ≥ 1 mm durante ejercicio es un marcador establecido "
            "de isquemia miocárdica. A mayor depresión, mayor probabilidad de "
            "enfermedad arterial coronaria significativa."
        ),
        "rango_valido": (0.0, 10.0),
        "tipo": "float",
        "opciones": None,
    },
    "ST slope": {
        "nombre_mostrado": "Pendiente del segmento ST",
        "unidad": "",
        "descripcion": (
            "Describe la forma de la pendiente del segmento ST durante la prueba "
            "de esfuerzo máximo en el electrocardiograma."
        ),
        "influencia": (
            "Una pendiente ascendente (1) es normal. La horizontal (2) o descendente (3) "
            "indican isquemia progresiva y se asocian con enfermedad coronaria severa."
        ),
        "rango_valido": (0, 3),
        "tipo": "int",
        "opciones": {
            0: "Sin pendiente / no aplica",
            1: "Ascendente (normal)",
            2: "Horizontal (intermedia)",
            3: "Descendente (patológica)",
        },
    },
}

TARGET_INFO = {
    0: {
        "etiqueta": "SIN ENFERMEDAD CARDÍACA",
        "color": "verde",
        "descripcion": "El análisis no detecta indicadores de enfermedad cardíaca significativa.",
    },
    1: {
        "etiqueta": "POSIBLE ENFERMEDAD CARDÍACA",
        "color": "rojo",
        "descripcion": "El análisis detecta indicadores compatibles con enfermedad cardíaca.",
    },
}


# ---------------------------------------------------------------------------
# FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ---------------------------------------------------------------------------

def cargar_dataset(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV y realiza limpieza básica.

    Parámetros
    ----------
    ruta : str
        Ruta absoluta o relativa al archivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio con las columnas esperadas.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta}")

    df = pd.read_csv(ruta)

    # Verificar columnas esperadas
    columnas_esperadas = FEATURE_NAMES + [TARGET_NAME]
    columnas_faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Columnas faltantes en el dataset: {columnas_faltantes}")

    # Eliminar filas con valores nulos
    filas_antes = len(df)
    df = df.dropna(subset=columnas_esperadas)
    filas_eliminadas = filas_antes - len(df)
    if filas_eliminadas > 0:
        print(f"  [INFO] Se eliminaron {filas_eliminadas} filas con valores nulos.")

    # Asegurar tipos de datos correctos
    for col in FEATURE_NAMES:
        info = VARIABLE_INFO.get(col, {})
        if info.get("tipo") == "float":
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)

    df[TARGET_NAME] = df[TARGET_NAME].astype(int)

    return df


def obtener_X_y(df: pd.DataFrame):
    """
    Separa el DataFrame en matriz de características X y vector objetivo y.

    Retorna
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return X, y


# ---------------------------------------------------------------------------
# FUNCIONES DE DESCRIPCIÓN Y VALIDACIÓN
# ---------------------------------------------------------------------------

def describir_variable(nombre_col: str) -> str:
    """
    Devuelve una descripción formateada de una variable médica.
    """
    info = VARIABLE_INFO.get(nombre_col)
    if info is None:
        return f"Variable '{nombre_col}' sin descripción disponible."

    lineas = [
        f"  Variable  : {info['nombre_mostrado']}",
        f"  Unidad    : {info['unidad'] or 'categórica'}",
        f"  Descripción: {info['descripcion']}",
        f"  Influencia : {info['influencia']}",
        f"  Rango     : [{info['rango_valido'][0]}, {info['rango_valido'][1]}]",
    ]
    if info["opciones"]:
        lineas.append("  Opciones  :")
        for k, v in info["opciones"].items():
            lineas.append(f"              {k} = {v}")
    return "\n".join(lineas)


def validar_valor(nombre_col: str, valor) -> tuple:
    """
    Valida que un valor se encuentre dentro del rango aceptable para la variable.

    Retorna
    -------
    (valido: bool, mensaje: str)
    """
    info = VARIABLE_INFO.get(nombre_col)
    if info is None:
        return True, ""

    rmin, rmax = info["rango_valido"]

    try:
        if info["tipo"] == "float":
            val = float(valor)
        else:
            val = int(float(valor))
    except (ValueError, TypeError):
        return False, f"Se esperaba un número {'decimal' if info['tipo'] == 'float' else 'entero'}."

    if not (rmin <= val <= rmax):
        return False, f"Valor fuera de rango. Debe estar entre {rmin} y {rmax}."

    if info["opciones"] and int(val) not in info["opciones"]:
        opciones_str = ", ".join(f"{k}={v}" for k, v in info["opciones"].items())
        return False, f"Valor no permitido. Opciones válidas: {opciones_str}"

    return True, ""


def formatear_separador(ancho: int = 60, caracter: str = "=") -> str:
    """Retorna una línea separadora de texto."""
    return caracter * ancho


def imprimir_encabezado(titulo: str, ancho: int = 60) -> None:
    """Imprime un encabezado formateado en consola."""
    sep = formatear_separador(ancho)
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
