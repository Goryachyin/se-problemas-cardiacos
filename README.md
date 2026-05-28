# Sistema Experto Determinista para Diagnóstico de Enfermedades Cardíacas

> **Proyecto Académico** — Inteligencia Artificial Simbólica  
> Árbol de Decisión + Reglas IF-THEN + Inferencia Determinista

---

## Índice

1. [Introducción](#introducción)
2. [Objetivo](#objetivo)
3. [Sistema Experto Determinista](#sistema-experto-determinista)
4. [IA Simbólica](#ia-simbólica)
5. [Dataset](#dataset)
6. [Variables Médicas](#variables-médicas)
7. [Flujo del Programa](#flujo-del-programa)
8. [Arquitectura del Sistema](#arquitectura-del-sistema)
9. [Árbol de Decisión](#árbol-de-decisión)
10. [Fórmulas Matemáticas](#fórmulas-matemáticas)
11. [Ejemplos de Ejecución](#ejemplos-de-ejecución)
12. [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)
13. [Por qué es Determinista](#por-qué-es-determinista)
14. [Ventajas y Limitaciones](#ventajas-y-limitaciones)
15. [Bibliografía](#bibliografía)

---

## Introducción

Este proyecto implementa un **Sistema Experto Determinista** para el diagnóstico de enfermedades cardíacas. A diferencia de los sistemas modernos basados en redes neuronales o aprendizaje probabilístico, este sistema utiliza **IA Simbólica**: su conocimiento está representado explícitamente como reglas IF-THEN derivadas de un árbol de decisión.

El resultado es un sistema completamente **explicable**, **trazable** y **reproducible**: para cualquier combinación de variables médicas, el sistema siempre producirá el mismo diagnóstico y podrá explicar exactamente qué condiciones llevaron a esa conclusión.

---

## Objetivo

Desarrollar un sistema capaz de:

- Aprender automáticamente a partir de un dataset real de enfermedades cardíacas.
- Generar reglas de diagnóstico interpretables en lenguaje IF-THEN.
- Evaluar a un paciente recorriendo determinísticamente el árbol de decisión.
- Explicar en detalle el razonamiento utilizado para llegar al diagnóstico.
- Servir como herramienta de apoyo académico para comprender la IA simbólica.

---

## Sistema Experto Determinista

Un **sistema experto** es un programa de computadora que simula la capacidad de razonamiento de un experto humano en un dominio específico. Sus componentes principales son:

| Componente           | Descripción                                                    | En este proyecto                       |
|----------------------|----------------------------------------------------------------|----------------------------------------|
| **Base de conocimiento** | Conjunto de reglas y hechos del dominio                    | Reglas IF-THEN del árbol de decisión   |
| **Motor de inferencia**  | Mecanismo que aplica las reglas a los datos de entrada      | Recorrido determinista del árbol       |
| **Base de hechos**       | Datos del caso actual                                      | Variables médicas del paciente         |
| **Módulo de explicación**| Justificación del razonamiento                             | Traza nodo a nodo del árbol            |

El sistema es **determinista** porque:
- No utiliza probabilidades en la fase de inferencia.
- El árbol de decisión tiene una única ruta posible para cada entrada.
- Para los mismos valores de entrada, siempre se produce el mismo diagnóstico.

---

## IA Simbólica

La **Inteligencia Artificial Simbólica** (también llamada IA clásica o GOFAI — *Good Old-Fashioned AI*) representa el conocimiento mediante estructuras explícitas y manipulables: reglas lógicas, árboles, grafos y ontologías.

**Contraste con IA conexionista (redes neuronales):**

| Característica      | IA Simbólica (este proyecto) | IA Conexionista (redes neuronales) |
|---------------------|------------------------------|------------------------------------|
| Representación      | Reglas IF-THEN explícitas    | Pesos numéricos opacos             |
| Explicabilidad      | Total — se puede auditar     | Limitada — "caja negra"            |
| Determinismo        | Garantizado                  | Depende de la arquitectura         |
| Datos necesarios    | Pocos miles                  | Millones                           |
| Interpretación      | Directa                      | Requiere técnicas especiales (SHAP, LIME) |

---

## Dataset

**Nombre:** Heart Statlog Cleveland Hungary Final  
**Fuente:** [Kaggle — Enfermedades del Corazón](https://www.kaggle.com/datasets/oliverquiros/enfermedades-del-corazon)  
**Origen:** Combinación de los datasets Cleveland, Hungarian y Statlog del UCI Machine Learning Repository.

### Características del dataset

| Característica         | Valor                     |
|------------------------|---------------------------|
| Total de registros     | ~1,190 pacientes          |
| Variables de entrada   | 11 variables médicas      |
| Variable objetivo      | Diagnóstico (0/1)         |
| Clases                 | 0 = Sin enfermedad, 1 = Con enfermedad |
| Tipo de problema       | Clasificación binaria     |

### Distribución de clases

El dataset está relativamente balanceado, con aproximadamente 49% de casos positivos (enfermedad cardíaca) y 51% negativos, lo que lo hace adecuado para entrenamiento sin técnicas de balanceo.

---

## Variables Médicas

### 1. Edad (`age`)
- **Tipo:** Entero (años)
- **Rango típico:** 20 – 80 años
- **Descripción:** Edad cronológica del paciente.
- **Influencia en el diagnóstico:** El envejecimiento deteriora la elasticidad arterial y favorece la acumulación de placa aterosclerótica. Pacientes mayores de 50 años presentan un riesgo significativamente elevado.

### 2. Sexo (`sex`)
- **Tipo:** Binario (0 = Femenino, 1 = Masculino)
- **Descripción:** Sexo biológico del paciente.
- **Influencia:** Los hombres tienen mayor incidencia de cardiopatía isquémica en edades medias (40–55 años). Después de la menopausia, el riesgo en mujeres se equipara al masculino.

### 3. Tipo de dolor de pecho (`chest pain type`)
- **Tipo:** Categórico (1–4)
- **Opciones:**
  - `1` = Angina típica: dolor constrictivo por esfuerzo que irradia al brazo/mandíbula
  - `2` = Angina atípica: síntomas parciales de angina
  - `3` = Dolor no anginoso: dolor torácico sin relación con isquemia
  - `4` = Asintomático: sin dolor, isquemia silente
- **Influencia:** La angina típica es el marcador clínico más específico de isquemia. El tipo 4 (asintomático) puede indicar enfermedad avanzada sin manifestación.

### 4. Presión arterial en reposo (`resting bp s`)
- **Tipo:** Entero (mmHg)
- **Rango normal:** 90 – 120 mmHg
- **Umbral hipertensión:** > 140 mmHg
- **Influencia:** La hipertensión crónica daña el endotelio vascular, acelera la aterosclerosis y aumenta la carga de trabajo del ventrículo izquierdo.

### 5. Colesterol sérico (`cholesterol`)
- **Tipo:** Entero (mg/dl)
- **Clasificación:**
  - Deseable: < 200 mg/dl
  - Borderline alto: 200–239 mg/dl
  - Alto: ≥ 240 mg/dl
- **Influencia:** El LDL-colesterol elevado se deposita en las paredes arteriales, formando placas que reducen el lumen vascular y pueden provocar trombosis.

### 6. Azúcar en sangre en ayunas (`fasting blood sugar`)
- **Tipo:** Binario (0 = ≤120 mg/dl, 1 = >120 mg/dl)
- **Descripción:** Indicador indirecto de diabetes mellitus.
- **Influencia:** La hiperglucemia crónica daña las paredes vasculares mediante glucosilación del endotelio, multiplicando el riesgo cardiovascular 2–4 veces.

### 7. Electrocardiograma en reposo (`resting ecg`)
- **Tipo:** Categórico (0–2)
- **Opciones:**
  - `0` = Normal
  - `1` = Anomalía en onda ST-T (inversión de onda T, elevación/depresión de ST)
  - `2` = Hipertrofia ventricular izquierda por criterio de Estes
- **Influencia:** Las alteraciones del ECG reflejan daño estructural o eléctrico del miocardio. La hipertrofia ventricular indica sobrecarga crónica.

### 8. Frecuencia cardíaca máxima (`max heart rate`)
- **Tipo:** Entero (latidos por minuto)
- **Valor teórico máximo:** 220 - edad (años)
- **Influencia:** Una FCmáx reducida para la edad indica menor reserva funcional cardíaca. Valores altos son señal de buena condición cardiovascular.

### 9. Angina inducida por ejercicio (`exercise angina`)
- **Tipo:** Binario (0 = No, 1 = Sí)
- **Descripción:** Presencia de dolor de pecho durante prueba de esfuerzo.
- **Influencia:** La angina de esfuerzo es el síntoma clásico de obstrucción coronaria significativa (estenosis > 70%). El corazón no puede satisfacer la demanda de oxígeno aumentada.

### 10. Depresión del segmento ST (`oldpeak`)
- **Tipo:** Decimal (mm)
- **Descripción:** Depresión del segmento ST durante ejercicio comparado con el reposo, medida en el ECG.
- **Influencia:** Una depresión ≥ 1 mm es criterio diagnóstico de isquemia miocárdica. A mayor depresión, mayor severidad de la obstrucción coronaria.

### 11. Pendiente del segmento ST (`ST slope`)
- **Tipo:** Categórico (0–3)
- **Opciones:**
  - `0` = Sin pendiente definida
  - `1` = Ascendente (respuesta normal al esfuerzo)
  - `2` = Horizontal (borderline patológico)
  - `3` = Descendente (altamente patológico, indicativo de isquemia severa)
- **Influencia:** La pendiente descendente del ST durante ejercicio es uno de los indicadores más específicos de enfermedad arterial coronaria obstructiva.

---

## Flujo del Programa

```
┌─────────────────────────────────────────────────────┐
│              INICIO DEL PROGRAMA                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│    ¿Existe modelo entrenado en data/?               │
│          SÍ ──────────────────────────► Cargar modelo
│          NO ──► Cargar dataset                      │
│                 Entrenar árbol de decisión           │
│                 Generar reglas IF-THEN              │
│                 Guardar modelo en data/             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                 MENÚ PRINCIPAL                      │
│  [1] Diagnosticar paciente                          │
│  [2] Ver base de conocimiento                       │
│  [3] Ver variables médicas                          │
│  [4] Estadísticas del sistema                       │
│  [5] Re-entrenar modelo                             │
│  [6] Ver teoría matemática                          │
│  [7] Salir                                          │
└──────────────────────┬──────────────────────────────┘
                       │ Opción [1]
                       ▼
┌─────────────────────────────────────────────────────┐
│         RECOLECCIÓN DE DATOS DEL PACIENTE           │
│  Ingresar 11 variables médicas (con validación)     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           INFERENCIA DETERMINISTA                   │
│  Nodo raíz → evaluar condición → rama izquierda/   │
│  derecha → siguiente nodo → ... → nodo hoja        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              RESULTADO Y EXPLICACIÓN                │
│  • Diagnóstico (0 = sin / 1 = con enfermedad)      │
│  • Traza paso a paso de cada nodo recorrido        │
│  • Regla IF-THEN activada                           │
│  • Datos del paciente evaluado                      │
└─────────────────────────────────────────────────────┘
```

---

## Arquitectura del Sistema

```
luzv2/
│
├── data/
│   ├── heart_statlog_cleveland_hungary_final.csv  ← Dataset original
│   ├── modelo_arbol.pkl                           ← Árbol serializado
│   └── reglas.pkl                                 ← Reglas IF-THEN serializadas
│
├── images/
│   └── arbol_decision.png                         ← Visualización del árbol
│
├── src/
│   ├── main.py           ← Punto de entrada, interfaz de consola
│   ├── train_model.py    ← Entrenamiento, evaluación y extracción de reglas
│   ├── expert_system.py  ← Motor de inferencia determinista
│   └── utils.py          ← Carga de datos, validaciones, metadatos médicos
│
├── tree.png              ← Copia del árbol en raíz del proyecto
├── requirements.txt      ← Dependencias Python
└── README.md             ← Este documento
```

### Responsabilidades de cada módulo

| Módulo             | Responsabilidad                                               |
|--------------------|---------------------------------------------------------------|
| `utils.py`         | Metadatos de variables, carga de CSV, validación de entradas |
| `train_model.py`   | Clase `ModeloArbol`: entrenar, evaluar, extraer reglas, visualizar |
| `expert_system.py` | Clase `SistemaExperto`: motor de inferencia, traza del razonamiento |
| `main.py`          | Clase `AplicacionConsola`: menú interactivo, orquestación    |

---

## Árbol de Decisión

### ¿Qué es un árbol de decisión?

Un árbol de decisión es una estructura jerárquica donde:
- **Nodos internos**: preguntan sobre una variable (`¿Pendiente ST ≤ 1.5?`)
- **Ramas**: representan las dos posibles respuestas (SÍ / NO)
- **Hojas**: contienen la decisión final (clase mayoritaria)

### Estructura del árbol generado

El árbol se configura con los siguientes hiperparámetros:

| Hiperparámetro         | Valor | Justificación                              |
|------------------------|-------|--------------------------------------------|
| `criterion`            | gini  | Criterio estándar para clasificación       |
| `max_depth`            | 5     | Evita sobreajuste, mantiene legibilidad    |
| `min_samples_leaf`     | 10    | Evita reglas sobre casos aislados          |
| `random_state`         | 42    | Garantiza reproducibilidad total           |

### Visualización

El árbol se guarda automáticamente en:
- `images/arbol_decision.png`
- `tree.png` (copia en la raíz del proyecto)

Cada nodo del árbol muestra:
- La variable y umbral de la división
- El índice Gini del nodo
- El número de muestras que llegan al nodo
- La distribución de clases

### Explicación de nodos, ramas y hojas

**Nodo raíz:** Primera pregunta del árbol. Usa la variable que mejor separa las dos clases (mayor ganancia de información / menor Gini ponderado).

**Nodos internos:** Cada nivel profundiza la clasificación usando la variable más informativa disponible dado el subconjunto de datos.

**Ramas izquierdas:** Condición cumplida (valor ≤ umbral).

**Ramas derechas:** Condición no cumplida (valor > umbral).

**Hojas:** Nodos sin hijos. Contienen la predicción: la clase que tiene más ejemplos de entrenamiento en ese subconjunto.

---

## Fórmulas Matemáticas

### 1. Índice de Gini

Mide la "impureza" de un nodo. Un nodo puro tiene Gini = 0.

$$\text{Gini}(t) = 1 - \sum_{i=1}^{K} p(i|t)^2$$

Donde $p(i|t)$ es la proporción de ejemplos de clase $i$ en el nodo $t$, y $K$ es el número de clases.

**Ejemplo numérico:**
- Nodo con 70 pacientes con enfermedad y 30 sin enfermedad:
$$\text{Gini} = 1 - (0.70^2 + 0.30^2) = 1 - (0.49 + 0.09) = 0.42$$

- Nodo perfectamente puro (100 pacientes con enfermedad):
$$\text{Gini} = 1 - (1.00^2) = 0$$

### 2. Entropía (Shannon)

Medida de incertidumbre o desorden de un nodo.

$$H(t) = -\sum_{i=1}^{K} p(i|t) \cdot \log_2 p(i|t)$$

**Casos extremos:**
- Máxima incertidumbre (50%/50%): $H = -(0.5 \cdot \log_2 0.5 + 0.5 \cdot \log_2 0.5) = 1.0$ bit
- Nodo puro: $H = 0$ bits

### 3. Ganancia de Información

Cuantifica cuánto reduce la impureza dividir por un atributo $A$.

$$\text{IG}(A, t) = H(t) - \sum_{j} \frac{|t_j|}{|t|} \cdot H(t_j)$$

El árbol selecciona en cada nodo el atributo $A^*$ que maximiza la ganancia:

$$A^* = \arg\max_A \; \text{IG}(A, t)$$

### 4. Gini ponderado de un split

Alternativa al IG usando el índice Gini:

$$\text{Gini}_{\text{split}} = \frac{|t_L|}{|t|} \cdot \text{Gini}(t_L) + \frac{|t_R|}{|t|} \cdot \text{Gini}(t_R)$$

Se selecciona el split que minimiza este valor.

### 5. Clasificación determinista (inferencia)

Dado el árbol $T$ con parámetros aprendidos y un vector de entrada $\mathbf{x} \in \mathbb{R}^{11}$:

$$\hat{y} = T(\mathbf{x}) = \text{clase}_{\text{hoja}}(n^*)$$

donde $n^*$ es el nodo hoja alcanzado siguiendo:

$$n_{k+1} = \begin{cases} \text{hijo\_izq}(n_k) & \text{si } x_{f(n_k)} \leq \theta_{n_k} \\ \text{hijo\_der}(n_k) & \text{si } x_{f(n_k)} > \theta_{n_k} \end{cases}$$

siendo $f(n_k)$ el índice de la variable y $\theta_{n_k}$ el umbral del nodo $n_k$.

---

## Ejemplos de Ejecución

### Caso 1: Paciente de alto riesgo

```
Ingrese datos del paciente:
  Edad                        : 63
  Sexo                        : 1  (Masculino)
  Tipo de dolor de pecho      : 4  (Asintomático)
  Presión arterial en reposo  : 145
  Colesterol sérico           : 233
  Azúcar en sangre en ayunas  : 1  (> 120 mg/dl)
  ECG en reposo               : 0  (Normal)
  Frecuencia cardíaca máxima  : 150
  Angina por ejercicio        : 0  (No)
  Oldpeak (depresión ST)      : 2.3
  Pendiente ST                : 3  (Descendente)

DIAGNÓSTICO: POSIBLE ENFERMEDAD CARDÍACA

Traza del razonamiento:
  Paso 1 | Nodo 000 | Pendiente ST ≤ 1.50?  → NO (valor: 3) → Rama derecha
  Paso 2 | Nodo 004 | Frecuencia cardíaca máxima ≤ 151.00? → SÍ → Rama izquierda
  Paso 3 | Nodo 008 | Oldpeak ≤ 0.75?  → NO (valor: 2.3) → Rama derecha
  → Hoja alcanzada: CON ENFERMEDAD (Gini=0.12, Confianza=94%)

Regla activada:
  SI   Pendiente ST > 1.50
  AND  Frecuencia cardíaca máxima ≤ 151.00
  AND  Oldpeak > 0.75
  ENTONCES: CON ENFERMEDAD CARDÍACA
```

### Caso 2: Paciente de bajo riesgo

```
Ingrese datos del paciente:
  Edad                        : 35
  Sexo                        : 0  (Femenino)
  Tipo de dolor de pecho      : 2  (Angina atípica)
  Presión arterial en reposo  : 118
  Colesterol sérico           : 182
  Azúcar en sangre en ayunas  : 0  (Normal)
  ECG en reposo               : 0  (Normal)
  Frecuencia cardíaca máxima  : 174
  Angina por ejercicio        : 0  (No)
  Oldpeak (depresión ST)      : 0.0
  Pendiente ST                : 1  (Ascendente)

DIAGNÓSTICO: SIN ENFERMEDAD CARDÍACA

Traza del razonamiento:
  Paso 1 | Nodo 000 | Pendiente ST ≤ 1.50?  → SÍ (valor: 1) → Rama izquierda
  Paso 2 | Nodo 001 | Oldpeak ≤ 0.45?  → SÍ (valor: 0.0) → Rama izquierda
  Paso 3 | Nodo 002 | Tipo de dolor ≤ 3.50?  → SÍ → Rama izquierda
  → Hoja alcanzada: SIN ENFERMEDAD (Gini=0.07, Confianza=96%)
```

---

## Cómo Ejecutar el Proyecto

### Requisitos previos

- Python 3.8 o superior
- pip

### Instalación

```bash
# 1. Clonar/descargar el proyecto
cd ruta/al/proyecto/luzv2

# 2. Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar el sistema experto

```bash
# Desde la carpeta raíz del proyecto
cd src
python main.py
```

El sistema detectará automáticamente si existe un modelo entrenado:
- **Primera ejecución**: entrena el árbol, genera reglas, guarda imágenes y serializa el modelo.
- **Ejecuciones posteriores**: carga el modelo directamente (inicio rápido).

### Entrenar solo el modelo

```bash
cd src
python train_model.py
```

Esto genera:
- `data/modelo_arbol.pkl` — árbol serializado
- `data/reglas.pkl` — reglas IF-THEN
- `images/arbol_decision.png` — visualización del árbol
- `tree.png` — copia del árbol en la raíz

---

## Por qué es Determinista

El sistema garantiza **determinismo total** mediante tres mecanismos:

### 1. Semilla fija en el entrenamiento

```python
DecisionTreeClassifier(random_state=42)
train_test_split(..., random_state=42)
```

Aunque el árbol no necesita aleatoriedad, sklearn usa un generador de números aleatorios para desempatar splits iguales. Fijar `random_state=42` garantiza que el mismo dataset produce siempre el mismo árbol.

### 2. Inferencia sin aleatoriedad

El recorrido del árbol es una función matemática pura:

```
f(x) → camino único → hoja única → clase única
```

No hay muestreo, no hay probabilidades condicionales, no hay procesos estocásticos.

### 3. Reglas deterministas

Cada regla IF-THEN es una conjunción de condiciones booleanas. Para un vector `x` dado:
- Exactamente una regla es compatible con `x`.
- Esa regla siempre asigna la misma clase.

---

## Ventajas y Limitaciones

### Ventajas

| Ventaja                  | Descripción                                                      |
|--------------------------|------------------------------------------------------------------|
| **Explicabilidad total** | Se puede rastrear cada decisión hasta las condiciones originales |
| **Determinismo**         | Sin componente aleatorio en la inferencia                        |
| **Eficiencia**           | Inferencia en O(profundidad) — extremadamente rápida            |
| **Auditabilidad**        | Médicos y reguladores pueden revisar las reglas                  |
| **Sin caja negra**       | No hay parámetros ocultos ni representaciones latentes          |
| **Pocos datos**          | Funciona bien con datasets de cientos de registros              |

### Limitaciones

| Limitación              | Descripción                                                       |
|-------------------------|-------------------------------------------------------------------|
| **Fronteras lineales**  | El árbol solo crea fronteras ortogonales (paralelas a los ejes)  |
| **Sobreajuste**         | Sin `max_depth`, el árbol memoriza datos de entrenamiento        |
| **Variables continuas** | Los umbrales discretizan variables continuas perdiendo información |
| **Sesgo de varianza**   | Un árbol solo puede ser inestable con pequeñas variaciones en datos |
| **No captura interacciones complejas** | Relaciones no lineales entre variables pueden perderse |

### Cuándo usar este enfoque

✅ Cuando la **explicabilidad** es un requisito (medicina, derecho, finanzas reguladas)  
✅ Cuando el dataset es moderado (< 100,000 registros)  
✅ Cuando se necesita **auditar** las decisiones  
✅ En entornos académicos o de investigación  
✅ Como baseline interpretable antes de modelos más complejos  

---

## Bibliografía

1. **Breiman, L., Friedman, J., Olshen, R., & Stone, C.** (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole.

2. **Quinlan, J. R.** (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81–106.

3. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

4. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

5. **Gini, C.** (1912). *Variabilità e mutabilità*. Tipografia di Paolo Cuppini, Bologna.

6. **Shannon, C. E.** (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423.

7. **UCI Machine Learning Repository** — Heart Disease Dataset. Frank, A., & Asuncion, A. (2010). *UCI Machine Learning Repository*. University of California, Irvine.

8. **World Health Organization (WHO).** (2023). *Cardiovascular Diseases (CVDs) — Key Facts*. Retrieved from https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

9. **Hayes-Roth, F., Waterman, D. A., & Lenat, D. B.** (1983). *Building Expert Systems*. Addison-Wesley.

10. **Mitchell, T. M.** (1997). *Machine Learning*. McGraw-Hill. Chapter 3: Decision Tree Learning.

---

> **Nota académica:** Este sistema es una herramienta educativa. No debe utilizarse como sustituto del diagnóstico médico profesional. Siempre consulte a un médico especialista para evaluaciones cardíacas reales.
#   s e - p r o b l e m a s - c a r d i a c o s  
 