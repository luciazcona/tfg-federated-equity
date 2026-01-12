# TFG: Estudio sobre el impacto de la distribución de los datos en la equidad de modelos de aprendizaje federado

Este repositorio contiene el código del Trabajo Fin de Grado titulado "Estudio sobre el impacto de la distribución de los datos en la equidad de modelos de aprendizaje federado".

El código implementa experimentos de aprendizaje federado sobre el dataset **Adult**, comparando modelos federados y centralizados en términos de **precisión** y **equidad** (Disparate Impact) bajo distintos escenarios de distribución de datos:
- **iid** (Independiente e idénticamente distribuidos)
- **no‑iid por variable sensible S**
- **no‑iid por etiqueta Y**

## Estructura del repositorio

- `tfg_experimento_equidad.ipynb`: notebook principal con los experimentos (iid, no‑iid por S, no‑iid por Y).
- `funciones_tfg.py`: módulo con todas las funciones auxiliares:
  - Carga y preprocesado de datos.
  - Reparto iid / no‑iid por variable sensible o por variable clase.
  - Entrenamiento federado (con TFF) y centralizado.
  - Cálculo de métricas de equidad (Disparate Impact, intervalos de confianza).
  - Funciones de visualización.
- `Figuras/`: carpeta con las gráficas generadas por el experimento (boxplots, curvas de loss, etc.).

## Cómo ejecutar el experimento

### Requisitos

El código se ha desarrollado y ejecutado en un entorno virtual de Python con las siguientes versiones:

- Python: `3.10.12`
- TensorFlow: `2.14.1`
- TensorFlow Federated: `0.71.0`
- pandas: `1.5.3`
- NumPy: `1.25.2`
- scikit-learn: `1.7.2`
- Matplotlib: `3.10.6`
- Seaborn: `0.13.2`
- SciPy: `1.9.3`

1. **Crear y activar el entorno virtual**:
    ```bash
    conda create -n mi_tfg python=3.10
    conda activate mi_tfg
    ```

2. **Instalar las librerías necesarias**:
    ```bash
    # Instalación de dependencias principales
    conda install pandas numpy scikit-learn matplotlib seaborn scipy

    # Instalación de TensorFlow y TensorFlow Federated
    pip install tensorflow==2.14.1
    pip install tensorflow-federated==0.71.0
    ```

3. **Instalar Jupyter y crear el kernel**:
    ```bash
    conda install jupyter
    python -m ipykernel install --user --name mi_tfg --display-name "TFG Federado"
    ```

### Preparar los datos

Asegúrate de tener los archivos necesarios en la carpeta **`adult_dataset/`**. Deberías colocar los archivos `adult.data.csv` y `adult.test.csv` en este directorio.

### Ejecutar el notebook

1. **Abrir Jupyter**:
    ```bash
    jupyter notebook
    ```

2. **Abrir el notebook `tfg_experimento_equidad.ipynb`**.

3. **Seleccionar el kernel adecuado**:
   - En la barra de menú de Jupyter, selecciona el kernel **"TFG Federado"**.

4. **Ejecutar todas las celdas**:
   - Una vez abierto el notebook, selecciona **"Ejecutar todo"** en el menú de Jupyter para ejecutar todas las celdas.

### Resultados

El experimento principal generará los siguientes resultados:

- **Intervalos de Confianza** para las métricas de rendimiento:
  - **Accuracy**
  - **Disparate Impact** (comparando modelos federados frente a centralizados)

- **Gráficas de Caja (Boxplots)** comparando las métricas de desempeño.

- **Curvas de Convergencia** del loss por ronda de entrenamiento.

### Archivos generados

- Las gráficas y figuras generadas se guardarán automáticamente en la carpeta **`Figuras/`**.

## Notas importantes

- **Diferentes distribuciones de datos**: El experimento cubre tres escenarios de distribución de datos:
  - **IID (Independent and Identically Distributed)**: Los datos son distribuidos de manera uniforme entre los distintos nodos.
  - **no‑IID por variable sensible S**: Los datos están distribuidos de forma no uniforme en función de una variable sensible (por ejemplo, género o raza).
  - **no‑IID por etiqueta Y**: Los datos se distribuyen según la etiqueta de clasificación, lo que puede afectar a la equidad del modelo.

- **Ejecución en entorno virtual**: Asegúrate de trabajar siempre dentro del entorno virtual creado para evitar problemas de dependencias.

- **Resultados reproducibles**: Para garantizar que los resultados sean reproducibles, sigue los pasos de instalación exactos descritos anteriormente.
