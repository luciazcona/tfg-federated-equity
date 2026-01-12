import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import pandas as pd
import numpy as np
import random
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv  


# ------------------------------------------------------
"""
    Nombre: cargar_datos
    Descripción: Carga y combina los conjuntos de datos de entrenamiento y prueba del dataset Adult.
    Devuelve: DataFrame combinado de los datos de entrenamiento y prueba.
"""

def cargar_datos(ruta_train, ruta_test):
    cols = [
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "OrigEthn", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"
    ]
    df_train = pd.read_csv(ruta_train, names=cols, sep=r'\s*,\s*', engine='python', na_values="?")
    df_test = pd.read_csv(ruta_test, names=cols, sep=r'\s*,\s*', engine='python', na_values="?")
    df = pd.concat([df_test, df_train])
    df.reset_index(inplace=True, drop=True)
    return df

# ------------------------------------------------------
"""
    Nombre: preprocesar_datos
    Descripción: Limpia, transforma y codifica los datos del dataset Adult, aplicando one-hot encoding y escalado.
    Devuelve: Arrays X (características), y (etiqueta) y lista de nombres de columnas.
    """
def preprocesar_datos(df, variable_sensible):
    data = df.copy()
    data['Child'] = np.where(data['Relationship']=='Own-child', 'ChildYes', 'ChildNo')
    data['OrigEthn'] = np.where(data['OrigEthn']=='White', 'CaucYes', 'CaucNo')
    data = data.drop(columns=['fnlwgt','Relationship','Country','Education'])
    data = data.replace('<=50K.','<=50K')
    data = data.replace('>50K.','>50K')
    
    # One hot/preparación
    data_ohe = data.copy()
    data_ohe['Target'] = np.where(data_ohe['Target']=='>50K', 1., 0.)
    data_ohe['OrigEthn'] = np.where(data_ohe['OrigEthn']=='CaucYes', 1., 0.)
    data_ohe['Sex'] = np.where(data_ohe['Sex']=='Male', 1., 0.)
    for col in ['Workclass', 'Martial Status', 'Occupation', 'Child']:
        if len(set(list(data_ohe[col])))==2:
            val = data_ohe[col][0]
            data_ohe[col] = np.where(data_ohe[col]==val, 1., 0.)
        else:
            data_ohe = pd.get_dummies(data_ohe, prefix=[col], columns=[col])
    y = data_ohe['Target'].values.reshape(-1,1)
    data_ohe_wo_target = data_ohe.drop(columns=['Target'])
    X_col_names = list(data_ohe_wo_target.columns)
    X = data_ohe_wo_target.values
    # Randomizar todo
    shuffled_idx = np.random.permutation(len(X))
    X = X[shuffled_idx]
    y = y[shuffled_idx]
    # Escalado columnas numéricas
    num_cols = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 2]
    scaler = StandardScaler()
    X[:, num_cols] = scaler.fit_transform(X[:, num_cols])
    return X, y, X_col_names

# ------------------------------------------------------
"""
    Nombre: repartir_datos
    Descripción: Divide los datos entre clientes de forma IID o no-IID basándose en una variable sensible.
    Devuelve: Lista de diccionarios con 'X' y 'y' para cada cliente.
"""

def repartir_datos(X, y, variable_sensible, modo_reparto, num_clientes, porc_min=0.05, porc_max=0.95, X_col_names=None):
    """
    modo_reparto:
        0 -> iid
        1 -> no-iid por variable sensible S
    """
    idx_sensitive = np.where(np.array(X_col_names)==variable_sensible)[0][0]
    n = X.shape[0]
    repartos = []
    indices_usados = set()

    if modo_reparto == 1:
        # no-iid por S
        major_fracs = np.linspace(porc_min, porc_max, num_clientes)
        idx_group1 = np.where(X[:, idx_sensitive]==1)[0]
        idx_group0 = np.where(X[:, idx_sensitive]==0)[0]
        np.random.shuffle(idx_group1)
        np.random.shuffle(idx_group0)
        total_group1 = len(idx_group1)
        total_group0 = len(idx_group0)
        max_per_client = int(min(total_group0/np.sum(major_fracs), total_group1/np.sum(1-major_fracs)))
        pos1, pos0 = 0, 0
        for frac0 in major_fracs:
            n0 = int(round(max_per_client * frac0))
            n1 = max_per_client - n0
            idx0 = idx_group0[pos0:pos0 + n0]
            idx1 = idx_group1[pos1:pos1 + n1]
            pos0 += n0
            pos1 += n1
            indices = np.concatenate([idx0, idx1])
            indices_usados.update(indices)
            np.random.shuffle(indices)
            repartos.append({'X': X[indices], 'y': y[indices]})
        indices_restantes = np.array(list(set(range(n)) - indices_usados))
        if len(indices_restantes) > 0:
            repartos.append({'X': X[indices_restantes], 'y': y[indices_restantes]})
    else:
        # iid
        indices = np.random.permutation(n)
        size = n // num_clientes
        for i in range(num_clientes):
            start, end = i*size, (i+1)*size if i<num_clientes-1 else n
            idx = indices[start:end]
            indices_usados.update(idx)
            repartos.append({'X': X[idx], 'y': y[idx]})
        indices_restantes = np.array(list(set(range(n)) - indices_usados))
        if len(indices_restantes) > 0:
            repartos.append({'X': X[indices_restantes], 'y': y[indices_restantes]})

    repartos = [cl for cl in repartos if cl['X'].shape[0] > 0]
    return repartos

# ------------------------------------------------------
"""
    Nombre: seleccionar_clientes_sesgados
    Descripción: Selecciona subconjuntos de clientes según su distribución de la variable sensible.
    Devuelve: Lista de repartos filtrada según el modo de sesgo (mayoritario, minoritario o aleatorio).
"""

def seleccionar_clientes_sesgados(repartos, variable_sensible, m, X_col_names, modo_sesgo='mayoritario'):
    """
    modo_sesgo: 'mayoritario', 'minoritario' o 'aleatorio'
    """
    porcentajes = []

    if(variable_sensible=='Target'):
        for i, reparto in enumerate(repartos):
            y_cl = reparto['y'].flatten()
            total = y_cl.shape[0]
            grupo1 = np.sum(y_cl == 1)
            porcentaje = grupo1 / total if total > 0 else 0
            if porcentaje > 0 and porcentaje < 1:
                porcentajes.append((i, porcentaje))
    else:
        idx = X_col_names.index(variable_sensible)
        for i, reparto in enumerate(repartos):
            grupo1 = np.sum(reparto['X'][:, idx])
            total = reparto['X'].shape[0]
            porcentaje = grupo1 / total if total > 0 else 0
            if porcentaje > 0 and porcentaje < 1:
                porcentajes.append((i, porcentaje))
    
    if len(porcentajes) == 0:
        # si no hay clientes mixtos, usa todos
        return repartos

    if modo_sesgo == 'mayoritario':
        seleccionados = sorted(porcentajes, key=lambda x: x[1])[:m]
    elif modo_sesgo == 'minoritario':
        seleccionados = sorted(porcentajes, key=lambda x: -x[1])[:m]
    elif modo_sesgo == 'aleatorio':
        #indices_validos = [i for i, _ in porcentajes]
        indices_validos = list(range(len(repartos)))
        if m >= len(indices_validos):
            seleccionados = [(i, 0) for i in indices_validos]
        else:
            escogidos = np.random.choice(indices_validos, size=m, replace=False)
            seleccionados = [(i, 0) for i in escogidos]
    else:
        raise ValueError("modo_sesgo debe ser 'mayoritario', 'minoritario' o 'aleatorio'")

    return [repartos[i] for i, _ in seleccionados]

# ------------------------------------------------------
"""
    Nombre: repartir_datos_noiid_target
    Descripción: Reparte los datos de forma no-IID basándose en la variable target (etiqueta principal).
    Devuelve: Lista de repartos con X e y distribuidos por cliente.
"""

def repartir_datos_noiid_target(X, y, num_clientes, porc_min=0.05, porc_max=0.95):
    n = X.shape[0]
    repartos = []
    indices_usados = set()

    major_fracs = np.linspace(porc_min, porc_max, num_clientes)
    idx_target1 = np.where(y.flatten() == 1)[0]
    idx_target0 = np.where(y.flatten() == 0)[0]
    np.random.shuffle(idx_target1)
    np.random.shuffle(idx_target0)
    total1 = len(idx_target1)
    total0 = len(idx_target0)

    max_per_client = int(min(total0/np.sum(major_fracs), total1/np.sum(1-major_fracs)))
    pos1, pos0 = 0, 0

    for frac0 in major_fracs:
        n0 = int(round(max_per_client * frac0))
        n1 = max_per_client - n0
        idx0 = idx_target0[pos0:pos0 + n0]
        idx1 = idx_target1[pos1:pos1 + n1]
        pos0 += n0
        pos1 += n1
        indices = np.concatenate([idx0, idx1])
        indices_usados.update(indices)
        np.random.shuffle(indices)
        repartos.append({'X': X[indices], 'y': y[indices]})

    indices_restantes = np.array(list(set(range(n)) - indices_usados))
    if len(indices_restantes) > 0:
        repartos.append({'X': X[indices_restantes], 'y': y[indices_restantes]})
    repartos = [cl for cl in repartos if cl['X'].shape[0] > 0]
    return repartos

# ------------------------------------------------------
"""
    Nombre: crear_modelo
    Descripción: Crea y devuelve un modelo de red neuronal Keras con arquitectura simple o compleja.
    Devuelve: Modelo Keras compilado sin entrenar.
"""

def crear_modelo(tipo_modelo, input_shape):
    if tipo_modelo == 'simple':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(input_shape,))
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    return model

# ------------------------------------------------------
"""
    Nombre: df_to_tf_dataset
    Descripción: Convierte arrays NumPy de entrada y salida en un tf.data.Dataset para entrenamiento federado.
    Devuelve: Dataset TensorFlow preparado para entrenamiento.
"""

def df_to_tf_dataset(X_np, y_np, batch_size=16, num_local_epochs=1):
    X_np = X_np.astype('float32')
    y_np = y_np.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((X_np, y_np))
    ds = ds.repeat(num_local_epochs).shuffle(buffer_size=len(X_np)).batch(batch_size)
    return ds

# ------------------------------------------------------
"""
    Nombre: entrenar_federado
    Descripción: Entrena un modelo federado usando TFF (Federated Averaging) aplicando sesgo en la selección de clientes.
    Devuelve: Diccionario con el modelo entrenado y el historial de entrenamiento.
"""

def entrenar_federado(reparto, variable_sensible, rounds, c, X_col_names, modo_sesgo='mayoritario'):
    reparto = [cl for cl in reparto if cl['X'].shape[0] > 0]
    federated_data = [df_to_tf_dataset(r['X'], r['y']) for r in reparto]
    def model_fn():
        model = crear_modelo('simple', reparto[0]['X'].shape[1])
        return tff.learning.models.from_keras_model(
            keras_model=model,
            input_spec=federated_data[0].element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()])
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = iterative_process.initialize()
    history = []
    num_clients = len(federated_data)

    # 2. Corregir C=1 y evitar m=0
    m = int(max(1, round(c * num_clients)))
    m = min(m, num_clients)

    for r in range(1, rounds+1):
        seleccionados = seleccionar_clientes_sesgados(
            reparto, variable_sensible, m, X_col_names, modo_sesgo=modo_sesgo
        )
        selected_clients = [df_to_tf_dataset(cl['X'], cl['y']) for cl in seleccionados]
        result = iterative_process.next(state, selected_clients)
        state = result.state
        metrics = result.metrics
        history.append({
            "Ronda": r,
            "Loss": metrics['client_work']['train']['loss'],
            "Binary Accuracy": metrics['client_work']['train']['binary_accuracy'],
            "Num Examples": metrics['client_work']['train']['num_examples']
        })
    keras_model = crear_modelo('simple', reparto[0]['X'].shape[1])
    fed_weights = iterative_process.get_model_weights(state)
    fed_weights.assign_weights_to(keras_model)
    return {'modelo': keras_model, 'hist': history}

# ------------------------------------------------------
"""
    Nombre: entrenar_centralizado
    Descripción: Entrena un modelo centralizado tradicional (no federado) con Keras.
    Devuelve: Diccionario con el modelo entrenado.
"""

def entrenar_centralizado(X_train, y_train, modelo, epochs=5):
    modelo.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005), 
                   loss='binary_crossentropy',
                   metrics=['binary_accuracy'])
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    return {'modelo': modelo}

# ------------------------------------------------------
"""
    Nombre: cptDI
    Descripción: Calcula el Disparate Impact (DI) entre un grupo sensible S y la variable de salida Y.
    Devuelve: Valor numérico del Disparate Impact.
"""
def cptDI(S, Y):
    pi_1 = np.mean(S)
    pi_0 = 1 - pi_1
    p_1 = np.mean(S * Y)
    p_0 = np.mean((1 - S) * Y)
    if (pi_0 == 0) or (pi_1 == 0) or (p_0 == 0) or (p_1 == 0):
        return np.nan
    DI = p_0 * pi_1 / (p_1 * pi_0)
    return DI

# ------------------------------------------------------
"""
    Nombre: evaluar_modelo
    Descripción: Evalúa un modelo en términos de precisión y disparidad (disparate impact) con intervalo de confianza.
    Devuelve: Accuracy, Disparate Impact y su intervalo de confianza.
"""

def evaluar_modelo(modelo, X_test, y_test, variable_sensible, X_col_names, alpha=0.05):
    # Predicciones
    y_pred = (modelo.predict(X_test) > 0.5).astype(int).flatten()
    y_true = y_test.flatten()
    acc = np.mean(y_pred == y_true)

    # Definir S según la variable sensible que queremos evaluar
    if variable_sensible == 'Target':
        # Queremos evaluar el DI por Sex, aunque el reparto sea por Target
        idx_sensitive = X_col_names.index("Sex")
        S = X_test[:, idx_sensitive].ravel()
    else:
        # Evaluar por Sex u OrigEthn
        idx_sensitive = X_col_names.index(variable_sensible)
        S = X_test[:, idx_sensitive].ravel()

    # X_ext = [X_test | y_pred] para la función disparate
    X_ext = np.concatenate([X_test, y_pred.reshape(-1, 1)], axis=1)
    Y_idx = X_ext.shape[1] - 1

    # Disparate impact “simple”
    di = cptDI(S, y_pred)

    # Intervalo de confianza del DI
    lower, di_hat, upper, _ = disparate(X_ext, idx_sensitive, Y_idx, alpha)

    return acc, di, (lower, di_hat, upper)


# ------------------------------------------------------
"""
    Nombre: calcular_intervalo_confianza
    Descripción: Calcula un intervalo de confianza del 95% para una muestra de valores numéricos.
    Devuelve: Tupla con los límites inferior y superior del IC.
"""
def calcular_intervalo_confianza(arr):
    ic = stats.t.interval(
        confidence=0.95,
        df=len(arr)-1,
        loc=np.mean(arr), scale=stats.sem(arr)
    )
    return ic

# ------------------------------------------------------
"""
    Nombre: h
    Descripción: Función auxiliar usada en el cálculo del disparate impact.
    Devuelve: Resultado de la fórmula h(x).
"""

def h(x):
    return x[0] * x[3] / (x[1] * x[2])

"""
    Nombre: grad_h
    Descripción: Calcula el gradiente de la función h(x).
    Devuelve: Vector gradiente correspondiente a x.
"""
def grad_h(x):
    return np.array([
        x[3] / (x[1] * x[2]),
        -x[0] * x[3] / ((x[1]**2) * x[2]),
        -x[0] * x[3] / ((x[2]**2) * x[1]),
        x[0] / (x[1] * x[2])
    ])

"""
    Nombre: disparate
    Descripción: Calcula el Disparate Impact y su intervalo de confianza usando la delta-method.
    Devuelve: Intervalo inferior, valor DI, superior y Balanced Error Rate.
"""
def disparate(X, S, Y, alpha):
    n = X.shape[0]
    if S >= X.shape[1]:
        raise ValueError(f"S must be between 0 and {X.shape[1]-1}")

    pi_1 = np.mean(X[:, S])
    pi_0 = 1 - pi_1
    p_1 = np.mean(X[:, S] * X[:, Y])
    p_0 = np.mean((1 - X[:, S]) * X[:, Y])

    if (pi_0 == 0) or (pi_1 == 0) or (p_0 == 0) or (p_1 == 0):
        return np.nan, np.nan, np.nan, np.nan

    Tn = p_0 * pi_1 / (p_1 * pi_0)

    grad = grad_h(np.array([p_0, p_1, pi_0, pi_1]))
    Cov_4 = np.zeros((4, 4))
    Cov_4[0,1] = -p_0 * p_1
    Cov_4[0,2] = pi_1 * p_0
    Cov_4[0,3] = -pi_1 * p_0
    Cov_4[1,2] = -pi_0 * p_1
    Cov_4[1,3] = pi_0 * p_1
    Cov_4[2,3] = -pi_0 * pi_1
    Cov_4 = Cov_4 + Cov_4.T + np.diag([
        p_0 * (1 - p_0),
        p_1 * (1 - p_1),
        pi_0 * pi_1,
        pi_0 * pi_1
    ])

    sigma = np.sqrt(grad @ Cov_4 @ grad.T)
    z = norm.ppf(1 - alpha/2)
    lower = Tn - (sigma * z) / np.sqrt(n)
    upper = Tn + (sigma * z) / np.sqrt(n)
    BER = 0.5 * (p_0/pi_0 + 1 - p_1/pi_1)

    return lower, Tn, upper, BER

# ------------------------------------------------------
"""
    Nombre: plot_reparto_variable_sensible
    Descripción: Grafica la distribución de una variable sensible por cliente y muestra estadísticas globales.
    Devuelve: No devuelve nada. Muestra un gráfico y estadísticas por consola.
"""

def plot_reparto_variable_sensible(repartos, X_col_names, variable_sensible='Sex'):
    """
    Grafica la distribución de la variable sensible por cliente.
    Funciona para: 'Sex', 'OrigEthn' y 'Target'
    """
    # Definir información específica para cada variable sensible
    if variable_sensible == 'Sex':
        nombres_grupos = ['Mujeres', 'Hombres']
        colores = ['salmon', 'skyblue']
        titulo = 'Distribución de Sexo por cliente'
        grupo0_name = 'Mujeres'
        grupo1_name = 'Hombres'
        
    elif variable_sensible == 'OrigEthn':
        nombres_grupos = ['No Caucásico', 'Caucásico']
        colores = ['lightgreen', 'lightcoral']
        titulo = 'Distribución de Origen Étnico por cliente'
        grupo0_name = 'No Caucásico'
        grupo1_name = 'Caucásico'
        
    elif variable_sensible == 'Target':
        nombres_grupos = ['<=50K', '>50K']
        colores = ['lightblue', 'orange']
        titulo = 'Distribución de Target (Ingresos) por cliente'
        grupo0_name = '<=50K'
        grupo1_name = '>50K'
        
    else:
        nombres_grupos = ['Grupo 0', 'Grupo 1']
        colores = ['gray', 'darkgray']
        titulo = f'Distribución de {variable_sensible} por cliente'
        grupo0_name = 'Grupo 0'
        grupo1_name = 'Grupo 1'
    
    # Inicializar listas para almacenar datos
    clientes = []
    pct_grupo0 = []
    pct_grupo1 = []
    n_grupo0 = []
    n_grupo1 = []
    
    # Calcular estadísticas para cada cliente
    for i, cl in enumerate(repartos):
        Xc = cl['X']
        yc = cl['y']
        n_filas = Xc.shape[0]
        
        if n_filas == 0:
            continue
        
        if variable_sensible == 'Target':
            # Contar porcentaje de y=1 (>50K)
            n_g1 = int(np.sum(yc == 1))
            n_g0 = n_filas - n_g1
            
        else:
            # Para variables sensibles en X
            idx_var = X_col_names.index(variable_sensible)
            n_g1 = int(Xc[:, idx_var].sum())  # 1 = grupo privilegiado
            n_g0 = n_filas - n_g1
        
        clientes.append(f'Cliente_{i+1}')
        pct_grupo0.append(100 * n_g0 / n_filas if n_filas > 0 else 0)
        pct_grupo1.append(100 * n_g1 / n_filas if n_filas > 0 else 0)
        n_grupo0.append(n_g0)
        n_grupo1.append(n_g1)
    
    # Crear gráfico
    x = np.arange(len(clientes))
    bar_width = 0.35
    
    plt.figure(figsize=(max(10, len(clientes) * 0.5), 6))
    
    bars_grupo0 = plt.bar(x - bar_width/2, pct_grupo0,
                         width=bar_width, 
                         label=f'% {grupo0_name}',
                         color=colores[0])
    
    bars_grupo1 = plt.bar(x + bar_width/2, pct_grupo1,
                         width=bar_width,
                         label=f'% {grupo1_name}',
                         color=colores[1])
    
    # Añadir etiquetas con conteos
    for i, (b0, b1) in enumerate(zip(bars_grupo0, bars_grupo1)):
        # Etiqueta para grupo 0
        if n_grupo0[i] > 0:
            plt.text(b0.get_x() + b0.get_width()/2, 
                    b0.get_height()/2,
                    str(n_grupo0[i]),
                    ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
        
        # Etiqueta para grupo 1
        if n_grupo1[i] > 0:
            plt.text(b1.get_x() + b1.get_width()/2,
                    b1.get_height()/2,
                    str(n_grupo1[i]),
                    ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
    
    plt.xticks(x, clientes, rotation=45, ha='right')
    plt.xlabel('Cliente')
    plt.ylabel('Porcentaje (%)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajustar límites del eje Y
    plt.ylim(0, max(max(pct_grupo0 + pct_grupo1), 100) * 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas resumidas
    print(f"\nEstadísticas de distribución de {variable_sensible}:")
    print(f"Total de clientes analizados: {len(clientes)}")
    print(f"Total de muestras analizadas: {sum(n_grupo0) + sum(n_grupo1)}")
    print(f"Proporción global {grupo0_name}: {sum(n_grupo0)/(sum(n_grupo0)+sum(n_grupo1))*100:.1f}%")
    print(f"Proporción global {grupo1_name}: {sum(n_grupo1)/(sum(n_grupo0)+sum(n_grupo1))*100:.1f}%")
    
    # Mostrar clientes con distribuciones extremas
    if len(clientes) > 0:
        print(f"\nClientes con mayor proporción de {grupo1_name}:")
        for idx in np.argsort(pct_grupo1)[-3:][::-1]:
            print(f"  {clientes[idx]}: {pct_grupo1[idx]:.1f}% ({n_grupo1[idx]}/{n_grupo0[idx]+n_grupo1[idx]})")
        
        print(f"\nClientes con menor proporción de {grupo1_name}:")
        for idx in np.argsort(pct_grupo1)[:3]:
            print(f"  {clientes[idx]}: {pct_grupo1[idx]:.1f}% ({n_grupo1[idx]}/{n_grupo0[idx]+n_grupo1[idx]})")


# ------------------------------------------------------
"""
    Nombre: plot_box_metrics
    Descripción: Crea boxplots comparando métricas (accuracy y DI) federado vs centralizado.
    Devuelve: Ruta del fichero de imagen generado.
"""

def plot_box_metrics(accs_fed, di_fed, accs_cent, di_cent,
                     n_reps, nombre_exp):
    # --- datos ---
    acc_data = [accs_fed, accs_cent]
    di_data  = [di_fed,  di_cent]

    # --- figura con 2 subplots lado a lado ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # dos gráficas cuadradas aprox

    # ==================== ACCURACY ====================
    ax_acc = axes[0]
    bp_acc = ax_acc.boxplot(
        acc_data,
        labels=["Fed Accuracy", "Cent Accuracy"],
        widths=0.6,
        patch_artist=True,
        showfliers=True
    )
    for b in bp_acc['boxes']:
        b.set(facecolor='#1f77b4', alpha=0.5)

    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.8, 0.9)
    ax_acc.set_title("Accuracy")
    ax_acc.grid(axis='y', linestyle='--', alpha=0.4)

    # ==================== DISPARATE IMPACT ====================
    ax_di = axes[1]
    bp_di = ax_di.boxplot(
        di_data,
        labels=["Fed DI", "Cent DI"],
        widths=0.6,
        patch_artist=True,
        showfliers=True
    )
    for b in bp_di['boxes']:
        b.set(facecolor='#ff7f0e', alpha=0.5)

    ax_di.set_ylabel("Disparate Impact")
    ax_di.set_ylim(0, 0.4)
    ax_di.set_title("Disparate Impact")
    ax_di.grid(axis='y', linestyle='--', alpha=0.4)

    # --- título global y guardado ---
    fig.suptitle(f"Resultados en las {n_reps} repeticiones ({nombre_exp})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs("Figuras", exist_ok=True)
    ruta = os.path.join("Figuras", f"Resultados_{nombre_exp}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.show()

    return ruta


# ------------------------------------------------------
"""
    Nombre: algoritmo_principal
    Descripción: Ejecuta el flujo completo de una simulación de federated learning con análisis de equidad.
    Devuelve: Diccionario con resultados, intervalos de confianza y rutas de gráficos generados.
"""


def algoritmo_principal(
    VARIABLE_SENSIBLE,
    MODELO,
    PORCENTAJE_TEST,
    N_REPETICIONES,
    NUM_CLIENTES,
    N_ROUNDS,
    MAX_DATOS,
    MODO_SESGO_CLIENTES,
    C_MIN,
    C_MAX,
    MODO_REPARTO,
    FICHERO_PROPORCIONES="proporciones_clientes.csv",
    nombre_exp="Exp"
):
    # 1. Cargar datos
    df = cargar_datos('./adult_dataset/adult.data.csv',
                      './adult_dataset/adult.test.csv')

    # 2. Preprocesado según MODO_REPARTO y VARIABLE_SENSIBLE
    if MODO_REPARTO == 1 and VARIABLE_SENSIBLE == 'Sex':
        df_male   = df[df['Sex'] == 'Male']
        df_fem    = df[df['Sex'] == 'Female']
        df_male_s = df_male.sample(n=5000, random_state=0)
        df_fem_s  = df_fem.sample(n=5000, random_state=0)
        df_balanced = pd.concat([df_male_s, df_fem_s]).sample(
            frac=1, random_state=0).reset_index(drop=True)
        X, y, X_col_names = preprocesar_datos(df_balanced, VARIABLE_SENSIBLE)

    elif MODO_REPARTO == 1 and VARIABLE_SENSIBLE == 'OrigEthn':
        df_white    = df[df['OrigEthn'] == 'White']
        df_nonwhite = df[df['OrigEthn'] != 'White']
        n = 5000
        df_w_s  = df_white.sample(n=n, random_state=0)
        df_nw_s = df_nonwhite.sample(n=n, random_state=0)
        df_balanced = pd.concat([df_w_s, df_nw_s]).sample(
            frac=1, random_state=0).reset_index(drop=True)
        X, y, X_col_names = preprocesar_datos(df_balanced, VARIABLE_SENSIBLE)
    elif MODO_REPARTO == 2:
        df_white    = df[df['Target'] == '>50K']
        df_nonwhite = df[df['Target'] != '>50K']
        n = 5000
        df_w_s  = df_white.sample(n=n, random_state=0)
        df_nw_s = df_nonwhite.sample(n=n, random_state=0)
        df_balanced = pd.concat([df_w_s, df_nw_s]).sample(
            frac=1, random_state=0).reset_index(drop=True)
        X, y, X_col_names = preprocesar_datos(df_balanced, 'Target')
    else:
        X, y, X_col_names = preprocesar_datos(df, VARIABLE_SENSIBLE)

    # 3. Fichero proporciones
    with open(FICHERO_PROPORCIONES, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["repeticion", "cliente", "tipo_sesgo", "proporcion"])

    resultados = []
    hist_losses = []

    # 4. Bucle de repeticiones
    for rep in range(N_REPETICIONES):
        np.random.seed(rep)

        indices = np.random.choice(len(X), size=MAX_DATOS, replace=False)
        X_rep = X[indices]
        y_rep = y[indices]

        C = random.uniform(C_MIN, C_MAX)

        if MODO_REPARTO in (0, 1):
            repartos = repartir_datos(
                X_rep, y_rep, VARIABLE_SENSIBLE,
                modo_reparto=MODO_REPARTO,
                num_clientes=NUM_CLIENTES,
                porc_min=0.01, porc_max=0.99,
                X_col_names=X_col_names
            )
        else:
            repartos = repartir_datos_noiid_target(
                X_rep, y_rep,
                num_clientes=NUM_CLIENTES,
                porc_min=0.05, porc_max=0.95
            )

        # guardar proporciones
        with open(FICHERO_PROPORCIONES, mode='a', newline='') as f:
            writer = csv.writer(f)
            if MODO_REPARTO == 2:
                for i, cl in enumerate(repartos):
                    num_y1 = np.sum(cl['y'])
                    total   = cl['y'].shape[0]
                    prop_y1 = num_y1 / total if total > 0 else 0.0
                    writer.writerow([rep+1, i, "Y=1", f"{prop_y1:.4f}"])
            else:
                info_vs = BINARIZACIONES[VARIABLE_SENSIBLE]
                idx_var = X_col_names.index(VARIABLE_SENSIBLE)
                for i, cl in enumerate(repartos):
                    num_g1 = np.sum(cl['X'][:, idx_var])
                    total  = cl['X'].shape[0]
                    prop_g1 = num_g1 / total if total > 0 else 0.0
                    writer.writerow([rep+1, i, info_vs['nombre_grupo1'],
                                     f"{prop_g1:.4f}"])

        if len(repartos) > NUM_CLIENTES:
            repartos = repartos[:NUM_CLIENTES]

        X_total = np.vstack([cl['X'] for cl in repartos])
        y_total = np.vstack([cl['y'] for cl in repartos])
        X_train, X_test, y_train, y_test = train_test_split(
            X_total, y_total, test_size=PORCENTAJE_TEST
        )

        # federado
        dict_fed = entrenar_federado(
            repartos, VARIABLE_SENSIBLE,
            rounds=N_ROUNDS, c=C,
            X_col_names=X_col_names,
            modo_sesgo=MODO_SESGO_CLIENTES
        )
        acc_fed, di_fed, ic_di_fed = evaluar_modelo(
            dict_fed['modelo'], X_test, y_test, VARIABLE_SENSIBLE, X_col_names
        )
        hist_losses.append([h['Loss'] for h in dict_fed['hist']])

        # centralizado
        modelo_cent = crear_modelo(MODELO, X_train.shape[1])
        dict_cent = entrenar_centralizado(X_train, y_train, modelo_cent, epochs=64)
        acc_cent, di_cent, ic_di_cent = evaluar_modelo(
            dict_cent['modelo'], X_test, y_test, VARIABLE_SENSIBLE, X_col_names
        )

        resultados.append({
            'acc_test_federado': acc_fed,
            'di_test_federado': di_fed,
            'acc_test_centralizado': acc_cent,
            'di_test_centralizado': di_cent,
        })

    # análisis IC
    accs_fed  = [r['acc_test_federado'] for r in resultados]
    di_fed    = [r['di_test_federado'] for r in resultados]
    accs_cent = [r['acc_test_centralizado'] for r in resultados]
    di_cent   = [r['di_test_centralizado'] for r in resultados]

    ic_acc_fed  = calcular_intervalo_confianza(accs_fed)
    ic_di_fed   = calcular_intervalo_confianza(di_fed)
    ic_acc_cent = calcular_intervalo_confianza(accs_cent)
    ic_di_cent  = calcular_intervalo_confianza(di_cent)

    print("Intervalo de confianza Fed Accuracy:", ic_acc_fed)
    print("Intervalo de confianza Fed Disparate Impact:", ic_di_fed)
    print("Intervalo de confianza Centralizado Accuracy:", ic_acc_cent)
    print("Intervalo de confianza Centralizado Disparate Impact:", ic_di_cent)
    
    ruta_box = plot_box_metrics(
        accs_fed, di_fed, accs_cent, di_cent,
        N_REPETICIONES,
        nombre_exp
    )
    
    # plot y guardado
    os.makedirs("Figuras", exist_ok=True)
    plt.figure(figsize=(8, 4))
    for i, loss_rondas in enumerate(hist_losses):
        rounds = np.arange(1, len(loss_rondas) + 1)
        plt.plot(rounds, loss_rondas, alpha=0.4, label=f"Rep {i+1}")
    plt.xlabel("Ronda")
    plt.ylabel("Loss")
    plt.title(f"Convergencia loss ({nombre_exp})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ruta_fig = os.path.join("Figuras", f"Convergencia_{nombre_exp}.png")
    plt.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "resultados": resultados,
        "ic_acc_fed": ic_acc_fed,
        "ic_di_fed": ic_di_fed,
        "ic_acc_cent": ic_acc_cent,
        "ic_di_cent": ic_di_cent,
        "ruta_fig": ruta_fig,
    }



# Binarizaciones y labels por variable sensible
BINARIZACIONES = {
    'Sex': {'valor_positivo': 'Male', 'nombre_grupo1': 'Hombres', 'nombre_grupo0': 'Mujeres'},
    'OrigEthn': {'valor_positivo': 'CaucYes', 'nombre_grupo1': 'CaucYes', 'nombre_grupo0': 'CaucNo'},
    'Target': {'valor_positivo': '>50K', 'nombre_grupo1': '>50K', 'nombre_grupo0': '<=50K'},
    
}