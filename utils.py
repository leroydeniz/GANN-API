import pandas as pd
import numpy as np
from scipy import stats
from io import StringIO, BytesIO
from typing import List, Tuple, Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
import torch.onnx

import onnx
from onnx2pytorch import ConvertModel

from core import genetic_algorithm, NeuralNetwork

from constants import *

# Función para verificar el tipo de archivo csv
def verificar_extension_csv(filename):
        return filename.lower().endswith(('.csv'))

# Función para verificar el tipo de archivo onnx
def verificar_extension_onnx(filename):
        return filename.lower().endswith(('.onnx'))

# Función para verificar que el csv tiene contenido
def verificar_vacio_csv(file):
        if len(file) == 0:
                return True
        else:
                return False

def preprocessing(file:str):
        '''
        Función para verificar que el archivo cumple la estructura de csv y tiene valores numéricos
        '''

        # Convertir la información deserealizada de nuevo a un tipo File
        decFile = StringIO(file)

        # Eliminar header=0 en caso que la primera fila no tenga los nombres de las columnas (depende de la política usada)
        df = pd.read_csv(decFile, sep=',', header=0)

        # Hay que cerrar el archivo
        decFile.close()

        # Verifica que el dataset tenga entre 6 y 100k rows
        total_rows, _ = df.shape
        if total_rows < 6 and total_rows > 100000:
                return None

        # Factorizamos las clases
        df.iloc[:, -1] = pd.factorize(df.iloc[:, -1], sort=True)[0]

        # Elimina todas las columnas que tengan el mismo valor, no aportan información y complejizan el modelo
        df = df[[i for i in df if len(set(df[i]))>1]]

        # Elimina las columnas que tengan la mitad +1 posición a Null
        min_nulls = (total_rows / 2) +1
        df = df.dropna(axis=1, thresh=min_nulls)

        # Para aquellas filas que tengan valores a Null, se eliminan
        df = df.dropna()

        # Elimina los registros duplicados
        df = df.drop_duplicates()

        # Elimina los registros outliers, basándose en el z-score de la columna, en relación con la media de la columna y la desviación estándar
        constrains = df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(np.abs(stats.zscore(x)) - 3) > EPSILON).all(axis=1)
        df.drop(df.index[~constrains], inplace=True)

        # Estandarizar todos los strings a lowercase y convertirlos a valores numéricos
        df = df.applymap(lambda s:s.lower() if type(s) == str else s)
        columns = df.select_dtypes(include=[np.object])
        for c in columns:
                df[c] = pd.factorize(df[c])[0]

        # Randomizar las filas del dataset para evitar datasets ordenados
        df = df.sample(df.shape[0])

        return df

def exportar_onnx(modelo:NeuralNetwork, size_entrada:int, requires_grad:bool = True) -> str:
        # Entrada del modelo
        x = torch.randn(BATCH_SIZE, 1, size_entrada, requires_grad=requires_grad)
        torch_out = modelo(x)

        # Exportar el modelo
        fichero = ""
        with torch.no_grad():
                with BytesIO() as f:
                        torch.onnx.export(
                                modelo,                    # Modelo de ejecución
                                x,                         # Entrada de modelo (o una tupla para múltiples entradas)
                                f,                         # Dónde guardar el modelo (puede ser un archivo o un objeto similar a un archivo)
                                export_params=True,        # Almacenar los pesos de los parámetros entrenados dentro del archivo del modelo
                                opset_version=10,          # La versión ONNX para exportar el modelo a
                                do_constant_folding=True,  # Si ejecutar plegado constante para optimización
                                input_names = ['input'],   # Nombres de entrada del modelo
                                output_names = ['output'], # Nombres de salida del modelo
                                dynamic_axes={'input' : {0 : 'batch_size'},    # Ejes de longitud variable
                                'output' : {0 : 'batch_size'}}
                        )
                        fichero = f.getvalue()
        return fichero

def importar_onnx(onnx_file: BytesIO) -> NeuralNetwork:
        pass

def clasificar(modelo, test_dataset) -> Any:
        pass

def evaluar(modelo, test_dataset) -> Any:
        predicted = clasificar(modelo, test_dataset)

def optimizar(file) -> Tuple[NeuralNetwork, List[float], int]:
        df = preprocessing(file)
        dV = df.iloc[:, :-1]  # Variables de decisión
        dC = df.iloc[:, -1]  # Variables de clase
        entrada, salida = dV.shape[1], dC.nunique(dropna=True)

        train_values = tensor(dV[:int((len(dV))*P_TRAIN)].values)
        train_target = tensor(dC[:int((len(dC))*P_TRAIN)].values)
        test_values = tensor(dV[int((len(dV))*P_TRAIN):].values)
        test_target = tensor(dC[int((len(dC))*P_TRAIN):].values)

        train_tensor = TensorDataset(train_values, train_target)
        test_tensor = TensorDataset(test_values, test_target)

        train_dataloader = DataLoader(train_tensor, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_tensor, batch_size=BATCH_SIZE)

        del train_values, train_target, test_values, test_target

        opt, tst_scores = genetic_algorithm(
                dataset=(train_dataloader, test_dataloader),
                nin_nout=(entrada, salida),
                rango_neuronas=RANGO_NEURONAS,
                rango_capas=RANGO_CAPAS,
                prob_mut_sesgos=PROB_MUT_SESGOS,
                prob_mut_pesos=PROB_MUT_PESOS,
                max_epochs_nn=MAX_EPOCHS_NN,
                prob_cruce=PROB_CRUCE,
                prob_mut_neurona=PROB_MUT_NEURONA,
                prob_mut_capa=PROB_MUT_CAPA,
                tamano_poblacion=TAMANO_POBLACION,
                max_generaciones=MAX_GENERACIONES,
                seed=SEED
        )
        return opt, tst_scores, entrada